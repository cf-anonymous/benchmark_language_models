
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM
from pytorch_pretrained_bert.modeling import gelu
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np
from random import randint

from reformer_pytorch import Reformer

from transformers.file_utils import (
    WEIGHTS_NAME,
    cached_path,
)

class ReformerConfig(): 
    def __init__(self, n_hashes=4, num_labels=2):
        self.num_tokens = 30522 
        self.dim = 512#512
        self.depth = 8
        self.max_seq_len = 512#2**14 
        self.heads = 8
        self.bucket_size = 64
        self.n_hashes = n_hashes
        self.add_local_attn_hash = False
        self.ff_chunks = 100
        self.attn_chunks = 1
        self.causal = False
        self.weight_tie = False
        self.lsh_dropout = 0.1
        self.ff_dropout = 0.1
        self.post_attn_dropout = 0.1
        self.layer_dropout = 0.1
        self.final_dropout = 0.1
        self.random_rotations_per_head = False
        self.twin_attention = False
        self.use_scale_norm = False
        self.use_full_attn = False

        #0 means LSH, large means full attention
        #always LSH if you hash
        self.full_attn_thres = 0 if n_hashes > 0 else 1e9#self.max_seq_len
        #0 means always reverse, large means normal backprop
        #always reverse if you hash, never reserve on full_att
        self.reverse_thres = 0 #if n_hashes > 0 else 1e9#self.max_seq_len
        self.num_mem_kv = 0
        self.ff_mult = 4
        self.one_value_head = False
        self.emb_dim = None
        self.return_embeddings = False
        self.config = self
        self.num_labels = num_labels
        

    def from_pretrained(path, num_labels=2, finetuning_task=None, cache_dir=None):
        return ReformerConfig(n_hashes=0, num_labels=num_labels)

    def save_pretrained(self, save_directory):
        return


    def to_json_string(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

class ReformerConfig16k():
    def __init__(self, num_labels=None):
        self.num_tokens = 30522 
        self.dim = 512#512
        self.depth = 12
        self.max_seq_len = 2**14 
        self.heads = 8
        self.bucket_size = 64
        self.n_hashes = 4
        self.add_local_attn_hash = False
        self.ff_chunks = 100
        self.attn_chunks = 1
        self.causal = False
        self.weight_tie = False
        self.lsh_dropout = 0.1
        self.ff_dropout = 0.1
        self.post_attn_dropout = 0.1
        self.layer_dropout = 0.1
        self.final_dropout = 0.1
        self.random_rotations_per_head = False
        self.twin_attention = False
        self.use_scale_norm = False
        self.use_full_attn = False
        self.full_attn_thres =  0 if self.n_hashes > 0 else 1e9
        self.reverse_thres = 0
        self.num_mem_kv = 0
        self.ff_mult = 1
        self.one_value_head = False
        self.emb_dim = None
        self.return_embeddings = False
        self.config = self
        self.num_labels = num_labels
        

    def from_pretrained(path, num_labels=2, finetuning_task=None, cache_dir=None):

        return ReformerConfig16k(num_labels=num_labels)

    def save_pretrained(self, save_directory):
        return


    def to_json_string(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

class ReformerConfig512():
    def __init__(self, num_labels=None):
        self.num_tokens = 30522 
        self.dim = 768#512
        self.depth = 12
        self.max_seq_len = 512#2**14 
        self.heads = 12
        self.bucket_size = 64
        self.n_hashes = 8
        self.add_local_attn_hash = False
        self.ff_chunks = 100
        self.attn_chunks = 1
        self.causal = False
        self.weight_tie = False
        self.lsh_dropout = 0.1
        self.ff_dropout = 0.1
        self.post_attn_dropout = 0.1
        self.layer_dropout = 0.1
        self.final_dropout = 0.1
        self.random_rotations_per_head = False
        self.twin_attention = False
        self.use_scale_norm = False
        self.use_full_attn = False
        self.full_attn_thres = 0
        self.reverse_thres = 0
        self.num_mem_kv = 0
        self.ff_mult = 4
        self.one_value_head = False
        self.emb_dim = None
        self.return_embeddings = False
        self.config = self
        self.num_labels = num_labels
        

    def from_pretrained(path, num_labels=2, finetuning_task=None, cache_dir=None):

        return ReformerConfig(num_labels=num_labels)

    def save_pretrained(self, save_directory):
        return


    def to_json_string(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
def identity(x):
    return x


class ReformerMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config.dim
        self.config = config
        #self.max_seq_len = config.max_seq_len

        self.token_emb = nn.Embedding(config.num_tokens, emb_dim)
        self.to_model_dim = identity if emb_dim == config.dim else nn.Linear(emb_dim, config.dim)

       
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.final_dropout)
        self.reformer = Reformer(config.dim, config.depth, config.max_seq_len, heads = config.heads, bucket_size = config.config.bucket_size, n_hashes = config.n_hashes, add_local_attn_hash = config.add_local_attn_hash, ff_chunks = config.ff_chunks, attn_chunks = config.attn_chunks, causal = config.causal, weight_tie = config.weight_tie, lsh_dropout = config.lsh_dropout, ff_dropout = config.ff_dropout, post_attn_dropout = 0., layer_dropout = config.layer_dropout, random_rotations_per_head = config.random_rotations_per_head, twin_attention = config.twin_attention, use_scale_norm = config.use_scale_norm, use_full_attn = config.use_full_attn, full_attn_thres = config.full_attn_thres, reverse_thres = config.reverse_thres, num_mem_kv = config.num_mem_kv, one_value_head = config.one_value_head)#, ff_mult = config.ff_mult)
        self.to_logits = nn.Linear(config.dim, config.num_tokens)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def resize_token_embeddings(self, new_num_tokens=None):
        if self.config.num_tokens != new_num_tokens:
            exit("hey, your hardcoded token num is wrong?")

    def from_pretrained(mdir, from_tf=None,config=None,cache_dir=None):
            if config is None:
                config = ReformerConfig()



            model = ReformerMaskedLM(config)
            model_path = os.path.join(mdir, "pytorch_model.bin")
            model.load_state_dict(torch.load(model_path))

            return model
            

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
       
    
        
        x = input_ids
        x = self.token_emb(x)
        x = x + self.pos_emb(x)#.type(x.type())

        x = self.to_model_dim(x)
        x = self.reformer(x)
        x = self.dropout(x)
        prediction_scores = self.to_logits(x)#.transpose(1,2)
        return (prediction_scores,)#.view(-1, self.config.num_tokens), masked_lm_labels.view(-1)
        '''
        outputs = (prediction_scores,) 

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.num_tokens), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        #(loss, all_token_preds)
        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)'''


class ReformerClassification(nn.Module):
    def __init__(self, config):
        super(ReformerClassification, self).__init__()
        self.config = config
        #self.max_seq_len = config.max_seq_len

        self.token_emb = nn.Embedding(config.num_tokens, config.dim)
       
        self.pos_emb = AbsolutePositionalEmbedding(config.dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.final_dropout)
        self.reformer = Reformer(config.dim, config.depth, config.max_seq_len, heads = config.heads, bucket_size = config.config.bucket_size, n_hashes = config.n_hashes, add_local_attn_hash = config.add_local_attn_hash, ff_chunks = config.ff_chunks, attn_chunks = config.attn_chunks, causal = config.causal, weight_tie = config.weight_tie, lsh_dropout = config.lsh_dropout, ff_dropout = config.ff_dropout, post_attn_dropout = 0., layer_dropout = config.layer_dropout, random_rotations_per_head = config.random_rotations_per_head, twin_attention = config.twin_attention, use_scale_norm = config.use_scale_norm, use_full_attn = config.use_full_attn, full_attn_thres = config.full_attn_thres, reverse_thres = config.reverse_thres, num_mem_kv = config.num_mem_kv, one_value_head = config.one_value_head)#, ff_mult = config.ff_mult)
        
        self.num_labels = config.num_labels
        self.classifier1 = torch.nn.Linear(config.dim, config.num_labels)

   

    def from_pretrained(mdir, from_tf=None,config=None,cache_dir=None):
            if config is None:
                config = ReformerConfig()
            #config.num_labels = 2

            model = ReformerClassification(config)
            
            model_path = os.path.join(mdir, "pytorch_model.bin")
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            return model

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        x = input_ids
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.reformer(x)
        x = self.dropout(x[:,0])
        logits = self.classifier1(x)

        outputs = (logits,) 
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        return outputs



class ReformerForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super(ReformerForQuestionAnswering, self).__init__()
        self.config = config
        #self.max_seq_len = config.max_seq_len

        self.token_emb = nn.Embedding(config.num_tokens, config.dim)
        self.pos_emb = AbsolutePositionalEmbedding(config.dim, config.max_seq_len)
        self.type_emb = nn.Embedding(2, config.dim)
        self.dropout = nn.Dropout(config.final_dropout)
        self.reformer = Reformer(config.dim, config.depth, config.max_seq_len, heads = config.heads, bucket_size = config.config.bucket_size, n_hashes = config.n_hashes, add_local_attn_hash = config.add_local_attn_hash, ff_chunks = config.ff_chunks, attn_chunks = config.attn_chunks, causal = config.causal, weight_tie = config.weight_tie, lsh_dropout = config.lsh_dropout, ff_dropout = config.ff_dropout, post_attn_dropout = 0., layer_dropout = config.layer_dropout, random_rotations_per_head = config.random_rotations_per_head, twin_attention = config.twin_attention, use_scale_norm = config.use_scale_norm, use_full_attn = config.use_full_attn, full_attn_thres = config.full_attn_thres, reverse_thres = config.reverse_thres, num_mem_kv = config.num_mem_kv, one_value_head = config.one_value_head)#, ff_mult = config.ff_mult)
        
        self.num_labels = config.num_labels
        self.classifier1 = torch.nn.Linear(config.dim, config.num_labels)


    def from_pretrained(mdir, from_tf=None,config=None,cache_dir=None):
            if config is None:
                config = ReformerConfig()

            model = ReformerForQuestionAnswering(config)
            
            model_path = os.path.join(mdir, "pytorch_model.bin")
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            return model

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        start_positions=None,
        end_positions=None,
    ):

        
        x = input_ids
        x = self.token_emb(x) + self.pos_emb(x) 
        if token_type_ids is not None:
            x = x + self.type_emb(token_type_ids) 
        x = self.reformer(x)
        #x = self.dropout(x[:,0])
        logits = self.classifier1(x)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)



class NormalBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, max_tokens, max_seq_len, cls_token, sep_token,  num_labels=2):
        super(NormalBert, self).__init__(config)
        self.num_labels = num_labels

        self.sequence_len = max_seq_len

        self.cls_token = cls_token
        self.sep_token = sep_token
        
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask, labels =None):
        bs = input_ids.size(0)

        cur_input = self.get_input(input_ids, 0, bs)
        cur_mask  = self.get_mask(input_mask, 0, bs)
        cur_seg  = torch.ones(bs, self.sequence_len, device=input_ids.device)


        _, x    = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        outputs = (logits,) 
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs












from torch.utils.checkpoint import get_device_states, set_device_states
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, depth=None, send_signal = False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = False
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = True
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx
from torch.autograd.function import Function
class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class MyReversibleSequence(nn.Module):
    def __init__(self, blocks, layer_dropout = 0., reverse_thres = 0, send_signal = False):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres

        self.blocks = nn.ModuleList([ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)])
        self.irrev_blocks = None

    def forward(self, x, arg_route = (True, True), **kwargs):
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks

        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}

        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x

        import pdb; pdb.set_trace()
        #for layer in blocks: 


        return _ReversibleFunction.apply(x, blocks, block_kwargs)



import reformer_pytorch as rp

class MyReformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, add_local_attn_hash = False, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_activation = None, ff_mult = 4, post_attn_dropout = 0., layer_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv

        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        get_attn = lambda: rp.reformer_pytorch.LSHSelfAttention(dim, heads, bucket_size, n_hashes, add_local_attn_hash = add_local_attn_hash, causal = causal, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head)
        get_ff = lambda: rp.reformer_pytorch.FeedForward(dim, dropout = ff_dropout, activation = ff_activation, mult = ff_mult)

        if weight_tie:
            get_attn = rp.reformer_pytorch.cache_fn(get_attn)
            get_ff = rp.reformer_pytorch.cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()

            f = rp.reformer_pytorch.WithNorm(norm_type, dim, attn)
            g = rp.reformer_pytorch.WithNorm(norm_type, dim, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = rp.reformer_pytorch.Chunk(ff_chunks, g, along_dim = -2)

            blocks.append(nn.ModuleList([f, g]))

        self.layers = MyReversibleSequence(nn.ModuleList(blocks), layer_dropout = layer_dropout, reverse_thres = reverse_thres)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        arg_route = (True, self.twin_attention)
        x = self.layers(x, arg_route = arg_route, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)


class ReformerClassification2(nn.Module):
    def __init__(self, config):
        super(ReformerClassification2, self).__init__()
        self.config = config
        #self.max_seq_len = config.max_seq_len

        self.token_emb = nn.Embedding(config.num_tokens, config.dim)
       
        self.pos_emb = AbsolutePositionalEmbedding(config.dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.final_dropout)
        self.reformer = MyReformer(config.dim, config.depth, config.max_seq_len, heads = config.heads, bucket_size = config.config.bucket_size, n_hashes = config.n_hashes, add_local_attn_hash = config.add_local_attn_hash, ff_chunks = config.ff_chunks, attn_chunks = config.attn_chunks, causal = config.causal, weight_tie = config.weight_tie, lsh_dropout = config.lsh_dropout, ff_dropout = config.ff_dropout, post_attn_dropout = 0., layer_dropout = config.layer_dropout, random_rotations_per_head = config.random_rotations_per_head, twin_attention = config.twin_attention, use_scale_norm = config.use_scale_norm, use_full_attn = config.use_full_attn, full_attn_thres = config.full_attn_thres, reverse_thres = config.reverse_thres, num_mem_kv = config.num_mem_kv, one_value_head = config.one_value_head)#, ff_mult = config.ff_mult)
        
        self.num_labels = config.num_labels
        self.classifier1 = torch.nn.Linear(config.dim * config.depth, config.num_labels)

   

    def from_pretrained(mdir, from_tf=None,config=None,cache_dir=None):
            if config is None:
                config = ReformerConfig()
            #config.num_labels = 2

            model = ReformerClassification2(config)
            
            model_path = os.path.join(mdir, "pytorch_model.bin")
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            return model

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        x = input_ids
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.reformer(x)
        import pdb; pdb.set_trace()
        x = self.dropout(x[:,0])
        logits = self.classifier1(x)

        outputs = (logits,) 
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        return outputs

        
'''class BertMuliheadedClassifier(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, max_tokens, max_seq_len, cls_token, sep_token, do_pos_encoding, do_min_att,  num_labels=2):
        super(BertMuliheadedClassifier, self).__init__(config)
        self.num_labels = num_labels

        self.sequence_len = max_seq_len

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.do_min_att =  do_min_att
        self.do_pos_encoding = do_pos_encoding

        if do_pos_encoding:
            self.pos_emb = nn.Embedding(max_seq_len, config.hidden_size)

        
        use_rnn = False

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.label_att  = LabelsHeadAttention(num_labels, config.hidden_size, do_min_alpha=do_min_att, use_rnn = use_rnn)

        class_head_size = config.hidden_size * (2 if use_rnn else  1) 
        self.classifiers = nn.ModuleList([nn.Linear(class_head_size , 1) for i in range(num_labels)])

        self.apply(self.init_bert_weights)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)

        cur_input = self.get_input(input_ids, 0, bs)
        cur_mask  = self.get_mask(input_mask, 0, bs)
        cur_seg  = torch.ones(bs, self.sequence_len, device=input_ids.device)


        zs, _    = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)

        #import pdb; pdb.set_trace()

        if self.do_pos_encoding:
            pos_enc = self.pos_emb(torch.arange(zs.shape[1], device=zs.device)).unsqueeze(0)
            zs = zs + pos_enc

        logits = []

        label_attention_output, norms, full_rnn_h = self.label_att(zs)
        for i  in range(self.num_labels):
            x   = self.dropout(label_attention_output[i])
            logits.append(self.classifiers[i](x))


        return torch.cat(logits, dim = 1)



class ReformerGuidedBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, sequence_len, input_len,  num_labels=2, num_tokens=None):
        super(ReformerGuidedBert, self).__init__(config)
        self.num_labels = num_labels

        self.sequence_len = sequence_len
        self.total_input_len = sequence_len * input_len
        
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier3 = torch.nn.Linear(config.hidden_size * 3, num_labels)
        self.classifier4 = torch.nn.Linear(config.hidden_size * 4, num_labels)


        reformer_emb_size = 512#config.hidden_size
        self.token_emb = nn.Embedding(num_tokens, reformer_emb_size)
        self.pos_emb = nn.Embedding(self.total_input_len, reformer_emb_size)
        self.reform = Reformer(
            emb = reformer_emb_size,
            depth = 12,
            max_seq_len = self.total_input_len,
            heads = 8,
            lsh_dropout = 0.1,
            causal = False
        )
        self.reformer_query = torch.nn.Linear(reformer_emb_size, config.hidden_size)
        self.leaky = nn.LeakyReLU(0.2)

        self.att = QueryAttention(config.hidden_size)


        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask):

        if self.train: self.att.rnn.flatten_parameters()

        x = input_ids
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        #x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        reformer_emb = self.reformer_query(torch.tanh(self.reform(x)[:,0] / 8))

        
        input_ids   = input_ids.view(input_ids.size(0), self.sequence_len, -1)
        segment_ids = segment_ids.view(segment_ids.size(0), self.sequence_len, -1)
        input_mask  = input_mask.view(input_mask.size(0), self.sequence_len, -1)

        
        zs = []
        for i in range(self.sequence_len):
            z = self.bert(input_ids[:,i], segment_ids[:,i], input_mask[:,i], output_all_encoded_layers=False)[1]
            zs.append(z)


        zs = torch.stack(zs, dim=1)

        attention_output, norms = self.att(zs, reformer_emb)
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] 
        att_seg = []
        att_mask = [] 

        for batch_index in range(input_ids.size(0)):
            i = norm_index[batch_index]
            att_id.append(input_ids[batch_index,i])
            att_seg.append(segment_ids[batch_index,i])
            att_mask.append(input_mask[batch_index,i])

        attention_z = self.bert(torch.stack(att_id), torch.stack(att_seg), torch.stack(att_mask), output_all_encoded_layers=False)[1]

        x       = torch.cat([attention_z, reformer_emb, attention_output], dim = 1)
        #x       = torch.cat([attention_z, attention_output], dim = 1)
        #x       = torch.cat([attention_output], dim = 1)
        #x       = torch.cat([attention_z, reformer_emb], dim = 1)
        x       = self.dropout(x)

        #logits  = self.classifier8(x)
        logits  = self.classifier3(x)
        
        return logits


class HierarchicalBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, max_tokens, max_seq_len, cls_token, sep_token, token_shift, num_labels=2):
        super(HierarchicalBert, self).__init__(config)
        self.num_labels = num_labels

        self.sequence_len = max_seq_len
        self.max_tokens = max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.token_shift = token_shift

        self.CLS_seg = nn.Embedding(1, config.hidden_size)

        self.CLS_embed = nn.Parameter(torch.randn(config.hidden_size, device='cuda'))

        self.bert = BertModel(config)
        encoder_layer  = nn.TransformerEncoderLayer(d_model= config.hidden_size, nhead=8)
        self.h_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = torch.nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)
        initrange = 0.1
        self.CLS_seg.weight.data.uniform_(-initrange, initrange)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)
        token_shift = self.token_shift
        iters = int(self.max_tokens/token_shift)


        i = 0
        zs = []
        #device=input_ids.device

        for j in range(iters):
            i = j * token_shift
            if i + self.sequence_len - 2 > self.max_tokens: break

            cur_input = self.get_input(input_ids, i, bs)
            cur_mask  = self.get_mask(input_mask, i, bs)
            cur_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device)

            z = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)[1]

            #if i != 0 : 
            z = z.detach()
            zs.append(z)
        #emb_index = torch.zeros(bs, dtype=torch.long).cuda()
        
        #zs = [self.CLS_seg(emb_index)]
        #zs = [self.CLS_embed.repeat(bs,1)]
    
        x = torch.stack(zs, dim = 1)

        x = self.h_transformer(x)

        x       = self.dropout(x[:,0])
        #logits  = self.classifier1(x[:,:2].view(bs, -1))
        logits  = self.classifier1(x)

        return logits


class NewAttention(nn.Module):
    def __init__(self, sent_hidden_size=784, do_min_alpha=True, use_rnn=True):
        super(NewAttention, self).__init__()

        self.use_rnn = use_rnn
        self.rnn = nn.LSTM(sent_hidden_size, sent_hidden_size, batch_first=True , bidirectional=True)

        self.word_attention = nn.Linear(2 * sent_hidden_size, sent_hidden_size)
        self.do_min_alpha = do_min_alpha

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(sent_hidden_size, 1, bias=False)


    def forward(self, x):
        #self.gru.flatten_parameters()
        if self.use_rnn:
            rnn_output, hidden = self.rnn(x)
            # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)

            # Find attention vectors by applying the attention linear layer on the output of the RNN
            att_w = self.word_attention(rnn_output)  # (n_words, att_size)
        else:
            rnn_output = x
            att_w = x

        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(2)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))
        if self.do_min_alpha:
            min_alph    = torch.tensor(1/((word_alphas.size(1)*2)), dtype=torch.float, device=word_alphas.device)
            word_alphas = torch.max(word_alphas, min_alph)
            word_alphas2 = word_alphas / torch.sum(word_alphas, dim=1, keepdim=True)
        else:
            word_alphas2 = word_alphas
        # Find sentence embeddings
        sentences = rnn_output * word_alphas2.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas2, rnn_output

class LabelsHeadAttention(nn.Module):
    def __init__(self, num_labels, sent_hidden_size=784, do_min_alpha=True, use_rnn=True):
        super(LabelsHeadAttention, self).__init__()

        self.use_rnn = use_rnn
        self.rnn = nn.LSTM(sent_hidden_size, sent_hidden_size, batch_first=True , bidirectional=True)

        self.word_attention = nn.Linear(2 * sent_hidden_size, sent_hidden_size)
        self.do_min_alpha = do_min_alpha



        # Word context vector to take dot-product with
        self.word_context_vectors = nn.ModuleList([nn.Linear(sent_hidden_size , 1, bias=False) for i in range(num_labels)])
        self.num_labels = num_labels

    def forward(self, x, rnn_output = None):
        #self.gru.flatten_parameters()
        if self.use_rnn:
            if rnn_output is None:
                rnn_output, hidden = self.rnn(x)
            # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)

            # Find attention vectors by applying the attention linear layer on the output of the RNN
            att_w_ = self.word_attention(rnn_output)  # (n_words, att_size)
        else:
           
            rnn_output = x
            att_w_ = x

        att_w_ = torch.tanh(att_w_)  # (n_words, att_size)

        ret_vec = []
        ret_alphas = []
        for i in range(self.num_labels):
            # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
            att_w = self.word_context_vectors[i](att_w_).squeeze(2)  # (n_words)

            # Compute softmax over the dot-product manually
            # Manually because they have to be computed only over words in the same sentence

            # First, take the exponent
            max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
            att_w = torch.exp(att_w - max_value)  # (n_words)

            # Calculate softmax values as now words are arranged in their respective sentences
            word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))
            if self.do_min_alpha:
                min_alph    = torch.tensor(1/((word_alphas.size(1)*2)), dtype=torch.float, device=word_alphas.device)
                word_alphas = torch.max(word_alphas, min_alph)
                word_alphas2 = word_alphas / torch.sum(word_alphas, dim=1, keepdim=True)
            else:
                word_alphas2 = word_alphas
            # Find sentence embeddings
            sentences = rnn_output * word_alphas2.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
            sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)
            ret_vec.append(sentences)
            ret_alphas.append(word_alphas2)

        return ret_vec, ret_alphas, rnn_output


class QueryAttention(nn.Module):
    def __init__(self, sent_hidden_size=784):
        super(QueryAttention, self).__init__()

        self.rnn = nn.LSTM(sent_hidden_size, sent_hidden_size, batch_first=True , bidirectional=True)

        self.word_attention = nn.Linear(sent_hidden_size, sent_hidden_size)
        self.word_attention2 = nn.Linear(2 * sent_hidden_size, sent_hidden_size)



    def forward(self, x, context):
        #self.gru.flatten_parameters()
        #rnn_output, hidden = self.rnn(x.transpose(0,1))
        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        #att_w = self.word_attention(rnn_output)  # (n_words, att_size)
        att_w = self.word_attention(x)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        #OLD:att_w = self.word_context_vector(att_w).squeeze(2)  # (n_words)
        att_w = torch.bmm(att_w, torch.tanh(context.unsqueeze(2) / 10) ).squeeze(2)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Find sentence embeddings
        sentences = x * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas


class AttentionLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config,  max_tokens, max_seq_len, cls_token, sep_token, token_shift, do_pos_encoding, do_min_att, num_labels=2):
        super(AttentionLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size

        self.sequence_len = max_seq_len
        self.max_tokens = max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.token_shift = token_shift
        self.do_min_att =  do_min_att
        self.do_pos_encoding  = do_pos_encoding

        if do_pos_encoding:
            self.pos_emb = nn.Embedding(int(max_tokens/token_shift), config.hidden_size)

        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        use_rnn = True
        class_head_size = config.hidden_size * (3 if use_rnn else  2)
        self.classifier1 = torch.nn.Linear(config.hidden_size * 3, num_labels)

        self.apply(self.init_bert_weights)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        #self.att = NewAttention(config.hidden_size, use_rnn = False)
        self.att = NewAttention(config.hidden_size, do_min_alpha=do_min_att, use_rnn = True)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)
        token_shift = self.token_shift
        iters = int(self.max_tokens/token_shift)

        self.att.rnn.flatten_parameters()

        i = 0
        zs = []
        processing = True
        #device=input_ids.device



        while processing:
            if input_mask[:,i].sum() == 0: break

            if i + self.sequence_len - 2 > self.max_tokens: 
                i = self.max_tokens - (self.sequence_len - 2)
                processing = False


            cur_input = self.get_input(input_ids, i, bs)
            cur_mask  = self.get_mask(input_mask, i, bs)
            cur_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device)

            z = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)[1].detach()

            zs.append(z)
            i += token_shift
                



        zs = torch.stack(zs, dim = 1)
        if self.do_pos_encoding:
            pos_enc = self.pos_emb(torch.arange(zs.shape[1], device=zs.device))
            zs = zs + pos_enc

            att_pos_encs = []


        attention_output, norms, full_rnn_h = self.att(zs)
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            #j is the index if the max alpha for this batch
            j = norm_index[batch_index]
            if self.do_pos_encoding:
                att_pos_encs.append(pos_enc[j])
            i = j * token_shift
            i = min(i , self.max_tokens - (self.sequence_len - 2))

            att_id.append(self.get_input(input_ids[batch_index].unsqueeze(0), i, 1).squeeze(0))
            att_mask.append(self.get_mask(input_mask[batch_index].unsqueeze(0), i, 1).squeeze(0))

        att_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device, dtype=torch.long)
        attention_z = self.bert(torch.stack(att_id), att_seg, torch.stack(att_mask), output_all_encoded_layers=False)[1]

        if self.do_pos_encoding:
            attention_z = attention_z + torch.stack(att_pos_encs)

        x       = torch.cat([attention_output,attention_z], dim = 1)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits


class MultiLabelHeadAttentionLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config,  max_tokens, max_seq_len, cls_token, sep_token, token_shift, do_pos_encoding, do_min_att, num_labels=2):
        super(MultiLabelHeadAttentionLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size

        self.sequence_len = max_seq_len
        self.max_tokens = max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.token_shift = token_shift
        self.do_min_att =  do_min_att
        self.do_pos_encoding  = do_pos_encoding

        if do_pos_encoding:
            self.pos_emb = nn.Embedding(int(max_tokens/token_shift), config.hidden_size)

        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        use_rnn = False
        self.use_rnn = use_rnn
        class_head_size = config.hidden_size * (5 if use_rnn else  3) 

        self.classifiers = nn.ModuleList([nn.Linear(class_head_size , 1) for i in range(num_labels)])
        self.classifier1 = torch.nn.Linear(config.hidden_size * 3, num_labels)

        self.apply(self.init_bert_weights)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        self.global_att = NewAttention(config.hidden_size, use_rnn = use_rnn)
        self.label_att  = LabelsHeadAttention(num_labels, config.hidden_size, do_min_alpha=do_min_att, use_rnn = use_rnn)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)
        token_shift = self.token_shift
        iters = int(self.max_tokens/token_shift)

        if self.use_rnn:
            self.global_att.rnn.flatten_parameters()
            self.label_att.rnn.flatten_parameters()

        i = 0
        zs = []
        processing = True
        #device=input_ids.device



        while processing:
            if i == self.max_tokens: break
            if input_mask[:,i].sum() == 0: break

            if i + self.sequence_len - 2 > self.max_tokens: 
                i = self.max_tokens - (self.sequence_len - 2)
                processing = False


            cur_input = self.get_input(input_ids, i, bs)
            cur_mask  = self.get_mask(input_mask, i, bs)
            cur_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device)

            z = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)[1].detach()

            zs.append(z)
            i += token_shift
                



        zs = torch.stack(zs, dim = 1)
        if self.do_pos_encoding:
            pos_enc = self.pos_emb(torch.arange(zs.shape[1], device=zs.device))
            zs = zs + pos_enc

            att_pos_encs = []

        #TODO: get a list of attention vectors,

        attention_output, norms, full_rnn_h = self.global_att(zs)
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            #j is the index if the max alpha for this batch
            j = norm_index[batch_index]
            if self.do_pos_encoding:
                att_pos_encs.append(pos_enc[j])
            i = j * token_shift
            i = min(i , self.max_tokens - (self.sequence_len - 2))

            att_id.append(self.get_input(input_ids[batch_index].unsqueeze(0), i, 1).squeeze(0))
            att_mask.append(self.get_mask(input_mask[batch_index].unsqueeze(0), i, 1).squeeze(0))

        att_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device, dtype=torch.long)
        attention_z = self.bert(torch.stack(att_id), att_seg, torch.stack(att_mask), output_all_encoded_layers=False)[1]

        if self.do_pos_encoding:
            attention_z = attention_z + torch.stack(att_pos_encs)

        glob_att = self.dropout(torch.cat([attention_output,attention_z], dim = 1))


        #get local attention 
        logits = []
        label_attention_output, _, _ = self.label_att(zs, full_rnn_h)
        for i  in range(self.num_labels):
            x   = self.dropout(label_attention_output[i])
            x   = torch.cat([glob_att,x], dim = 1)
            logits.append(self.classifiers[i](x))

        
        return torch.cat(logits, dim = 1)


class TransformerAttentionLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config,  max_tokens, max_seq_len, cls_token, sep_token, token_shift, num_labels=2):
        super(TransformerAttentionLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size

        self.sequence_len = max_seq_len
        self.max_tokens = max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.token_shift = token_shift
        
        self.bert = BertModel(config)
        self.bert_layers = 12

        encoder_layer  = nn.TransformerEncoderLayer(d_model= config.hidden_size, nhead=8)
        self.h_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1 = torch.nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.classifier2 = torch.nn.Linear(config.hidden_size , num_labels)

        self.apply(self.init_bert_weights)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        self.att = NewAttention(config.hidden_size)

    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)
        token_shift = self.token_shift
        iters = int(self.max_tokens/token_shift)

        self.att.rnn.flatten_parameters()

        i = 0
        zs = []
        processing = True
        #device=input_ids.device

        while processing:
            if i + self.sequence_len - 2 > self.max_tokens: 
                i = self.max_tokens - (self.sequence_len - 2)
                processing = False

            cur_input = self.get_input(input_ids, i, bs)
            cur_mask  = self.get_mask(input_mask, i, bs)
            cur_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device)

            z = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)[1].detach()

            zs.append(z)
            i += token_shift

        x = torch.stack(zs, dim = 1)

        x = self.h_transformer(x)

        attention_output, norms, full_rnn_h = self.att(x)
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            #j is the index if the max alpha for this batch
            j = norm_index[batch_index]
            i = j * token_shift

            att_id.append(self.get_input(input_ids[batch_index].unsqueeze(0), i, 1).squeeze(0))
            att_mask.append(self.get_mask(input_mask[batch_index].unsqueeze(0), i, 1).squeeze(0))

        att_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device, dtype=torch.long)
        attention_z = self.bert(torch.stack(att_id), att_seg, torch.stack(att_mask), output_all_encoded_layers=False)[1]

        x       = torch.cat([x[:,0],attention_output,attention_z], dim = 1)
        x       = self.classifier1(x)
        x       = gelu(x)
        x       = self.dropout(x)
        logits  = self.classifier2(x)

        
        return logits

class Transformer2AttentionLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config,  max_tokens, max_seq_len, cls_token, sep_token, token_shift, num_labels=2):
        super(Transformer2AttentionLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size

        self.sequence_len = max_seq_len
        self.max_tokens = max_tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.token_shift = token_shift
        
        self.bert = BertModel(config)
        self.bert_layers = 12

        encoder_layer  = nn.TransformerEncoderLayer(d_model= config.hidden_size, nhead=8)
        self.content_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.query_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1 = torch.nn.Linear(config.hidden_size * 3, num_labels)

        self.apply(self.init_bert_weights)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        self.get_query_att   = NewAttention(config.hidden_size)
        self.apply_query_att = QueryAttention(config.hidden_size * 2)


    def get_input(self, input_ids, index, batch_size):
        CLS = self.cls_token 
        SEG = self.sep_token 

        b_cls = torch.tensor([CLS] * batch_size, device=input_ids.device)
        b_input = input_ids[:,index:index+self.sequence_len-2]
        b_SEG = torch.tensor([SEG] * batch_size , device=input_ids.device)
        
        return torch.cat([b_cls, b_input, b_SEG], dim=1)

    def get_mask(self, input_mask, index, batch_size):
        b_mask_1 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        b_mask_2 = input_mask[:,index:index+self.sequence_len-2]
        b_mask_3 = torch.ones(batch_size , dtype=torch.long, device=input_mask.device).view(batch_size,1)
        
        return torch.cat([b_mask_1, b_mask_2, b_mask_3], dim =1)

    def forward(self, input_ids, input_mask):
        bs = input_ids.size(0)
        token_shift = self.token_shift
        iters = int(self.max_tokens/token_shift)

        self.get_query_att.rnn.flatten_parameters()

        i = 0
        zs = []
        processing = True
        #device=input_ids.device

        while processing:
            if i + self.sequence_len - 2 > self.max_tokens: 
                i = self.max_tokens - (self.sequence_len - 2)
                processing = False

            cur_input = self.get_input(input_ids, i, bs)
            cur_mask  = self.get_mask(input_mask, i, bs)
            cur_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device)

            z = self.bert(cur_input, cur_mask, cur_seg, output_all_encoded_layers=False)[1].detach()

            zs.append(z)
            i += token_shift



        zs = torch.stack(zs, dim = 1)

        query_vec, _, _ = self.get_query_att(self.query_transformer(zs))
        x = self.content_transformer(zs)
        x = torch.cat([x,x.clone()],dim=2)

 
        attention_output, norms = self.apply_query_att(x, query_vec)

        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            #j is the index if the max alpha for this batch
            j = norm_index[batch_index]
            i = j * token_shift

            att_id.append(self.get_input(input_ids[batch_index].unsqueeze(0), i, 1).squeeze(0))
            att_mask.append(self.get_mask(input_mask[batch_index].unsqueeze(0), i, 1).squeeze(0))

        att_seg  = torch.ones(bs,self.sequence_len, device=input_ids.device, dtype=torch.long)
        attention_z = self.bert(torch.stack(att_id), att_seg, torch.stack(att_mask), output_all_encoded_layers=False)[1]

        x       = torch.cat([attention_output,attention_z], dim = 1)
        x       = self.dropout(x)
        logits  = self.classifier1(x)

        
        return logits

class Attention2HeadLongBert(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, config, sequence_len, input_len, num_labels=2, num_tokens=None):
        super(Attention2HeadLongBert, self).__init__(config)
        self.num_labels = num_labels
        self.mem_size = config.hidden_size
        self.sequence_len = sequence_len
        self.total_input_len = sequence_len * input_len
        
        self.bert = BertModel(config)
        self.bert_layers = 12
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout25 = torch.nn.Dropout(0.25)
        #self.rnn = nn.GRU( config.hidden_size,  config.hidden_size * 2, bidirectional = True)
        
        self.lstm = nn.LSTM( config.hidden_size,  config.hidden_size * 2, bidirectional = True)
        self.classifier1 = torch.nn.Linear(config.hidden_size * 11, num_labels)
        self.classifier8 = torch.nn.Linear(config.hidden_size * 8, num_labels)
        self.attention1 = torch.nn.Linear(self.sequence_len, 64)
        self.attention2 = torch.nn.Linear(64, 128)
        self.attention3 = torch.nn.Linear(128 + config.hidden_size, 2*config.hidden_size)
        #self.classifier10 = torch.nn.Linear(config.hidden_size * 10, num_labels)
        #self.classifier12_1 = torch.nn.Linear(config.hidden_size * 12, config.hidden_size * 4)
        #self.classifier12_2 = torch.nn.Linear(config.hidden_size * 4, num_labels)
        #self.bn = nn.BatchNorm1d(config.hidden_size * 9)

        self.apply(self.init_bert_weights)
        self.leaky = nn.LeakyReLU(0.2)

        #self.att = DocAttNet(sent_hidden_size=config.hidden_size, doc_hidden_size = self.mem_size, num_classes = num_labels)
        self.att = NewAttention(config.hidden_size)

        reformer_emb_size = 512#config.hidden_size
        self.token_emb = nn.Embedding(num_tokens, reformer_emb_size)
        self.pos_emb = nn.Embedding(self.total_input_len, reformer_emb_size)
        self.reform = Reformer(
            emb = reformer_emb_size,
            depth = 12,
            max_seq_len = self.total_input_len,
            heads = 8,
            lsh_dropout = 0.1,
            causal = False
        )
        self.reformer_query = torch.nn.Linear(reformer_emb_size, config.hidden_size)

        self.reformer_att = QueryAttention(config.hidden_size)


    def forward(self, input_ids, segment_ids, input_mask):
        assert segment_ids.shape[1] == self.total_input_len
        zs = []
        #if self.training:
        #    self.att.rnn.flatten_parameters()
        self.att.rnn.flatten_parameters()
        self.reformer_att.rnn.flatten_parameters()
        x = input_ids
        x = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        #x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        reformer_emb = self.reformer_query(torch.tanh(self.reform(x)[:,0].detach() / 8))

        input_ids   = input_ids.view(input_ids.size(0), self.sequence_len, -1)
        segment_ids = segment_ids.view(segment_ids.size(0), self.sequence_len, -1)
        input_mask  = input_mask.view(input_mask.size(0), self.sequence_len, -1)

       
        rand = randint(1, self.sequence_len) - 1

        for i in range(self.sequence_len):
            z = self.bert(input_ids[:,i], segment_ids[:,i], input_mask[:,i], output_all_encoded_layers=False)[1].detach()
            #if i != rand:
            #    z = z.detach()
            zs.append(z)

        zs1 = torch.stack(zs, dim=1)

        reformer_attention_out, _ = self.reformer_att(zs1, reformer_emb)

        attention_output, norms, full_rnn_h = self.att(zs1)
        h = full_rnn_h.view(self.sequence_len, -1, 2, self.mem_size)
        #TODO: try argmax without first_z
        norm_index = torch.argmax(norms, dim = 1)

        att_id = [] # torch.zeros([input_ids.size(0),256]).cuda()
        att_seg = [] #torch.zeros([input_ids.size(0),256]).cuda()
        att_mask = [] #torch.zeros([input_ids.size(0),256]).cuda()

        for batch_index in range(input_ids.size(0)):
            i = norm_index[batch_index]
            att_id.append( input_ids[batch_index,i])
            att_seg.append(segment_ids[batch_index,i])
            att_mask.append(input_mask[batch_index,i])

        attention_z = self.bert(torch.stack(att_id), torch.stack(att_seg), torch.stack(att_mask), output_all_encoded_layers=False)[1]

        
        #pass norm index into fc_layers
        a = self.leaky(self.attention1(norms))
        a = self.leaky(self.attention2(a))
        #a = self.attention3(torch.cat([a,attention_z], dim = 1))
        a = self.attention3(torch.cat([a,attention_z], dim = 1))

        first_z = self.bert(input_ids[:,0], segment_ids[:,0], input_mask[:,0], output_all_encoded_layers=False)[1].detach()

        #x       = torch.cat([first_z, x[0,:,1], x[1,:,0], x[-1,:,0],attention_output,a], dim = 1)
        #x       = torch.cat([first_z, h[0,:,1], h[1,:,0], h[-1,:,0], attention_output,a], dim = 1)
        x       = torch.cat([first_z, full_rnn_h[1],full_rnn_h[-1], attention_output,a, reformer_attention_out, reformer_emb], dim = 1)
        x       = self.dropout(x)

        #logits  = self.classifier8(x)
        logits  = self.classifier1(x)

        
        return logits



'''