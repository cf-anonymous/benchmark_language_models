# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import time
import shutil
from typing import Dict, List, Tuple

import numpy as np
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


from reformer_pytorch import Reformer

from transformers.file_utils import (
    WEIGHTS_NAME,
    cached_path,
)
#class ReformerConfig512():
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
        #logger.info("Model weights saved in {}".format(output_model_file))
        

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
        #return (prediction_scores,)#.view(-1, self.config.num_tokens), masked_lm_labels.view(-1)
        #'''
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
