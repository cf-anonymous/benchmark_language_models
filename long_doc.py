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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
import time
import json

from os.path import isfile

from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Sigmoid

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics
from model import  NormalBert 
from model import ReformerConfig16k,ReformerConfig512, ReformerClassification


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)

from collections import Counter
def calculate_class_weights(labels):
    ret = Counter(labels)

    import pdb; pdb.set_trace()

    return ret
   


def get_model(args, num_labels, num_tokens, cls_token,sep_token, token_shift):
    exp_type = args.experiment

    if exp_type == "base":
        model = NormalBert.from_pretrained(args.model_name_or_path, args.max_tokens, args.max_seq_length, cls_token,sep_token, num_labels = num_labels)
        return model

    if exp_type == "reformer512":
        config_class = ReformerConfig512
    elif exp_type == "reformer16k":
        config_class = ReformerConfig16k
    else:
        exit("invalid experiment type of: {}".format(exp_type))

    config = config_class.from_pretrained(
            n_hashes=args.reformer_hashes, 
            num_labels=num_labels, 
        )

    model = ReformerClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config_class.from_pretrained(n_hashes=args.reformer_hashes, num_labels=num_labels),
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    return model



def load_dataset(main_file, args, processor, tokenizer, output_mode, data_type = None):
     # Prepare data loader
    id_file     = main_file + "_ids_nocls.pt"
    mask_file   = main_file + "_mask_nocls.pt"
    label_file  = main_file + "_labels.pt"

    #case1 tensor files exist
    file_exist_count = sum([isfile(id_file), isfile(label_file), isfile(mask_file)])
    if  0 < file_exist_count  < 3:
        exit("Only part of the data is saved as tensor files. Delete those files and try again.")
    elif file_exist_count == 3:
        #import pdb; pdb.set_trace()
        all_input_ids = torch.load(id_file)
        all_masks     = torch.load(mask_file)
        all_label_ids = torch.load(label_file)
        return TensorDataset(all_input_ids, all_masks, all_label_ids)

    #load the mega object 
    if data_type == "train":
        features = convert_examples_to_features(
            processor.get_train_examples(args.data_dir), 
            processor.get_labels(), 
            tokenizer,
            args.max_tokens)
    elif data_type == "test":
        features = convert_examples_to_features(
            processor.get_dev_examples(args.data_dir), 
            processor.get_labels(), 
            tokenizer,
            args.max_tokens)
    elif data_type == "val":
        features = convert_examples_to_features(
            processor.get_val_examples(args.data_dir), 
            processor.get_labels(), 
            tokenizer,
            args.max_tokens)
    else:
        exit(f"invalid data_type {data_type}")

    #parse it carefully into tensor files
    torch.save(torch.tensor([f.input_ids    for f in features], dtype=torch.long), id_file)
    torch.save(torch.tensor([f.input_mask   for f in features], dtype=torch.long), mask_file)
    torch.save(torch.tensor([f.label_id     for f in features], dtype=torch.long), label_file)
    #call this function again!
    return load_dataset(main_file, args, processor, tokenizer, output_mode, data_type=data_type)

def get_batch(args, batch, device, cls_token):
    input_ids, input_mask,  label_ids = batch

    if "reformer" in args.experiment:
        b_cls = torch.tensor([cls_token] * input_ids.size(0))
        input_ids = torch.cat([b_cls, input_ids], dim=1)
        

    
    max_len = input_mask.sum(dim=1).max().item()
    max_len = 2**(max_len - 1).bit_length()
    max_len = min(max_len, args.max_tokens)

    input_ids = input_ids[:,:max_len]
    input_mask = input_mask[:,:max_len]


    input_ids  = input_ids.to(device)
    if args.experiment != "base_reformer":
        input_mask = input_mask.to(device)
    label_ids = label_ids.to(device)

    return input_ids, input_mask,  label_ids

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_model_path", default="", type=str, required=False,
                        help="Bert pretrained saved pytorch model path.")
    parser.add_argument("--reformer_model_path", default=None, type=str, required=False,
                        help="Bert pretrained saved pytorch model path.")
    parser.add_argument("--experiment", default="attention", type=str, required=False,
                        help="4 types: attention, base, long, ablation. "
                        "base: original bert"
                        "long: uses an lstm to keep track of all bert hidden representations, but backprop over the first"
                        "attention: uses an lstm + attention mechanism to backprop over more than the first representation"
                        "ablation: concat all the hidden representations"
                        )
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, required=True)
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--reformer_hashes", default=4, type=int, help="Reformer hash buckets")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_tokens",
                        default=16384,
                        type=int,
                        help="The total tokens for ease of processing")
    parser.add_argument("--token_shift",
                        default=200,
                        type=int,
                        help="")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_pos_encoding",
                        action='store_true',
                        help="train a model with positional coding.")
    parser.add_argument("--do_min_att",
                        action='store_true',
                        help="ensure attention has a minimal alpha.")
    parser.add_argument("--do_truncate",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--max_epochs",
                        default=20,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--warmup_epochs",
                        default=1.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--patience",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--val_split",
                        default=0.05,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    save_args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    cls_token = tokenizer.convert_tokens_to_ids(["[CLS]"])
    sep_token = tokenizer.convert_tokens_to_ids(["[SEP]"])

    model = get_model(args, num_labels, len(tokenizer.vocab), cls_token, sep_token, args.token_shift)
    

    if args.bert_model_path != "":
        print("Loading model from: " + args.bert_model_path)
        if args.do_train:
            pretrained_dict = torch.load(os.path.join(args.bert_model_path,"pytorch_model.bin"))
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if 'classifier1.weight' in pretrained_dict:# and pretrained_dict['classifier1.weight'].shape[0] != num_labels:
                del pretrained_dict['classifier1.weight']
                del pretrained_dict['classifier1.bias']
            '''if 'classifier2.weight' in pretrained_dict and pretrained_dict['classifier2.weight'].shape[0] != num_labels:
                del pretrained_dict['classifier2.weight']
                del pretrained_dict['classifier2.bias']'''
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(args.bert_model_path))



    sig = Sigmoid()
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
   
    loss_fct = CrossEntropyLoss()
    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()


        UC = "" if args.do_lower_case else "UC"
        cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
                    str(task_name),
                    str(args.max_tokens),
                    UC))

        # Prepare data loader
        logger.info("Loading training dataset")
        train_data = load_dataset(cached_train_features_file, args, processor, tokenizer, output_mode, data_type="train")
            
        if args.task_name == "arxiv":
            logger.info("Loading validation dataset")
            cached_val_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(),
                        str(task_name),
                        str(args.max_tokens),
                        UC))
            val_data = load_dataset(cached_val_features_file, args, processor, tokenizer, output_mode, data_type="val")
        else:
            logger.info("Spliting train dataset into validation dataset")
            train_data1, train_data2, train_data3 =  train_data.tensors
            #random.shuffle(train_data)
            rand = torch.randperm(train_data1.shape[0])
            train_data1 = train_data1[rand]
            train_data2 = train_data2[rand]
            train_data3 = train_data3[rand]

            val_size = int(train_data1.shape[0] * args.val_split)
            val_data1 = train_data1[:val_size]
            val_data2 = train_data2[:val_size]
            val_data3 = train_data3[:val_size]

            train_data1 = train_data1[val_size:]
            train_data2 = train_data2[val_size:]
            train_data3 = train_data3[val_size:]

            train_data = TensorDataset(train_data1, train_data2, train_data3)
            val_data = TensorDataset(val_data1, val_data2, val_data3)




        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            val_sampler = RandomSampler(val_data)
        else:
            train_sampler = DistributedSampler(train_data)
            val_sampler = DistributedSampler(val_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size)


        num_train_optimization_steps = (len(train_dataloader)) // args.gradient_accumulation_steps * args.max_epochs

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=(args.warmup_epochs / args.max_epochs),
                                                 t_total=num_train_optimization_steps)


        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=(args.warmup_epochs / args.max_epochs),
                                 t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        #logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_val_loss = 90999990.0
        patience = 0
        val_losses = []
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        e_iter = trange(int(args.max_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        for i, _ in enumerate(e_iter):
            torch.cuda.empty_cache()
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            t_iter = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, t_batch in enumerate(t_iter):
                input_ids, input_mask,  label_ids = get_batch(args, t_batch, device, cls_token)
                outputs = model(input_ids, input_mask, labels=label_ids)
                loss = outputs[0]   # model outputs are always tuple in transformers (see doc)
                
                if n_gpu > 1:
                    loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        # loading gpus takes a while the first iteration, get a better estimate this way
                        if i == 0 and step == 0: 
                            t_iter.start_t = time.time()
                            e_iter.start_t = time.time()

                        acc = np.sum(np.argmax(outputs[1].cpu().detach().numpy(), axis=1) == label_ids.cpu().numpy()) / label_ids.shape[0]
                        t_iter.set_description("loss{0:.3f},acc{1:.3f}".format(loss, acc))
                        tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
                        tb_writer.add_scalar('acc', acc, global_step)

            # input_ids;del input_mask;del label_ids;del outputs
            torch.cuda.empty_cache()
            model.eval()
            val_loss = 0
            out_label_ids = None
            with torch.no_grad():
                for v_batch in tqdm(val_dataloader, desc="valuating"):
                    input_ids, input_mask,  label_ids = get_batch(args, v_batch, device, cls_token)
                    outputs = model(input_ids, input_mask, labels=label_ids)
                    loss = outputs[0]   # model outputs are always tuple in transformers (see doc)
                    
                    if n_gpu > 1:  loss = loss.mean()
                    val_loss += loss.item()
            #del input_ids;del input_mask;del label_ids;del outputs
            val_losses.append(val_loss)
            #end training iter
            if val_loss < best_val_loss and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                best_val_loss = val_loss
                patience = 0
                # Save a trained model, configuration and tokenizer
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                print("best epoch {} loss {}".format(i,best_val_loss))
            else:
                patience+=1
                if patience >= args.patience:
                    break



    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    ### Example:
    #model = model_to_save
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        #torch.save(model_to_save.state_dict(), output_model_file)
        if "reformer" not in args.experiment:
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

            # Load a trained model and vocabulary that you have fine-tuned
            #model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

            # Good practice: save your training arguments together with the trained model
            output_args_file = os.path.join(args.output_dir, 'training_args.bin')
            torch.save(args, output_args_file)
            with open(os.path.join(args.output_dir,'commandline_args.txt'), 'w') as f:
                json.dump(save_args.__dict__, f, indent=2)
    else:
        model = get_model(args, num_labels, len(tokenizer.vocab), cls_token, sep_token, args.token_shift)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)



    ### Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        UC = "" if args.do_lower_case else "UC"
        cached_eval_features_file = os.path.join(args.data_dir, 'dev_{0}_{1}_{2}{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(),
                        str(task_name),
                        str(args.max_tokens),
                        UC))

        logger.info("Loading test dataset")
        eval_data =  load_dataset(cached_eval_features_file, args, processor, tokenizer, output_mode, data_type = "test")
        eval_data_long = []
        eval_data_short = []
        #import pdb; pdb.set_trace()
        '''for item in eval_data:
            if item[1].sum().item() <= args.max_seq_length -2:
                eval_data_short.append(item)
            else:
                eval_data_long.append(item)
            
        eval_data = eval_data_long'''

        logger.info("***** Running evaluation *****")
        #logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # Run prediction for full data
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None

        model.token_shift = args.token_shift
        torch.cuda.empty_cache()
        for t_batch in tqdm(eval_dataloader, desc="Evaluating"):

            input_ids, input_mask,  label_ids = get_batch(args, t_batch, device, cls_token)
            with torch.no_grad():
                outputs = model(input_ids, input_mask, labels = label_ids)
            
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        elif output_mode == "multi_classification":
            preds = preds > .5
        result = compute_metrics(task_name, preds, out_label_ids)

        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        with open(os.path.join(args.output_dir, 'val_loss.txt'), 'w') as f:
            for item in val_losses:
                f.write("%s\n" % item)

        acc = result['acc']
        with open(os.path.join(args.output_dir, "results.csv"), "w") as writer:
            writer.write(f"{args.task_name}, {args.experiment}, {args.model_name_or_path[13:]},{args.learning_rate},{args.reformer_hashes},{acc}\n")

        

       

if __name__ == "__main__":
    main()
