# This file is the entrance of main expriments and use the trainer for training and inference process.
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from arguments import parse_args
from t5 import (
    T5Model,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    T5Config
)
import torch.distributed as dist
import os
import logging
import torch
import sys

from tasks import DATASETS
from trainer import Trainer
from tasks import GeneralDataset
import logging

torch_version = torch.__version__
torch_version_float = float('.'.join(torch_version.split('.')[:2]))

logger = logging.getLogger("train")

def safe_barrier(args):
    if torch_version_float < 1.8:
        torch.distributed.barrier()
    else:
        torch.distributed.barrier(device_ids=[args.rank])

def distributed_init(args):
    if args.data_parallel_size > 1:
        # create default process group
        dist.init_process_group("nccl", rank=args.rank, world_size=args.data_parallel_size)
        torch.cuda.set_device(args.rank)

def model_provider(args):
    if args.rank == 0:
        # only the master process download model
        config = T5Config.from_pretrained(args.config_name_or_path, cache_dir=args.cache_dir)
        if args.dropout is not None:
            config.dropout_rate = args.dropout
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path,
        config=config, cache_dir=args.cache_dir)
        if args.add_cross_ffn:
            config.add_cross_ffn = args.add_cross_ffn
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path,
        config=config, cache_dir=args.cache_dir)
        if args.data_parallel_size > 1:
            safe_barrier(args)
    else:
        safe_barrier(args)
        config = T5Config.from_pretrained(args.config_name_or_path, cache_dir=args.cache_dir)
        if args.dropout is not None:
            config.dropout_rate = args.dropout
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path,
        config=config, cache_dir=args.cache_dir)
        if args.add_cross_ffn:
            config.add_cross_ffn = args.add_cross_ffn
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path,
        config=config, cache_dir=args.cache_dir)
    if args.data_parallel_size > 1:
        safe_barrier(args)
    return model, config, tokenizer

def dataset_provider(args, tokenizer):
    train_dataset, valid_dataset, test_dataset = None, None, None
    try:
        dataset_class = DATASETS[args.task]
    except KeyError as e:
        logger.info(f"Can't find specific dataset for task {args.task}, use a general dataset.")
        dataset_class = GeneralDataset
    if args.train:
        train_dataset = dataset_class('train', args.task, args.datapath, tokenizer, args.max_seq_length,
                                       args.max_target_length, args.tune_method)
    valid_dataset = dataset_class('valid', args.task, args.datapath, tokenizer, args.max_seq_length,
                                   args.max_target_length, args.tune_method)
    if args.test:
        test_dataset = dataset_class('test', args.task, args.datapath, tokenizer, args.max_seq_length,
                                      args.max_target_length, args.tune_method)
    return train_dataset, valid_dataset, test_dataset

def main():
    args = parse_args()
    distributed_init(args)
    # logger config
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper() if args.rank == 0 else "WARNING",
        stream=sys.stdout,
    )
    trainer = Trainer(args, model_provider, dataset_provider)
    if args.train:
        if args.load_prompt is not None:
            trainer.load_checkpoint(args.load_prompt, do_eval=False)
        trainer.train()
    else:
        current_state = trainer.load_checkpoint(args.load_prompt)
        trainer.valid(epoch=current_state['epoch'], num_updates=current_state['num_updates'])
    

if __name__=="__main__":
    main()