# This file is used for calculating the scores of each neurons and the scores are used in FFN Reduction method.
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import torch

from knowledge_neurons import KnowledgeNeurons
from t5 import T5ForConditionalGeneration
from arguments import parse_args
import argparse
import logging
import os
import sys
from t5 import (
    T5Model,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    T5Config
)
from train import model_provider, dataset_provider
from torch.utils.data import DataLoader
from trainer import Trainer
from tqdm import tqdm

logger = logging.getLogger('score-neuron')

def write_tensor_to_csv(tensor, filename):
    tensor_list = tensor.tolist()
    with open(filename, 'w') as f:
        for i in range(len(tensor_list)):
            current_list = list(map(str, tensor_list[i]))
            f.write(','.join(current_list) + '\n')

def extra_args_provider(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("intergrated-gradient")
    group.add_argument("--sample-size", default=50, type=int, help="Sample size to caculate intergrated gradient.")
    group.add_argument("--attribution-method", type=str, choices=['integrated_grads', 'max_activations', 'abs_activations'],
                       default='integrated_grads')
    group.add_argument("--intergrated-steps", type=int, default=20, help="steps to calculate intergrated gradient.")
    group.add_argument("--intergrated-batch-size", type=int, default=20, help="batch size for intergrated gradient.")
    return parser


if __name__=="__main__":
    args = parse_args(extra_args_provider=extra_args_provider)
    # logger config
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper() if args.rank == 0 else "WARNING",
        stream=sys.stdout,
    )
    trainer = Trainer(args, model_provider, dataset_provider)
    if args.load_prompt is not None:
        trainer.load_checkpoint(args.load_prompt)
    trainer.model.eval()
    valid_dataloader, valid_sampler = trainer.build_dataloader(args, trainer.valid_dataset,
                                                            batch_size=1, shuffle=True) # for random sample
    kn = KnowledgeNeurons(trainer.model, trainer.tokenizer, model_type='t5', device=trainer.device)
    if len(trainer.valid_dataset) < args.sample_size:
        logger.warning(f"Valid dataset size less than {args.sample_size}, use {len(trainer.valid_dataset)} samples for score.")
    sample_size = min(args.sample_size, len(trainer.valid_dataset))
    logger.info("Begin score on {:d} samples ...".format(sample_size))
    total_scores = []
    for step, batch in enumerate(tqdm(valid_dataloader, desc="Calulate attribution score for each neuron", \
                                      total=sample_size)):
        if step >= args.sample_size:
            break
        prepared_data = trainer.prepare_data(batch)
        decoder_input_ids = trainer.get_decoder_input_ids(prepared_data["inputs_embeds"])
        scores = kn.get_scores(
            input_embeb=prepared_data['inputs_embeds'],
            label=prepared_data['labels'],
            attention_mask=prepared_data['attention_mask'],
            decoder_input_id=decoder_input_ids,
            decoder_attention_mask=prepared_data['decoder_attention_mask'],
            target_length=1,
            batch_size=args.intergrated_batch_size,
            steps=args.intergrated_steps,
            attribution_method=args.attribution_method,
            pbar=False,
        )
        total_scores.append(scores.cpu())
    scores = torch.stack(total_scores, dim=0).mean(dim=0)
    logger.info(f"Top 5 score:\n{str(torch.topk(scores, 5)[0])}")
    logger.info(f"Bottom 5 score:\n{str(torch.topk(scores, 5, largest=False)[0])}")
    if args.score_file is not None:
        write_tensor_to_csv(scores, args.score_file)
        logger.info(f"write score to {args.score_file}")
