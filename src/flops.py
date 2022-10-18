# This file is for flops calculation of different FPT methods
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
from t5 import (
    T5Model,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    T5Config
)
from functools import partial
from ptflops import get_model_complexity_info
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name_or_path = "model-path"
tokenizer_name_or_path = "huggingface-tokenizer-path"


def _input_constructor(input_shape, tokenizer):
    max_length = input_shape[1]
    max_tgt_length = input_shape[2]

    inp_seq = ["mnli hypothesis: Product and geography are what make cream skimming work. \
               premise: Conceptually cream skimming has two basic dimensions - product and geography."]
    inputs = tokenizer.batch_encode_plus(
        inp_seq,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    assert input_ids.shape[1] == max_length
    assert attention_mask.shape[1] == max_length
    out_seq = [' '.join(["<extra_id_0>"] * (max_tgt_length - 7))] * input_shape[0]
    outputs = tokenizer.batch_encode_plus(
        out_seq,
        max_length=max_tgt_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    labels = outputs['input_ids'].masked_fill_(outputs['input_ids'] == tokenizer.pad_token_id, -100)
    decoder_attention_mask = outputs['attention_mask']
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "labels": labels.cuda(),
        "decoder_attention_mask": decoder_attention_mask.cuda()
    }
    print([(k, v.size()) for k, v in inputs.items()])
    return inputs


def cal_plm_flops_with_ptflops(path, tokenizer_name, model_class, tok_class, batch_size, max_seq_length, max_tgt_length):
    tok = tok_class.from_pretrained(tokenizer_name)
    for num_layer, d_ff in zip([18], [4]):
        config = T5Config.from_pretrained(path)
        config.num_layers = num_layer
        if max_tgt_length != 5:
            config.num_decoder_layers = num_layer
        print(f"encoder layer is {config.num_layers}")
        print(f"decoder layer is {config.num_decoder_layers}")
        config.d_ff = int(config.d_ff * d_ff / 4)
        print(f"dff is {config.d_ff}")
        model = model_class(config)
        for n, p in model.named_parameters():
            p.requir_grad = False
        model.cuda()
        flops_count, params_count = get_model_complexity_info(
            model,
            (batch_size, max_seq_length, max_tgt_length),
            as_strings=True,
            input_constructor=partial(_input_constructor, tokenizer=tok),
            print_per_layer_stat=False
        )
        print("%s | %s | %s" % ("[ptflops]", "Params(M)", "FLOPs(G)"))
        print("Model:  {}".format(model_class.__name__))
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops_count))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_count))
        model.cpu()


if __name__ == '__main__':
    batch_size = 1
    max_seq_length = 512
    max_tgt_length = 5

    cal_plm_flops_with_ptflops(model_name_or_path, tokenizer_name_or_path,
                               T5ForConditionalGeneration, T5Tokenizer, batch_size, max_seq_length, max_tgt_length)


