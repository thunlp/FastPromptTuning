# This file defines the functions about prompt initialization
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import torch
import logging
import random

logger = logging.getLogger('prompt-init')


def uniform_init(prompt, a=0.0, b=1.0):
    torch.nn.init.uniform_(prompt, a, b)
    logger.info("init prompt by uniform [{:.3f}, {:.3f}]".format(a, b))
    return prompt


def embedding_init(prompt, embeddings):
    prompt_len = prompt.shape[0]
    indexes = random.choices(range(3, 32000), k=prompt_len)  # magic number is for T5, remove special token embedding
    prompt_list = embeddings[indexes, :].tolist()
    prompt = torch.tensor(prompt_list, dtype=prompt.dtype, device=prompt.device, requires_grad=True)
    return prompt
