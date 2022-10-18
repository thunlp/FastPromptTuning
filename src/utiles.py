# This file defines the function to select ffn neurons and layers of FPT.
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import torch.nn as nn
import torch


def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


def load_tensor_from_csv(filename):
    if filename is None:
        return None
    tensor = []
    with open(filename, 'r') as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            tensor_list = list(map(float, line.split(',')))
            tensor.append(tensor_list)
    return torch.tensor(tensor, dtype=torch.float32)


def sequence_select_layers(num_selected_layers, total_layers, **kwargs):
    assert num_selected_layers <= total_layers
    return list(range(num_selected_layers))


def evenly_select_layers(num_selected_layers, total_layers, **kwargs):
    min_internal_num = (total_layers - num_selected_layers) // (num_selected_layers - 1) + 1
    to_add_one = num_selected_layers - 1 - (total_layers - num_selected_layers) % (num_selected_layers - 1)
    selected_layers = [0]
    for i in range(num_selected_layers - 1):
        if i < to_add_one:
            selected_layers.append(selected_layers[-1] + min_internal_num)
        else:
            selected_layers.append(selected_layers[-1] + min_internal_num + 1)
    return selected_layers


def score_select_layers(num_selected_layers, total_layers, **kwargs):
    scores = kwargs.pop('scores')
    num_encoder_layers = kwargs.get('num_encoder_layers', None)
    if num_encoder_layers is None:
        scores = scores[:total_layers, :]
    else:
        scores = scores[num_encoder_layers:, :]
    top_scores, top_index = torch.topk(scores, k=num_selected_layers, dim=0)
    top_index = top_index.view(-1).tolist()
    top_index.sort()
    return top_index


SELECT_METHOD = {
    'sequence': sequence_select_layers,
    'evenly': evenly_select_layers,
    'score': score_select_layers
}


def sequence_select_ffn(num_selected_neurons, total_neurons, **kwargs):
    assert num_selected_neurons <= total_neurons
    return list(range(num_selected_neurons))


def score_select_ffn(num_selected_neurons, total_neurons, **kwargs):
    scores = kwargs.pop('scores')
    layer_idx = kwargs.pop('layer_idx')
    scores = scores[layer_idx]
    top_scores, top_index = torch.topk(scores, k=num_selected_neurons)
    top_index = top_index.tolist()
    top_index.sort()
    return top_index


NEURON_METHOD = {
    'sequence': sequence_select_ffn,
    'score': score_select_ffn,
}
