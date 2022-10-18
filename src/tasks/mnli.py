# This file defines dataset classes for MNLI
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from tasks.data import AbstractDataset
import json
import logging
from tasks.evaluate import accuracy_metric
from collections import OrderedDict, Counter
import os

logger = logging.getLogger('{:^17s}'.format("glue.mnli"))
file_name_map = {'train': 'train.tsv', 'valid': 'dev_matched.tsv', 'test': 'test_matched.tsv'}


class MNLIDataset(AbstractDataset):
    """
    Dataset for MNLI
    """

    def __init__(self, task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length,
                 tune_method='model'):
        if tune_method == 'model':
            self.sentinel = "mnli"
        else:
            self.sentinel = None
        datapaths = [os.path.join(data_dir, file_name_map[task_name])]
        super().__init__(task_name, dataset_name, datapaths, tokenizer, max_seq_length, max_tgt_length)
        self.labels = {'contradiction': self.extra_id_0 + 'contradiction',
                       'entailment': self.extra_id_0 + 'entailment',
                       'neutral': self.extra_id_0 + 'neutral'}

    def process_samples_from_single_path(self, datapath):
        """"Implement abstract method."""
        logger.info(' > Processing {} ...'.format(datapath))
        samples = []
        total = 0
        target_counts = {}
        with open(datapath) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                if linenum == 0:
                    continue
                line_list = line.split('\t')
                sentence1 = line_list[8]
                sentence2 = line_list[9]
                idx = int(line_list[0])
                if self.sentinel is not None:
                    text_list = [self.sentinel, 'hypothesis:', sentence2, "premise:", sentence1]
                    # text_list = [sentence2, '?', self.extra_id_0, '.', sentence1]
                else:
                    text_list = ['hypothesis:', sentence2, "premise:", sentence1]
                text = ' '.join(text_list)
                target = self.extra_id_0 + line_list[-1]
                sample = {
                    'text': text,
                    'target': target,
                    'id': total,
                    'idx': idx,
                }
                try:
                    target_counts[target] += 1
                except KeyError as e:
                    target_counts[target] = 1
                total += 1
                samples.append(sample)
                if total % 50000 == 0:
                    logger.info('  > processed {} so far ...'.format(total))
        processed_info = ' >> processed {} samples. '.format(len(samples))
        for label in target_counts:
            processed_info += '{:s}: {} '.format(label, target_counts[label])
        logger.info(processed_info)
        logger.info("Example:" + str(samples[0]))
        return samples

    def convert_predictions(self, predictions):
        print("init_predictions", predictions[:10])
        new_predictions = []
        for p in predictions:
            new_predictions.append(self.extra_id_0 + p)
        count = Counter(new_predictions)
        if len(count) < 5:
            print(count)
        else:
            print(f"Total {len(count)} kinds predictions.")
        return new_predictions

    def evaluate(self, predictions):
        predictions = self.convert_predictions(predictions)
        labels = [d['target'] for d in self]
        return_dict = OrderedDict()
        acc = accuracy_metric(predictions, labels, self)
        return_dict['acc'] = acc
        return return_dict


