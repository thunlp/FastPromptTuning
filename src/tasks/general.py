# This file defines some general dataset classes which can be used for some uniform formats inputs.
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from tasks.data import AbstractDataset
import json
import logging
from tasks.evaluate import accuracy_metric, rouge_metric
from collections import OrderedDict, Counter
import os


class GeneralDataset(AbstractDataset):
    """
    Dataset for OtherDataset
    """

    def __init__(self, task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length,
                 tune_method='model'):
        if tune_method == 'model':
            self.sentinel = dataset_name
        else:
            self.sentinel = None
        self.logger = logging.getLogger('{:^17s}'.format(dataset_name))
        self.file_name_map = {
            'train': f"{dataset_name}_train.tsv",
            'valid': f"{dataset_name}_dev.tsv",
            'test': f"{dataset_name}_test.tsv",
        }
        datapaths = [os.path.join(data_dir, self.file_name_map[task_name])]
        super().__init__(task_name, dataset_name, datapaths, tokenizer, max_seq_length, max_tgt_length)

    def process_samples_from_single_path(self, datapath):
        """"Implement abstract method."""
        self.logger.info(' > Processing {} ...'.format(datapath))
        samples = []
        total = 0
        target_counts = {}
        with open(datapath) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                line_list = line.split('\t')
                sentences = line_list[0].split('[SEP]')
                sentences = [s.strip() for s in sentences]
                idx = total
                if self.sentinel is not None:
                    text_list = [self.sentinel]
                    text_list.extend(sentences)
                else:
                    text_list = sentences
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
                    self.logger.info('  > processed {} so far ...'.format(total))
        processed_info = ' >> processed {} samples. '.format(len(samples))
        if len(target_counts) <= 10:
            for label in target_counts:
                processed_info += '{:s}: {} '.format(label, target_counts[label])
        self.logger.info(processed_info)
        self.logger.info("Example:" + str(samples[0]))
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


class GeneralGenerationDataset(GeneralDataset):
    def __init__(self, task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length,
                 tune_method='model'):
        super().__init__(task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length, tune_method)

    def evaluate(self, predictions):
        labels = [d['target'][12:] for d in self]  # magic number to remove <extra_id_0>
        cnt = 0
        new_p = []
        new_l = []
        for index, p in enumerate(predictions):
            if len(p) == 0:
                continue
            new_p.append(predictions[index])
            new_l.append(labels[index])
        scores = rouge_metric(predictions, labels)
        return_dict = OrderedDict()
        for s in scores:
            return_dict[s] = scores[s]['f']
        return return_dict


