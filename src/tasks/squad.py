# This file defines dataset classes for SQuAD2.0
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from tasks.data import AbstractDataset
from tasks.evaluate import accuracy_metric, em_metric, f1_metric
import json
import logging
from collections import OrderedDict
import os

LABELS = {0: 'False', 1: 'True'}
logger = logging.getLogger('{:^17s}'.format("squad"))
file_name_map = {'train': 'train-v2.0.json', 'valid': 'dev-v2.0.json'}


class SQuADDataset(AbstractDataset):
    """
    Dataset for SQuAD2.0
    """

    def __init__(self, task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length,
                 tune_method='model'):
        if tune_method == 'model':
            self.sentinel = "squad"
        else:
            self.sentinel = None
        datapaths = [os.path.join(data_dir, file_name_map[task_name])]
        super().__init__(task_name, dataset_name, datapaths, tokenizer, max_seq_length, max_tgt_length)
        self.labels = {'True', 'False'}
        self.pos_label = 'True'

    def process_samples_from_single_path(self, datapath):
        """"Implement abstract method."""
        logger.info(' > Processing {} ...'.format(datapath))
        samples = []
        total = 0
        with open(datapath) as f:
            data_json = json.load(f)
            datas = data_json['data']
            for d_id, d_json in enumerate(datas):
                for p_id, p_json in enumerate(d_json['paragraphs']):
                    context = p_json['context']
                    for q_id, q_json in enumerate(p_json['qas']):
                        question = q_json['question']
                        origin_id = q_json['id']
                        answers = q_json['answers']
                        if len(answers) == 0:
                            target = 'no answer'
                        else:
                            target = answers[0]['text']
                        answers_text = []
                        for a in answers:
                            answers_text.append(self.extra_id_0 + a['text'])
                        if len(answers_text) == 0:
                            answers_text.append(self.extra_id_0 + 'no answer')
                        if self.sentinel is not None:
                            text_list = [self.sentinel, 'question:', question, 'context:', context]
                        else:
                            text_list = ['question:', question, 'context:', context]
                        text = ' '.join(text_list)
                        target = self.extra_id_0 + target
                        sample = {
                            'text': text,
                            'target': target,
                            'id': total,
                            'origin_id': origin_id,
                            'answers': answers_text,
                        }
                        samples.append(sample)
                        total += 1
                        if total % 50000 == 0:
                            logger.info('  > processed {} so far ...'.format(total))
        processed_info = ' >> processed {} samples. '.format(len(samples))
        logger.info(processed_info)
        logger.info("Example:" + str(samples[1]))
        return samples

    def convert_predictions(self, predictions):
        print("init_predictions", predictions[:5])
        new_predictions = []
        for p in predictions:
            new_predictions.append(self.extra_id_0 + p)
        return new_predictions

    def evaluate(self, predictions):
        return_dict = OrderedDict()
        labels = [d['target'] for d in self]
        predictions = self.convert_predictions(predictions)
        acc = accuracy_metric(predictions, labels, self)
        return_dict['acc'] = acc
        answers = [d['answers'] for d in self]
        em = em_metric(predictions, answers)
        return_dict['em'] = em
        bool_predictions = [str(p != (self.extra_id_0 + 'no answer')) for p in predictions]
        bool_labels = [str(d['target'] != (self.extra_id_0 + 'no answer')) for d in self]
        f1 = f1_metric(bool_predictions, bool_labels, self)
        return_dict['f1'] = f1
        return return_dict


