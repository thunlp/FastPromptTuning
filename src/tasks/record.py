# This file defines dataset classes for ReCoRD
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from tasks.data import AbstractDataset
import json
import logging
from tasks.evaluate import accuracy_metric, em_metric, squad_f1_score
from collections import OrderedDict, Counter
import re
import os
import numpy as np

logger = logging.getLogger('{:^17s}'.format("superglue.record"))
file_name_map = {'train': 'train.jsonl', 'valid': 'val.jsonl', 'test': 'test.jsonl'}

class ReCoRDDataset(AbstractDataset):
    """
    Dataset for ReCoRD
    """

    def __init__(self, task_name, dataset_name, data_dir, tokenizer, max_seq_length, max_tgt_length,
                 tune_method='model'):
        if tune_method == 'model':
            self.sentinel = "record"
        else:
            self.sentinel = None
        datapaths = [os.path.join(data_dir, file_name_map[task_name])]
        super().__init__(task_name, dataset_name, datapaths, tokenizer, max_seq_length, max_tgt_length)

    def process_samples_from_single_path(self, datapath):
        """Convert ReCoRD examples to text2text examples.

        ReCoRD contains a passage, query containing a '@placeholder' string, and a set
        of entities that are the possible values of the placeholder. Each train and
        validation example will have a list of answers, any of which would be
        considered correct.
        For example, a typical example from ReCoRD might look like
        {
          'passsage': 'This is the passage.',
          'query': 'A @placeholder is a bird.',
          'entities': ['penguin', 'potato', 'pigeon'],
          'answers': ['penguin', 'pigeon'],
        }
        which this preprocessor would turn into the following two examples:
        {
          'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                    'potato, pigeon passage: This is the passage.',
          'targets': 'penguin',
        }
        and
        {
          'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                    'potato, pigeon passage: This is the passage.',
          'targets': 'potato',
        }
        """
        logger.info(' > Processing {} ...'.format(datapath))
        samples = []
        total = 0
        target_counts = {}
        pattern_1 = re.compile(r'(\.|\?|\!|\"|\')\n@highlight\n')
        pattern_2 = re.compile(r'\n@highlight\n')
        def get_mathed_first(matched):
            # print(matched, matched.group(1))
            return matched.group(1) + ' '
        with open(datapath) as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                line_json = json.loads(line)
                passage = line_json['passage']['text']
                entities = []
                entities_set = set()
                for start_end in line_json['passage']['entities']:
                    start_index = start_end['start']
                    end_index = start_end['end']
                    entity = passage[int(start_index): int(end_index) + 1]
                    if entity not in entities_set:
                        entities.append(entity)
                        entities_set.add(entity)
                entities_text = ', '.join(entities)
                passage = re.sub(pattern_1, get_mathed_first, passage)
                passage = re.sub(pattern_2, '. ', passage)
                end_mark = passage[-1]
                if end_mark not in {'.', '"', "'", '?', '!'}:
                    passage += '.'  # add end token for passage
                for question_json in line_json['qas']:
                    query = question_json['query']
                    answers_set = set()
                    for answer_json in question_json['answers']:
                        answer = answer_json['text']
                        answers_set.add(answer)
                    for answer in answers_set:
                        idx = question_json['idx']
                        assert answer in entities_set, f"{answer} not in {str(entities)}"
                        if self.sentinel is not None:
                            text_list = [self.sentinel, 'query:', query, 'entities:',
                                         entities_text, 'passage:', passage]
                        else:
                            text_list = ['query:', query, 'entities:', entities_text, 'passage:', passage]
                        text = ' '.join(text_list)
                        target = self.extra_id_0 + answer
                        sample = {
                            'text': text,
                            'target': target,
                            'id': total,
                            'idx': idx,
                            'answers': answers_set,
                        }
                        total += 1
                        samples.append(sample)
                        if total % 50000 == 0:
                            logger.info('  > processed {} so far ...'.format(total))
                        if self.sentinel is not None:
                            break  # for finetune, only use one answer
        processed_info = ' >> processed {} samples. '.format(len(samples))
        logger.info(processed_info)
        logger.info("Example:" + str(samples[0]))
        return samples

    def convert_predictions(self, predictions):
        print("init_predictions", predictions[:20])
        new_predictions = []
        answers = []
        for index, p in enumerate(predictions):
            idx = int(self[index]['idx'])
            if len(new_predictions) == idx:
                assert len(new_predictions) == idx
                new_predictions.append(p)
                answers.append(self[index]['answers'])
        return new_predictions, answers

    def evaluate(self, predictions):
        predictions, answers = self.convert_predictions(predictions)
        print(predictions[:10])
        print(answers[:10])
        return_dict = OrderedDict()
        em = em_metric(predictions, answers)
        return_dict['em'] = em
        f1 = []
        for index, p in enumerate(predictions):
            current_f1 = []
            for ground_truth in answers[index]:
                current_f1.append(squad_f1_score(p, ground_truth))
            f1.append(max(current_f1))
        f1 = np.mean(f1) * 100
        return_dict['f1'] = f1
        return return_dict


