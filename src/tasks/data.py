# This file defines abstract dataset classes. The abstract dataset classes if the parent class for different classes.
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from torch.utils.data import Dataset
import logging
from abc import abstractmethod, ABC
from collections import Counter
import numpy as np

logger = logging.getLogger('dataset')


class AbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""

    def __init__(self, task_name, dataset_name, datapaths,
                 tokenizer, max_seq_length, max_tgt_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_tgt_length
        self.extra_id_0 = '<extra_id_0>'
        logger.info(' > building {} dataset for {}:'.format(self.task_name,
                                                            self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        logger.info(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))
        logger.info('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def collator(self, batches):
        keys = list(batches[0].keys())
        new_dict = {}
        # merge dict
        for key in keys:
            new_dict[key] = []
            for b in batches:
                new_dict[key].append(b[key])
        source = self.tokenizer.batch_encode_plus(
            new_dict['text'],
            max_length=self.max_seq_length, 
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        target = self.tokenizer.batch_encode_plus(
            new_dict['target'],
            max_length=self.max_target_length, 
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return_dict = {
            'input_ids': source['input_ids'],
            'attention_mask': source['attention_mask'],
            'labels': target['input_ids'].masked_fill_(target['input_ids'] == self.tokenizer.pad_token_id, -100),
            'decoder_attention_mask': target['attention_mask'],
        }
        new_dict.pop('text')
        new_dict.pop('target')
        for key in new_dict:
            return_dict[key] = new_dict[key]
        return return_dict

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass
    
    @abstractmethod
    def evaluate(self, predictions):
        pass

    def convert_predictions(self, predictions):
        print("init_predictions", predictions[:10])

        def get_label(p):
            return self.extra_id_0 + p
        new_predictions = list(map(get_label, predictions))
        count = Counter(new_predictions)
        if len(count) < 5:
            print(count)
        else:
            print(f"Total {len(count)} kinds predictions.")
        return new_predictions

    def get_max_target_length(self):
        length = []
        for sample in self.samples:
            target = sample['target']
            target_token = self.tokenizer.encode(target)
            length.append(len(target_token))
        return int(np.max(length))

    def set_max_target_length(self, max_target_length):
        self.max_target_length = max_target_length
