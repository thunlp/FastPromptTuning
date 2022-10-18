# This file defines the match between different datasets and their Python Classes
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from .record import ReCoRDDataset
from .squad import SQuADDataset
from .mnli import MNLIDataset
from .qqp import QQPDataset
from .general import GeneralDataset, GeneralGenerationDataset

DATASETS = {
    'record': ReCoRDDataset,
    'squad': SQuADDataset,
    'mnli': MNLIDataset,
    'qqp': QQPDataset,
    'xsum': GeneralGenerationDataset,
}
