# This file defines the basic evaluation funtions to used for different datasets
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from collections import defaultdict, Counter
from rouge import Rouge


def multirc_em(predictions, labels, dataset: Dataset):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    question_ids = [d['qid'] for d in dataset]
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, labels))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)
    return 100.0 * em


def f1_metric(predictions, labels, dataset: Dataset):
    # predictions_count = Counter(predictions)
    # if len(predictions_count) > 2:
    #     return 0.0
    # for p in predictions_count:
    #     if p not in dataset.labels:
    #         return 0.0
    new_predictions = []
    for p in predictions:
        if p in dataset.labels:
            new_predictions.append(p)
        else:
            new_predictions.append(dataset.pos_label)
    return 100.0 * f1_score(labels, new_predictions, pos_label=dataset.pos_label)


def f1_macro_metric(predictions, labels, dataset: Dataset):
    return 100.0 * f1_score(labels, predictions, average='macro')


def accuracy_metric(predictions, labels, dataset: Dataset):
    count = 0
    assert len(predictions) == len(labels)
    for prediction, label in zip(predictions, labels):
        count += prediction == label
    return count * 100.0 / len(predictions)


def em_metric(predictions, answers):
    em = 0
    for index, p in enumerate(predictions):
        if p in answers[index]:
            em += 1
    return 100.0 * em / len(predictions)


def squad_f1_score(prediction, ground_truth):
    if prediction == 'no answer':
        return 0
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_metric(predictions, answers):
    rouge = Rouge()
    scores = rouge.get_scores(predictions, answers, avg=True)
    return scores
