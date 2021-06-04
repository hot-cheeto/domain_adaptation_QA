""" Since we also expect the answers in the SQuAD format, we reuse its code """
""" Official evaluation script for v1.1 of the SQuAD dataset. """

from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):

    f1 = 0 
    total = len(prediction)

    for i in range(len(prediction)):
        prediction_tokens = normalize_answer(prediction[i]).split()
        ground_truth_tokens = normalize_answer(ground_truth[i]).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 += (2 * precision * recall) / (precision + recall)

    
    final_f1 = f1 / total

    return final_f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for dp in dataset:
        for qa in dp['qa']:
            total += 1
            if qa['id'] not in predictions:
                message = 'Question id ' + qa['id'] + \
                            ' not present. Will receive score 0.'
                print(message, file=sys.stderr)
                continue
            ground_truths = ['NA'] if len(qa['answers']) == 0 else qa['answers']
            prediction = predictions[qa['id']]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


