from collections import namedtuple
from timer import Timer

import json

import msgpack, msgpack_numpy
import numpy as np
from tqdm import tqdm

msgpack_numpy.patch()

ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp','fp','tn','fn'])

Evaluation = namedtuple('Evaluation',
                        ['num_positive', 'num_negative', 'confusion_matrix'])

def pct(n):
    return "%i%%" % (n * 100,)

def make_pair(id0, id1):
    return id0 < id1 and (id0, id1) or (id1, id0)

with open("ids", 'rb') as fp:
    with Timer("load ids"):
        ids = msgpack.load(fp)
print("# ids:", len(ids))

with open("testset") as fp:
    test_set = {}
    num_positive = 0
    num_negative = 0
    skipped = set()
    for line in fp.readlines():
        id0, id1, confidence = line.strip().split(" ")
        is_dup = confidence == '1'
        if not (ids == id0).any() or not (ids == id1).any():
            skipped.add((id0, id1, is_dup))
        else:
            if id0 != id1:
                pair = make_pair(id0, id1)
                test_set[pair] = is_dup
                if is_dup:
                    num_positive += 1
                elif not is_dup:
                    num_negative += 1
                else:
                    print("unknown confidence", confidence)
    print("skipped", len(skipped), "test pairs because ids were not in corpus")
    print(len(test_set), "test cases available")

with open("scores", 'rb') as fp:
    with Timer("load scores"):
        scores = msgpack.load(fp)

def discover(threshold):
    tp, fp, tn, fn = 0, 0, 0, 0
    for (id0, id1), is_dup in test_set.items():
        prediction = scores[ids == id0][0][ids == id1][0]
        if prediction > threshold: # positive prediction
            if is_dup: # positive actual
                tp += 1
            else: # negative actual
                fp += 1
        else: # negative prediction
            if not is_dup: # negative actual
                tn += 1
            else: # positive actual
                fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)

def evaluate(threshold):
    return Evaluation(
        num_positive=num_positive,
        num_negative=num_negative,
        confusion_matrix=discover(threshold))

def print_evaluation(eval_):
    print("known dups:", eval_.num_positive)
    print("known non-dups:", eval_.num_negative)
    print("discovered dups:", eval_.num_predicted)
    print("TP:", eval_.true_positives)
    print("FN:", eval_.false_negatives)
    print("TN:", eval_.true_negatives)
    print("FP:", eval_.false_positives)
    print("accuracy:", pct((eval_.true_positives + eval_.true_negatives) / (eval_.num_positive + eval_.num_negative)))
    print("misclassification rate:", pct((eval_.false_positives + eval_.false_negatives) / (eval_.num_positive + eval_.num_negative)))
    print("true positive rate | sensitivity | recall:", pct(eval_.true_positives / eval_.num_positive))
    print("false positive rate:", pct(eval_.false_positives / eval_.num_negative))
    print("true negative rate | specificity:", pct(eval_.true_negatives / eval_.num_negative))
    print("precision:", pct(eval_.true_positives / eval_.num_predicted))
    print("prevalence:", pct(eval_.num_positive / (eval_.num_positive + eval_.num_negative)))

# np.histogram uses last bin as max, to include 1.0 need a bin >1.0
bins = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 420)
hist = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
for row in tqdm(scores, desc="binning"):
    counts, _ = np.histogram(row, bins)
    for i, c in enumerate(counts):
        hist[bins[i]] += c
print(hist)

with Timer("calculate confusion matrix"):
    thresholds = (20, 30, 40, 50, 60, 70, 80, 90, 100)
    results = {i: evaluate(i) for i in thresholds}

print(results)

for x in thresholds:
    eval_ = results[x]
    total = eval_.num_positive + eval_.num_negative
    print(x, eval_.confusion_matrix.tp, eval_.confusion_matrix.fp, total, len(test_set), sum(eval_.confusion_matrix))
