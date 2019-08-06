from collections import namedtuple
from timer import Timer

import json

import msgpack, msgpack_numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from helpers import load_tests, load_scores

msgpack_numpy.patch()

ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp','fp','tn','fn'])

def pct(n):
    return "%i%%" % (n * 100,)

with open("ids", 'rb') as fp:
    with Timer("load ids"):
        ids = msgpack.load(fp)
print("# ids:", len(ids))

test_set, num_positive, num_negative = load_tests(ids)

scores = load_scores()

def evaluate(threshold):
    y_true = list(test_set.values())
    y_pred = [scores[ids==id0][0][ids==id1][0] > threshold for id0, id1 in test_set]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return ConfusionMatrix(tp, fp, tn, fn)

def print_evaluation(eval_):
    print("known dups:", num_positive)
    print("known non-dups:", num_negative)
    print("TP:", eval_.tp)
    print("FN:", eval_.fn)
    print("TN:", eval_.tn)
    print("FP:", eval_.fp)
    print("accuracy:", pct((eval_.tp + eval_.tn) / (num_positive + num_negative)))
    print("misclassification rate:", pct((eval_.fp + eval_.fn) / (num_positive + num_negative)))
    print("true positive rate | sensitivity | recall:", pct(eval_.tp / num_positive))
    print("false positive rate:", pct(eval_.fp / num_negative))
    print("true negative rate | specificity:", pct(eval_.tn / num_negative))
    print("prevalence:", pct(num_positive / (num_positive + num_negative)))

# np.histogram uses last bin as max, to include 1.0 need a bin >1.0
bins = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 420)
hist = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}
for row in tqdm(scores, desc="binning"):
    counts, _ = np.histogram(row, bins)
    for i, c in enumerate(counts):
        hist[bins[i]] += c
print(hist)

with Timer("calculate confusion matrix"):
    thresholds = (20, 30, 40, 50, 60, 70, 80, 90)
    results = {i: evaluate(i) for i in thresholds}

print(results)

for x in thresholds:
    print(x, results[x].tp, results[x].fp, sum(results[x]))
