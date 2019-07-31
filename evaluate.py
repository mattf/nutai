from collections import namedtuple
from timer import Timer

import json

import msgpack, msgpack_numpy
import numpy as np
from tqdm import tqdm

msgpack_numpy.patch()

def pct(n):
    return "%i%%" % (n * 100,)

with open("testset") as fp:
    known_dups = set()
    known_nondups = set()
    for line in fp.readlines():
        id0, id1, score = map(float, line.strip().split(" "))
        if id0 != id1:
            pair = (min(id0, id1), max(id0, id1))
            if score == 1:
                known_dups.add(pair)
            elif score == 0:
                known_nondups.add(pair)

with open("ids", 'rb') as fp:
    with Timer("load ids"):
        ids = msgpack.load(fp)
print("# ids:", len(ids))

with open("scores", 'rb') as fp:
    with Timer("load scores"):
        scores = msgpack.load(fp)

def discover(threshold):
    dups = set()
    for i, row in enumerate(tqdm(scores, "discovery {}".format(threshold))):
        hits = np.logical_and(
            np.logical_and(
                row > threshold, row < 100),
                np.array([False]*(i+1) + [True]*(len(row)-i-1)))
        for id_ in ids[hits]:
            dups.add((ids[i], id_))
    return dups

Evaluation = namedtuple('Evaluation',
                        ['num_positive', 'num_negative', 'num_predicted',
                            'true_negatives', 'false_positives',
                            'false_negatives', 'true_positives'])

def evaluate(threshold):
    predicted = discover(threshold)
    return Evaluation(
        num_positive=len(known_dups),
        num_negative=len(known_nondups),
        num_predicted=len(predicted),
        true_negatives=len(known_nondups.difference(predicted)),
        false_positives=len(known_nondups.intersection(predicted)),
        false_negatives=len(known_dups.difference(predicted)),
        true_positives=len(known_dups.intersection(predicted))
    )

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

thresholds = (20, 30, 40, 50, 60, 70, 80, 90, 100)
results = {i: evaluate(i) for i in thresholds}

print(results)

for x in thresholds:
    eval_ = results[x]
    total = eval_.num_positive + eval_.num_negative
    print(x, eval_.true_positives, eval_.false_positives, total)
