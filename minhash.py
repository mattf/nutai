import json
import os
import random
import zlib

import gensim
from gensim.utils import simple_preprocess, iter_windows
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

from timer import Timer

msgpack_numpy.patch()

def generate_shingles(words, count=2, mapper=lambda x: zlib.adler32(x.encode())):
    if len(words) < count:
        return [mapper(" ".join(words))]
    else:
        return map(lambda stride: mapper(" ".join(stride)), iter_windows([words], count))

def generate_hash_funcs(count, max=2**32-1, prime=4294969733):
    def func(a, b, c):
        return lambda x: (a * x + b) % c
    coeffs = random.sample(range(2**32 - 1), count * 2)
    with open("coeffs.json", 'w') as fp:
        json.dump({"prime": prime, "coeffs": coeffs}, fp)
    return [func(coeffs.pop(), coeffs.pop(), prime) for i in range(count)]

def calculate_signature(shingles, hash_funcs):
    return np.array([min(map(hash, shingles)) for hash in hash_funcs])

# this is...
# a = b = range(100); c = d = np.array(range(100))
# 1M iterations
#  np.count_nonzero(c==d) / len(c) ~= 2.6
#  sum(x == y for x, y in zip(a, b)) / len(a) ~= 12.5
#  sum(a[i] == b[i] for i in range(len(a))) / len(a) ~= 31.7
#  count = 0; for i in range(len(a)): if a[i] == b[i]: count += 1; count / len(a) ~= 27.6
# it takes longer to construct a np.array when calculating the signatures, but that cost
# increase is more than made up for in the scoring cost decrease
def approx_jaccard_score(a, b, axis=0):
    return np.count_nonzero(a==b, axis) / len(a)


def __main__():
    sig_len = 97

    seed = os.getenv("SEED")
    random.seed(seed)
    print("using seed:", seed)
    print("signature length:", sig_len)

    with open("solutions.json") as fp:
        data = json.load(fp)
        ids = []
        texts = []
        missing_issue = 0
        missing_body = 0
        missing_both = 0
        for i, doc in enumerate(tqdm(data, desc="loading docs")):
            text = str()
            if 'issue' not in doc:
                missing_issue += 1
            else:
                text += ' '.join(doc['issue'])
            if 'body' not in doc:
                missing_body += 1
            else:
                text += ' '.join(doc['body'])
            if len(text) == 0:
                missing_both += 1
            else:
                ids.append(doc['solution.id'])
                texts.append(gensim.utils.simple_preprocess(text))
        ids = np.array(ids)
        num_docs = len(ids)
    print("missing issues", missing_issue)
    print("missing body", missing_body)
    print("skipped solutions (missing both)", missing_both)
    print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    hash_funcs = list(generate_hash_funcs(sig_len))

    sigs = np.empty((num_docs, sig_len))
    for i, text in enumerate(tqdm(texts, desc="signatures")):
        shingles = list(generate_shingles(text))
        sigs[i] = calculate_signature(shingles, hash_funcs)
    print(sigs)

    scores = np.ndarray((num_docs, num_docs), dtype='uint8')
    for i, sig in enumerate(tqdm(sigs, desc="scoring")):
        scores[i] = approx_jaccard_score(sig, sigs, 1)*100
    print(scores)

    with Timer("saving time"):
        with open("ids", 'wb') as fp:
            msgpack.dump(ids, fp)
        with open("scores", 'wb') as fp:
            msgpack.dump(scores, fp)

if __name__ == "__main__":
    __main__()
