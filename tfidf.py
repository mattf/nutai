import json
import os
import time

import gensim
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

msgpack_numpy.patch()

class Timer:
    def __init__(self, message=None):
        self.message = message
    def __enter__(self):
        self.start = time.process_time()
        return self
    def __exit__(self, *args):
        self.end = time.process_time()
        self.interval = self.end - self.start
        if self.message:
            print(self.message, ":", self.interval)

def __main__():
    # TODO: fingerprint docs.json and link to scores
    with open("docs.json") as fp:
        data = json.load(fp)
        ids = np.zeros((len(data),), dtype=int)
        texts = []
        for i, doc in enumerate(tqdm(data, desc="loading docs")):
            ids[i] = doc['id']
            texts.append(gensim.utils.simple_preprocess(doc['text']))
            num_docs = len(ids)
    print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    dictionary = gensim.corpora.Dictionary(tqdm(texts, desc="building dictionary"))
    dictionary.filter_extremes()
    dictionary.compactify()

    with Timer("model build time"):
        tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    vecs = [dictionary.doc2bow(text) for text in tqdm(texts, desc="building index")]
    index = gensim.similarities.MatrixSimilarity(tfidf[vecs])

    scores = np.ndarray((num_docs, num_docs))
    for i, sim in enumerate(tqdm(index, desc="scoring")):
        scores[i] = sim
    print(scores)

    with Timer("saving time"):
        with open("ids", 'wb') as fp:
            msgpack.dump(ids, fp)
        with open("scores", 'wb') as fp:
            msgpack.dump(scores, fp)

    # np.histogram uses last bin as max, to include 1.0 need a bin >1.0
    bins = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)
    hist = {0: 0, .1: 0, .2: 0, .3: 0, .4: 0, .5: 0, .6: 0, .7: 0, .8: 0, .9: 0, 1: 0}
    for row in tqdm(scores, desc="binning"):
        counts, _ = np.histogram(row.round(3), bins)
        for i, c in enumerate(counts):
            hist[bins[i]] += c
    print(hist)

    with open("discovered_dups", "w") as fp:
        threshold = .7
        for i, row in enumerate(tqdm(scores, desc="discovery")):
            hits = np.logical_and(np.logical_and(row > threshold, row < 1),
                np.array([False]*(i+1) + [True]*(len(row)-i-1)))
            for id_, score in zip(ids[hits], row[hits]):
                print(ids[i], id_, score, file=fp)


if __name__ == "__main__":
    __main__()
