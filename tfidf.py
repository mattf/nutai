import json
import os
import time

import gensim
import numpy as np
from tqdm import tqdm

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
    score_file = "scores.json"
    if not os.path.exists(score_file):
        # TODO: fingerprint docs.json and link to scores
        with open("docs.json") as fp:
            data = json.load(fp)
            data = data[:10000]
            ids = np.zeros((len(data),), dtype=int)
            texts = []
            for i, doc in enumerate(tqdm(data, desc="loading docs")):
                ids[i] = doc['id']
                texts.append(gensim.utils.simple_preprocess(doc['text']))
        print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

        dictionary = gensim.corpora.Dictionary(tqdm(texts, desc="building dictionary"))
        dictionary.filter_extremes()
        dictionary.compactify()

        with Timer("model build time"):
            tfidf = gensim.models.TfidfModel(dictionary=dictionary)

        vecs = [dictionary.doc2bow(text) for text in tqdm(texts, desc="building index")]
        index = gensim.similarities.MatrixSimilarity(tfidf[vecs])

        scores = [sim for sim in tqdm(index, desc="scoring")]
        with Timer("saving time"):
            with open(score_file, "w") as fp:
                json.dump({
                    "ids": ids.tolist(),
                    "scores": [((row*1000).astype(int)/1000).tolist()
                                for row in tqdm(scores, desc="saving scores")]
                    }, fp)
    else:
        with Timer("loading scores time"):
            with open(score_file) as fp:
                data = json.load(fp)
                ids = np.array(data['ids'])
                scores = [np.array(row) for row in tqdm(data['scores'], desc="loading scores")]

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
