import json
import time

import gensim
import numpy as np

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
    with Timer("load time"):
        with open("docs.json") as fp:
            data = json.load(fp)
            ids = [doc['id'] for doc in data]
            texts = [gensim.utils.simple_preprocess(doc['text']) for doc in data]
        print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    with Timer("dictionary build time"):
        dictionary = gensim.corpora.Dictionary(texts)
        dictionary.filter_extremes()
        dictionary.compactify()

    with Timer("model build time"):
        tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    with Timer("index build time"):
        vecs = [dictionary.doc2bow(text) for text in texts]
        index = gensim.similarities.MatrixSimilarity(tfidf[vecs])

    with Timer("score time"):
        scores = [index[tfidf[dictionary.doc2bow(text)]] for text in texts]

    with Timer("bin time"):
        # np.histogram uses last bin as max, to include 1.0 need a bin >1.0
        bins = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 42)
        hist = {0: 0, .1: 0, .2: 0, .3: 0, .4: 0, .5: 0, .6: 0, .7: 0, .8: 0, .9: 0, 1: 0}
        for row in scores:
            counts, _ = np.histogram((row*10).astype(int)/10, bins)
            for i, c in enumerate(counts):
                hist[bins[i]] += c
    print(hist)

    with open("discovered_dups", "w") as fp:
        threshold = .7
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if threshold < scores[i][j] and scores[i][j] < 1:
                    print(ids[i], ids[j], scores[i][j], file=fp)


if __name__ == "__main__":
    __main__()
