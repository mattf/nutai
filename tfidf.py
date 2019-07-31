from timer import Timer

import json
import os
import time

import gensim
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

msgpack_numpy.patch()

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

    scores = np.ndarray((num_docs, num_docs), dtype='uint8')
    for i, sim in enumerate(tqdm(index, desc="scoring")):
        scores[i] = sim*100
    print(scores)

    with Timer("saving time"):
        with open("ids", 'wb') as fp:
            msgpack.dump(ids, fp)
        with open("scores", 'wb') as fp:
            msgpack.dump(scores, fp)

if __name__ == "__main__":
    __main__()
