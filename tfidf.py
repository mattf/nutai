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
            num_docs = len(ids)
        ids = np.array(ids)
    print("missing issues", missing_issue)
    print("missing body", missing_body)
    print("skipped solutions (missing both)", missing_both)
    print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    dictionary = gensim.corpora.Dictionary(tqdm(texts, desc="building dictionary"))
    dictionary.filter_extremes()
    dictionary.compactify()

    with Timer("model build time"):
        tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    vecs = [dictionary.doc2bow(text) for text in tqdm(texts, desc="building index")]
    index = gensim.similarities.SparseMatrixSimilarity(tfidf[vecs], num_features=len(dictionary))

    # TODO: this should be sparse, w/ 37k docs only 2% of scores are non-zero
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
