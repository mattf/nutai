import json

from gensim.utils import simple_preprocess
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

from timer import Timer

msgpack_numpy.patch()

# filename contents: {[solution.id: "xyz", body: "text", issue: "text"]*}
# processor(string) -> (np.array of ids, list of [string*])
def load_texts(filename):
    # TODO: fingerprint docs.json and link to scores
    with open(filename) as fp:
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
                texts.append(simple_preprocess(text))
        ids = np.array(ids)
        num_docs = len(ids)
    print("missing issues", missing_issue)
    print("missing body", missing_body)
    print("skipped solutions (missing both)", missing_both)
    print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    return ids, texts

def save_scores(ids, score_generator):
    # TODO: this should be sparse, w/ 37k docs only 2% of scores are non-zero
    scores = np.ndarray((len(ids), len(ids)), dtype='uint8')
    for i, sim in enumerate(tqdm(score_generator, desc="scoring")):
        scores[i] = sim*100
    print(scores)

    with Timer("saving time"):
        with open("ids", 'wb') as fp:
            msgpack.dump(ids, fp)
        with open("scores", 'wb') as fp:
            msgpack.dump(scores, fp)
