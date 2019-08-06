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

def load_scores():
    with open("scores", 'rb') as fp:
        with Timer("load scores"):
            return msgpack.load(fp)

def make_pair(id0, id1):
    return id0 < id1 and (id0, id1) or (id1, id0)

def load_tests(ids):
    with open("testset") as fp:
        test_set = {}
        num_positive = 0
        num_negative = 0
        skipped = set()
        blocked = set()
        for line in fp.readlines():
            id0, id1, confidence = line.strip().split(" ")
            is_dup = confidence == '1'
            if not (ids == id0).any() or not (ids == id1).any():
                skipped.add((id0, id1, is_dup))
            else:
                pair = make_pair(id0, id1)
                if id0 != id1 and pair not in blocked:
                    if pair in test_set:
                        if test_set[pair] != is_dup:
                            blocked.add(pair)
                            was_dup = test_set.pop(pair)
                            if was_dup:
                                num_positive -= 1
                            else:
                                num_negative -= 1
                    else:
                        test_set[pair] = is_dup
                        if is_dup:
                            num_positive += 1
                        elif not is_dup:
                            num_negative += 1
                        else:
                            print("unknown confidence", confidence)
        assert len(test_set) == num_positive + num_negative
        print("skipped", len(skipped), "test pairs because ids were not in corpus")
        print("blocked", len(blocked), "test pairs with contradicting labels:", blocked)
        print(len(test_set), "test cases available")

    return test_set, num_positive, num_negative
