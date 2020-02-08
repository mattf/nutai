from collections import namedtuple
import json

from gensim.utils import simple_preprocess
import msgpack
import msgpack_numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from mfoops.timer import Timer

msgpack_numpy.patch()


# ==== DATA INPUT =======================================================================

# filename contents: {[solution.id: "xyz", body: "text", issue: "text"]*}
# extra_output: choose whether to return documents tags and raw strings as well
# processor(string) -> (np.array of ids, list of [string*])
def load_texts(filename, extra_output=False, preprocessing=simple_preprocess):
    # TODO: fingerprint docs.json and link to scores
    with open(filename) as fp:
        data = json.load(fp)
        ids = []
        texts = []
        raw_texts = []
        tags = []
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
                texts.append(preprocessing(text))
                if extra_output:
                    raw_texts.append(text)
                    doc_tags = doc['tag'] if 'tag' in doc else []
                    doc_tags += doc['product'] if 'product' in doc else []
                    tags.append(doc_tags)
        ids = np.array(ids)
        num_docs = len(ids)
    print("missing issues", missing_issue)
    print("missing body", missing_body)
    print("skipped solutions (missing both)", missing_both)
    print(len(ids), ":", " ".join(map(str,ids[1:5])), "...", " ".join(map(str,ids[-4:])))

    if extra_output:
        return ids, texts, tags, raw_texts
    else:
        return ids, texts

# filename contents: csv file with three columns, id_a, id_b, label_ab, corresponding to testset duplicate pairs
# docs contents: dict keyed by training set doc ids
# returns lists of lists, each sublist of format [id_a (str), id_b (str), label_ab (int)]
def load_testset(filename, docs):
    with open(filename,"r") as f:
        testset = [line.split() for line in f.read().split("\n")[:-1]]
    testset = [[pair[0],pair[1],int(pair[2])] for pair in testset if str(pair[0]) in docs and str(pair[1]) in docs]
    print("Testset Size:",len(testset))
    return testset
    

def save_ids(ids):
    with open("ids", 'wb') as fp:
        msgpack.dump(ids, fp)

def load_ids():
    with open("ids", 'rb') as fp:
        with Timer("load ids"):
            return msgpack.load(fp)

def save_scores(ids, score_generator):
    # TODO: this should be sparse, w/ 37k docs only 2% of scores are non-zero
    scores = np.ndarray((len(ids), len(ids)), dtype='uint8')
    for i, sim in enumerate(tqdm(score_generator, desc="scoring")):
        scores[i] = sim*100
    print(scores)

    with Timer("saving time"):
        save_ids(ids)
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

# ==== EVALUATION =======================================================================
# n_docs: total number of docs in corpus
# vect_mat: matrix of size (n_docs, vector_size), stack of each document vector
# slice_size: number of documents to compare in one go, set to maximize run-time
#              but ensure the slices aren't too big as to crash
def all_to_all(n_docs, vect_mat, slice_size=1000):
    if 'sims' in globals().keys():
        del sims
    
    sims = np.zeros((n_docs,n_docs),dtype=np.dtype('u1'))
    for slice_idx in tqdm(range(0,n_docs,slice_size)):
        sims[slice_idx:slice_idx+slice_size,:] = cosine_similarity(vect_mat[slice_idx:slice_idx+slice_size],vect_mat)*255
    return sims


ConfusionMatrix = namedtuple('ConfusionMatrix', ['tn','fp','fn','tp'])

def get_score_from_matrix(scores, ids, id0, id1):
    return scores[ids==id0,ids==id1][0]

def get_score_from_docvecs(docvecs, ids, id0, id1):
    score = int((1 - docvecs.distance(id0, id1)) * 100)
    if score < 0:
        return 0
    else:
        return score

def evaluate(scores, test_set, ids, threshold, get_score=get_score_from_matrix):
    y_true = list(test_set.values())
    y_pred = [get_score(scores, ids, id0, id1) > threshold for id0, id1 in test_set]
    return ConfusionMatrix(*confusion_matrix(y_true, y_pred).ravel())
