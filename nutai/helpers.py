from collections import namedtuple
import json

from gensim.utils import simple_preprocess
from mfoops.timer import Timer
import msgpack
import msgpack_numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

msgpack_numpy.patch()


# ==== DATA INPUT =======================================================================

def load_docs(filename, process_text=simple_preprocess):
    with open(filename) as fp:
        data = json.load(fp)
        docs = {
            doc['solution.id']: doc
            for doc in tqdm(data, desc='loading docs')
            if 'issue' in doc or 'body' in doc
        }
        for doc in tqdm(docs.values(), desc='adding text'):
            doc['text'] = process_text(''.join(doc.get('issue', '')) + ' ' + ''.join(doc.get('body', '')))
    return docs


def simple_preprocess_and_filter_stopwords(stopwords):
    return lambda text: [word for word in simple_preprocess(text) if word not in stopwords]


# filename contents: {[solution.id: "xyz", body: "text", issue: "text"]*}
# extra_output: choose whether to return documents tags and raw strings as well
# text_filter(string) -> bool, to filter out elements in list of input documents
# processor(string) -> (np.array of ids, list of [string*])
def load_texts(filename, verbose=True, extra_output=False, text_filter=None, preprocessing=simple_preprocess):
    # TODO: fingerprint docs.json and link to scores

    if text_filter is None:
        text_filter = lambda x: False

    with open(filename) as fp:
        data = json.load(fp)
        ids = []
        texts = []
        raw_texts = []
        tags = []
        missing_issue = 0
        missing_body = 0
        missing_both = 0
        iterator = tqdm(data, desc="loading docs") if verbose else data
        for i, doc in enumerate(iterator):
            raw_text, text = "", []
            question, answer = [], []
            if 'issue' not in doc:
                missing_issue += 1
            else:
                raw_text += ' '.join(doc['issue'])
                filt_question = ' '.join([x for x in doc['issue'] if not text_filter(x)])
                question = preprocessing(filt_question)
                text += question

            if 'body' not in doc:
                missing_body += 1
            else:
                raw_text += "\n===\n" + " ".join(doc['body'])
                filt_answer = ' '.join([x for x in doc['body'] if not text_filter(x)])
                answer = preprocessing(filt_answer)
                text += answer

            if len(text) == 0:
                missing_both += 1
            else:
                ids.append(doc['solution.id'])
                texts.append({'text': text,
                              'question': question,
                              'answer': answer})
                if extra_output:
                    raw_texts.append({'raw': raw_text,
                                      'filt': filt_question + filt_answer,
                                      'real_raw': doc.get('issue', []) + ["==="] + doc.get('body', []),
                                      })
                    doc_tags = doc['tag'] if 'tag' in doc else []
                    doc_tags += doc['product'] if 'product' in doc else []
                    tags.append(doc_tags)
        ids = np.array(ids)

    if verbose:
        print("missing issues", missing_issue)
        print("missing body", missing_body)
        print("skipped solutions (missing both)", missing_both)
        print(len(ids), ":", " ".join(map(str, ids[1:5])), "...", " ".join(map(str, ids[-4:])))

    if extra_output:
        return ids, texts, tags, raw_texts
    else:
        return ids, texts


# filename contents: csv file with three columns, id_a, id_b, label_ab, corresponding to testset duplicate pairs
# docs contents: dict keyed by training set doc ids
# returns set of tuples, each in the format: (id_a (str), id_b (str), label_ab (int))
def load_testset(filename, docs, verbose=True):
    with open(filename, "r") as f:
        testset = [line.split() for line in f.read().split("\n")[:-1]]
    filt_testset = set()
    for pair in testset:
        if str(pair[0]) in docs and str(pair[1]) in docs:
            filt_testset.add((pair[0], pair[1], int(pair[2])))
    if verbose:
        print("Testset Size:", len(filt_testset))
    return filt_testset


# ==== EVALUATION =======================================================================
# n_docs: total number of docs in corpus
# vect_mat: matrix of size (n_docs, vector_size), stack of each document vector
# slice_size: number of documents to compare in one go, set to maximize run-time
#              but ensure the slices aren't too big as to crash
def all_to_all(n_docs, vect_mat, slice_size=1000):
    if 'sims' in globals().keys():
        del sims

    sims = np.zeros((n_docs, n_docs), dtype=np.dtype('u1'))
    for slice_idx in tqdm(range(0, n_docs, slice_size)):
        sims[slice_idx:slice_idx + slice_size, :] = cosine_similarity(vect_mat[slice_idx:slice_idx + slice_size],
                                                                      vect_mat) * 255
    return sims


ConfusionMatrix = namedtuple('ConfusionMatrix', ['tn', 'fp', 'fn', 'tp'])
