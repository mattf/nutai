import json

from gensim.utils import simple_preprocess
from tqdm.auto import tqdm


# ==== DATA INPUT =======================================================================

def load_docs(filename, process_text=lambda doc: simple_preprocess(combine_issue_and_body(doc))):
    with open(filename) as fp:
        data = json.load(fp)
        docs = {
            doc['solution.id']: doc
            for doc in tqdm(data, desc='loading docs')
            if 'issue' in doc or 'body' in doc
        }
        for doc in tqdm(docs.values(), desc='adding text'):
            doc['text'] = process_text(doc)
    return docs


def simple_preprocess_and_filter_stopwords(stopwords):
    return lambda text: [word for word in simple_preprocess(text) if word not in stopwords]


def combine_issue_and_body(doc):
    return ' '.join(doc.get('issue', []) + doc.get('body', []))


def combine_issue_and_body_filter_labels(doc):
    content = doc.get('issue', []) + doc.get('body', [])
    labels = set(doc.get('tag', []) + doc.get('product', []))
    return ' '.join(line for line in content if line not in labels)


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
