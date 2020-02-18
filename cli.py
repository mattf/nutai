import os

import click
from gensim.corpora import Dictionary
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import msgpack
from minhash import approx_jaccard_score
from nutai.helpers import load_texts, load_docs, load_testset, simple_preprocess_and_filter_stopwords
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import nutai.minhash

def pct(n):
    return "%i%%" % (n * 100,)


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('out', type=click.Path(exists=False))
def generate_stopwords(documents, out):
    _, texts = load_texts(documents)
    dictionary = Dictionary(texts)
    before = set(dictionary.values())
    dictionary.filter_extremes()
    after = set(dictionary.values())
    stopwords = before - after
    with open(out, 'wb') as fp:
        msgpack.dump(list(stopwords), fp)


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('labeled', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-set', type=click.Path(exists=False))
@click.argument('test-set', type=click.Path(exists=False))
@click.option('--seed', default=None, type=click.INT)
def split(documents, labeled, train_set, test_set, seed):
    print(documents, labeled, seed, train_set, test_set)
    all_ids, _ = load_texts(documents)
    all_ids = set(all_ids)
    labels = load_testset(labeled, all_ids)
    labeled_ids = set()
    for id0, id1, _ in labels:
        labeled_ids.add(id0)
        labeled_ids.add(id1)
    train_ids, test_ids = train_test_split(list(all_ids - labeled_ids), test_size=.2, train_size=.8)
    test_ids += labeled_ids
    with open(train_set, 'wb') as fp:
        msgpack.dump(list(train_ids), fp)
    with open(test_set, 'wb') as fp:
        msgpack.dump(list(test_ids), fp)


def d2v_predict(labels, d2v, docs):
    true = [label for _, _, label in labels]
    pred = [(1 - cosine(d2v.infer_vector(docs[id0]['text']),
                        d2v.infer_vector(docs[id1]['text'])))
            for id0, id1, _ in labels]

    return true, pred


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('stopwords', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-ids', type=click.Path(exists=True, dir_okay=False))
@click.argument('labeled', type=click.Path(exists=True, dir_okay=False))
@click.argument('model')
@click.option('--iterations', default=1, type=click.INT)
def train_d2v(documents, stopwords, train_ids, labeled, model, iterations):
    with open(stopwords, 'rb') as fp:
        stops = set(msgpack.load(fp))
    docs = load_docs(documents, process_text=simple_preprocess_and_filter_stopwords(stops))
    with open(train_ids, 'rb') as fp:
        ids = msgpack.load(fp)
        train_docs = {id_.decode(): docs[id_.decode()] for id_ in ids}
    labels = load_testset(labeled, list(docs.keys()))

    tagged_docs = [
        TaggedDocument(doc['text'],
                       tags=[id_] + doc.get('tag', [])) for id_, doc in train_docs.items()
    ]

    if os.path.exists(model):
        print("training an existing model:", model)
        d2v = Doc2Vec.load(model)
    else:
        print("training a new model")
        d2v = Doc2Vec(dm=0,
                      window=1,
                      vector_size=256,
                      workers=multiprocessing.cpu_count() - 1)

    # gensim does not allow update=True if there was no previous vocab, which seems like poor api design
    d2v.build_vocab(tagged_docs, update=os.path.exists(model))

    for i in range(iterations):
        # gensim marks total_examples and epochs as optional in its api docs, but throws runtime errors if they are not present
        d2v.train(tagged_docs,
                  total_examples=d2v.corpus_count,
                  epochs=1)

        true, pred = d2v_predict(labels, d2v, docs)
        d2v.threshold = calculate_best_threshold(pred, labels)

        print("threshold:", d2v.threshold)
        print_confusion_matrix(confusion_matrix(true, [p > d2v.threshold for p in pred]))

        d2v.save(model)


def calculate_best_threshold(pred, labels):
    true = [label for _, _, label in labels]
    best_thresh = best_rate = -1
    for thresh in range(0, 100):
        thresh /= 100
        (tn, fp), (fn, tp) = confusion_matrix(true, [p > thresh for p in pred])
        true_pos_rate = tp / (tp + fn)
        true_neg_rate = tn / (tn + fp)
        current_rate = min(true_pos_rate, true_neg_rate)
        if  current_rate > best_rate:
            best_thresh = thresh
            best_rate = current_rate
    return best_thresh


def print_confusion_matrix(cm):
    (tn, fp), (fn, tp) = cm
    num_pos = tp + fn
    num_neg = tn + fp

    print("TP:", tp, "FN:", fn, "TN:", tn, "FP:", fp)
    print("accuracy:", pct((tp + tn) / (num_pos + num_neg)))
    print("misclassification rate:", pct((fp + fn) / (num_pos + num_neg)))
    print("true positive rate | sensitivity | recall:", pct(tp / num_pos))
    print("false positive rate:", pct(fp / num_neg))
    print("true negative rate | specificity:", pct(tn / num_neg))
    print("prevalence:", pct(num_pos / (num_pos + num_neg)))


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('labeled', type=click.Path(exists=True, dir_okay=False))
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
def test_d2v(documents, labeled, model):
    docs = load_docs(documents)
    labels = load_testset(labeled, list(docs.keys()))
    d2v = Doc2Vec.load(model)

    true, pred = d2v_predict(labels, d2v, docs)
    print("threshold:", d2v.threshold)
    print_confusion_matrix(confusion_matrix(true, [p > d2v.threshold for p in pred]))


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('labeled', type=click.Path(exists=True, dir_okay=False))
def test_minhash(documents, labeled):
    docs = load_docs(documents)
    labels = load_testset(labeled, list(docs.keys()))
    model = nutai.minhash.Model()

    true = [label for _, _, label in labels]
    pred = [approx_jaccard_score(model.calculate_signature(docs[id0]['text']),
                                 model.calculate_signature(docs[id1]['text']))
            for id0, id1, _ in labels]
    best_thresh = calculate_best_threshold(pred, labels)

    print("best threshold:", best_thresh)
    print_confusion_matrix(confusion_matrix(true, [p > best_thresh for p in pred]))


@click.group()
def cli():
    pass


cli.add_command(generate_stopwords)
cli.add_command(split)
cli.add_command(train_d2v)
cli.add_command(test_d2v)
cli.add_command(test_minhash)


if __name__ == '__main__':
    cli()
