import click
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import msgpack
from nutai.helpers import load_texts, load_docs, load_testset
from sklearn.model_selection import train_test_split


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


@click.command()
@click.argument('documents', type=click.Path(exists=True, dir_okay=False))
@click.argument('model', type=click.Path(exists=False))
def train_d2v(documents, model):
    docs = load_docs(documents)

    tagged_docs = [
        TaggedDocument(doc['text'],
                       tags=[id_] + doc.get('tag', [])) for id_, doc in docs.items()
    ]

    d2v = Doc2Vec(tagged_docs,
                  dm=0,
                  window=1,
                  vector_size=256,
                  epochs=10,
                  workers=multiprocessing.cpu_count() - 1)

    d2v.save(model)


@click.group()
def cli():
    pass


cli.add_command(generate_stopwords)
cli.add_command(split)
cli.add_command(train_d2v)


if __name__ == '__main__':
    cli()
