import click
from gensim.corpora import Dictionary
import msgpack
from nutai.helpers import load_texts, load_testset
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


@click.group()
def cli():
    pass


cli.add_command(generate_stopwords)
cli.add_command(split)


if __name__ == '__main__':
    cli()
