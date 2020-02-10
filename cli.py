import click
from gensim.corpora import Dictionary
import msgpack
from nutai.helpers import load_texts


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


@click.group()
def cli():
    pass


cli.add_command(generate_stopwords)


if __name__ == '__main__':
    cli()
