import os

import click
import connexion

import nutai.api as api
import nutai.minhash
import nutai.doc2vec


def start(model, port=os.getenv('PORT')):
    app = connexion.FlaskApp(__name__)

    # setup operationIds
    nut = api.Nut(model)
    api.similarById = nut.similarById
    api.similarByContent = nut.similarByContent
    api.addDocument = nut.addDocument
    api.addDocuments = nut.addDocuments
    api.status = nut.status

    app.add_api('nutai.yaml')

    app.run(port)


@click.command()
def minhash():
    start(nutai.minhash.Model())


@click.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
def doc2vec(model):
    start(nutai.doc2vec.Model(model))


@click.group()
def cli():
    pass


cli.add_command(minhash)
cli.add_command(doc2vec)


if __name__ == '__main__':
    cli()
