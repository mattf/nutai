import os

import click
import connexion

import nutai.api as api
import nutai.minhash


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


@click.group()
def cli():
    pass


cli.add_command(minhash)


if __name__ == '__main__':
    cli()
