import os

import click
import connexion

import nutai.api as api
import nutai.minhash
import nutai.doc2vec


def create_app(model):
    app = connexion.FlaskApp(__name__)

    # setup operationIds
    nut = api.DocNut(model)
    api.similarById = nut.similar_by_id
    api.similarByContent = nut.similar_by_content
    api.addDocument = nut.add_document
    api.addDocuments = nut.add_documents
    api.status = nut.status

    app.add_api('doc_nut.yaml')

    return app


def start(model, port=os.getenv('PORT')):
    app = create_app(model)
    app.run(port)


@click.command()
def minhash():
    start(nutai.minhash.Model())


@click.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
def doc2vec(model):
    start(nutai.doc2vec.Model(model))


@click.command()
def topics(port=os.getenv('PORT')):
    app = connexion.FlaskApp(__name__)

    # setup operationIds
    nut = api.TopicNut()
    api.getTopics = nut.get_topics

    app.add_api('topic_nut.yaml')

    app.run(port)


@click.group()
def cli():
    pass


cli.add_command(minhash)
cli.add_command(doc2vec)
cli.add_command(topics)


if __name__ == '__main__':
    cli()
