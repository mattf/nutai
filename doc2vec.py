import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np
from tqdm import tqdm

from nutai.helpers import load_texts, save_scores
from mfoops.timer import Timer

def __main__():
    ids, texts = load_texts("solutions.json")
    tagged_docs = [TaggedDocument(text, [id_]) for id_, text in zip(ids, texts)]

    model = Doc2Vec()
    model.build_vocab(tqdm(tagged_docs, desc="building vocab"))

    model.train(tqdm(tagged_docs, desc="training"),
        total_examples=model.corpus_count, epochs=model.epochs)

    def score_generator(ids, model):
        for id_ in ids:
            yield model.docvecs.distances(id_)

    save_scores(ids, score_generator(ids, model))

if __name__ == "__main__":
    __main__()
