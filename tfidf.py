from nutai.helpers import load_texts, save_scores
from mfoops.timer import Timer

import gensim
from tqdm import tqdm


def __main__():
    ids, texts = load_texts("solutions.json")

    dictionary = gensim.corpora.Dictionary(tqdm(texts, desc="building dictionary"))
    dictionary.filter_extremes()
    dictionary.compactify()

    with Timer("model build time"):
        tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    vecs = [dictionary.doc2bow(text) for text in tqdm(texts, desc="building index")]
    index = gensim.similarities.SparseMatrixSimilarity(tfidf[vecs], num_features=len(dictionary))

    save_scores(ids, index)


if __name__ == "__main__":
    __main__()
