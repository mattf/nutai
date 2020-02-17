from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    def __init__(self, model_file):
        self.model = Doc2Vec.load(model_file)

    def set_store(self, store):
        self.store = store

    def get_signature_length(self):
        return self.model.vector_size

    def get_signature_type(self):
        return float

    def get_threshold(self):
        return self.model.threshold

    def calculate_signature(self, text):
        if type(text) == str:
            text = text.split(" ")
        return self.model.infer_vector(text, epochs=100)

    def calculate_similarity(self, signature):
        return cosine_similarity(signature.reshape(1, -1), self.store.sigs)[0]
