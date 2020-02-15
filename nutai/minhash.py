import minhash


class Model:
    def __init__(self):
        # signatures are only comparable when they have the same
        #  0. hash functions (# hashes, coeffs, prime modulus)
        #  1. input processing  (generate_shingles)
        # TODO: capture version of input processor
        self.hash_funcs = list(minhash.generate_hash_funcs(self.get_signature_length()))

    def set_store(self, store):
        self.store = store

    def get_signature_length(self):
        return 42

    def get_signature_type(self):
        return int

    def get_threshold(self):
        return .42

    def calculate_signature(self, text):
        if type(text) == str:
            text = text.split(" ")
        return minhash.calculate_signature(list(minhash.generate_shingles(text)), self.hash_funcs)

    def calculate_similarity(self, signature):
        return minhash.approx_jaccard_score(signature, self.store.sigs, 1)
