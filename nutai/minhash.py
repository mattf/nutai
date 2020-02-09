import minhash


class Model:
    def __init__(self, store):
        self.store = store
        # signatures are only comparable when they have the same
        #  0. hash functions (# hashes, coeffs, prime modulus)
        #  1. input processing  (generate_shingles)
        # TODO: capture version of input processor
        self.hash_funcs = list(minhash.generate_hash_funcs(42))

    def calculate_signature(self, text):
        return minhash.calculate_signature(list(minhash.generate_shingles(text.split(" "))), self.hash_funcs)

    def calculate_similarity(self, signature):
        return minhash.approx_jaccard_score(signature, self.store.sigs, 1)
