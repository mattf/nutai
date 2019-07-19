import numpy as np
import minhash

# TODO: remove magic 42

print("initializing...")
ids = np.ndarray((0,), dtype='<U42') # TODO: find appropriate id length
sigs = np.ndarray((0, 42), dtype=int)
hash_funcs = list(minhash.generate_hash_funcs(42))

def addDocuments(body):
    accepted = []
    rejected = []
    for doc in body:
        if addDocument(doc['id'], doc['content']):
            rejected.append(doc['id'])
        else:
            accepted.append(doc['id'])
    return {"accepted": accepted, "rejected": rejected}

def addDocument(id, body):
    if id in ids:
        return 'Document already exists', 409

    # TODO: make growing efficient
    ids.resize((len(ids) + 1,))
    sigs.resize((len(sigs) + 1, 42))

    ids[-1] = id
    shingles = list(minhash.generate_shingles(body.split(" ")))
    sigs[-1] = minhash.calculate_signature(shingles, hash_funcs)


def similarById(id):
    if id not in ids:
        return 'Not Found', 404

    sig = sigs[ids == id][0]
    scores = minhash.approx_jaccard_score(sig, sigs, 1)
    hits = scores > .42 # TODO: find appropriate threshold

    return [{"id": id, "score": score}
            for id, score in zip(ids[hits],
                                 (scores[hits]*100).astype(int).tolist())]
