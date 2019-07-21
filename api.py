import numpy as np
import minhash
import redis
import json

# TODO: remove magic 42

class NullRedis:
    def rpush(self, k, v): pass
    def llen(self, k): return 42
    def lrange(self, k, x, y): return []

class Store:
    def __init__(self):
        try:
            self.r = redis.Redis()
            self.r.echo("morning")
        except:
            self.r = NullRedis()
        self._end = 0
        len_ = max(self.r.llen('sigs'), 42) # if redis is empty, don't start w/ 0 len
        self.ids = np.ndarray((len_,), dtype='<U42') # TODO: find appropriate id length
        self.sigs = np.ndarray((len_, 42), dtype=int)
        for raw in self.r.lrange('sigs', 0, len_):
            data = json.loads(raw)
            self.ids[self._end] = data['id']
            self.sigs[self._end] = np.array(data['sig'])
            self._end += 1
        print("loaded", self._end, "signatures")

    def __contains__(self, id):
        return id in self.ids

    def add(self, id, sig):
        self._end += 1
        if self._end == len(self.ids):
            self.ids.resize((int(self._end * 1.25),))
            self.sigs.resize((int(self._end * 1.25), 42))
        self.ids[self._end] = id
        self.sigs[self._end] = sig
        self.r.rpush('sigs', json.dumps({"id":id,"sig":sig.tolist()}))


print("initializing...")
store = Store()
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
    if id in store:
        return 'Document already exists', 409

    shingles = list(minhash.generate_shingles(body.split(" ")))
    store.add(id, minhash.calculate_signature(shingles, hash_funcs))

def similarById(id):
    if id not in store:
        return 'Not Found', 404

    sig = store.sigs[store.ids == id][0]
    scores = minhash.approx_jaccard_score(sig, store.sigs, 1)
    hits = scores > .42 # TODO: find appropriate threshold

    return [{"id": id, "score": score}
            for id, score in zip(store.ids[hits],
                                 (scores[hits]*100).astype(int).tolist())]
