import os
import random
import time

import numpy as np
import minhash
import redis
import json

# TODO: remove magic 42


class NullRedis:
    def rpush(self, k, v): pass
    def llen(self, k): return 0
    def lrange(self, k, x, y): return []
    def info(self): return "fake, no redis"
    def echo(self, msg): return msg

class Store:
    def __init__(self, key):
        self.key = key
        print("key:", self.key)
        try:
            self.r = redis.Redis()
            self.r.echo("morning")
        except:
            self.r = NullRedis()
        self._end = 0
        len_ = max(self.r.llen(self.key), 42) # if redis is empty, don't start w/ 0 len
        self.ids = np.ndarray((len_,), dtype='<U42') # TODO: find appropriate id length
        self.sigs = np.ndarray((len_, 42), dtype=int)
        print("loaded", self._catch_up(), "signatures")

    def __contains__(self, id):
        self._catch_up()
        return id in self.ids

    def _catch_up(self):
        count = self._end
        # _end represents the number of (id,sig) pairs known locally
        # llen(key) represents the number of (id,sig) pairs known globally
        # local knowledge should never be ahead of global knowledge
        # while local knowledge trails global,
        #    pull down additional global knowledge
        len_ = self.r.llen(self.key)
        while self._end < len_:
            unknown = self.r.lrange(self.key, self._end, len_)
            self.ids.resize((len_,))
            self.sigs.resize((len_, 42))
            for raw in unknown:
                pair = json.loads(raw)
                self.ids[self._end] = pair['id']
                self.sigs[self._end] = np.array(pair['sig'])
                self._end += 1
            len_ = self.r.llen(self.key) # TODO: find a way to avoid infinite loop
        return self._end - count

    def add(self, id, sig):
        self._catch_up()
        if self._end == len(self.ids):
            # if redis gave us < 4 items, _end will be too small to grow w/ *1.25
            size = max(42, int(self._end * 1.25))
            self.ids.resize((size,))
            self.sigs.resize((size, 42))
        self.ids[self._end] = id
        self.sigs[self._end] = sig
        self._end += 1
        i = self.r.rpush(self.key, json.dumps({"id":id,"sig":sig.tolist()}))
        # TODO: _end should never be > i
        # TODO: handle _end < i, aka there was an addition between start of
        #       this method and the rpush need to catch up, and it's ok if
        #       local and remote lists are not in the same order, as long as
        #       they have the same contents


class Nut:
    def __init__(self, key=None):
        print("initializing...")
        seed = os.getenv('SEED', int(time.time()))
        random.seed(seed)
        print("using seed:", seed)
        # signatures are only comparable when they have the same
        #  0. hash functions (# hashes, coeffs, prime modulus)
        #  1. input processing  (generate_shingles)
        # TODO: capture version of input processor
        self.store = Store(key=(key or 'sigs:42:' + str(seed)))
        self.hash_funcs = list(minhash.generate_hash_funcs(42))

    def addDocuments(self, body):
        accepted = []
        rejected = []
        for doc in body:
            if self.addDocument(doc['id'], doc['content']):
                rejected.append(doc['id'])
            else:
                accepted.append(doc['id'])
        return {"accepted": accepted, "rejected": rejected}

    def addDocument(self, id, body):
        if len(id) > 42:
            return 'Id too long', 400

        if id in self.store:
            return 'Document already exists', 409

        shingles = list(minhash.generate_shingles(body.split(" ")))
        self.store.add(id, minhash.calculate_signature(shingles, self.hash_funcs))

    def similarById(self, id):
        if id not in self.store:
            return 'Not Found', 404

        sig = self.store.sigs[self.store.ids == id][0]
        scores = minhash.approx_jaccard_score(sig, self.store.sigs, 1)
        hits = scores > .42 # TODO: find appropriate threshold

        return [{"id": id, "score": score}
                for id, score in zip(self.store.ids[hits],
                                     (scores[hits]*100).astype(int).tolist())]

    def similarByContent(self, content):
        shingles = list(minhash.generate_shingles(content.split(" ")))
        sig = minhash.calculate_signature(shingles, self.hash_funcs)
        scores = minhash.approx_jaccard_score(sig, self.store.sigs, 1)
        hits = scores > .42 # TODO: find appropriate threshold

        return [{"id": id, "score": score}
                for id, score in zip(self.store.ids[hits],
                                     (scores[hits]*100).astype(int).tolist())]

    def status(self):
        return {'_end': self.store._end,
                'redis': self.store.r.info(),
                'len(ids)': len(self.store.ids),
                'ids': self.store.ids.tolist(),
                'len(sigs)': len(self.store.sigs),
                'sigs': self.store.sigs.tolist()}
