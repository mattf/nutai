import os
import random
import time

import numpy as np
import redis
import json

# TODO: remove magic 42


class Model:
    def __init__(self):
        pass

    def set_store(self, store):
        pass

    def get_signature_length(self):
        pass

    def get_signature_type(self):
        pass

    def get_threshold(self):
        pass

    def calculate_signature(self, text):
        pass

    def calculate_similarity(self, signature):
        pass


class NullRedis:
    def rpush(self, k, v): pass
    def llen(self, k): return 0
    def lrange(self, k, x, y): return []
    def info(self): return "fake, no redis"
    def echo(self, msg): return msg


class Store:
    def __init__(self, key, signature_length, signature_type):
        self.signature_length = signature_length
        self.key = key
        print("key:", self.key)
        try:
            self.r = redis.Redis()
            self.r.echo("morning")
        except redis.exceptions.ConnectionError:
            self.r = NullRedis()
        self._end = 0
        len_ = max(self.r.llen(self.key), 42)  # if redis is empty, don't start w/ 0 len
        self.ids = np.ndarray((len_,), dtype='<U42')  # TODO: find appropriate id length
        self.sigs = np.ndarray((len_, self.signature_length), dtype=signature_type)
        print("loaded", self._catch_up(), "signatures")

    def __contains__(self, id):
        self._catch_up()
        return id in self.ids

    def _extend(self, inc=None):
        if not inc:
            inc = int(max(1, self._end * .25))
        # assert self.ids.shape[0] == self.sigs.shape[0]
        size = self.ids.shape[0] + inc
        self.ids.resize((size,), refcheck=False)  # TODO: refcheck=True
        self.sigs.resize((size, self.signature_length), refcheck=False)  # TODO: refcheck=True

    def _catch_up(self):
        count = self._end
        # _end represents the number of (id,sig) pairs known locally
        # llen(key) represents the number of (id,sig) pairs known globally
        # local knowledge should never be ahead of global knowledge
        # while local knowledge trails global,
        #    pull down additional global knowledge
        len_ = self.r.llen(self.key)
        while self._end < len_:
            self._extend(len_ - self._end)
            unknown = self.r.lrange(self.key, self._end, len_)
            for raw in unknown:
                pair = json.loads(raw)
                self.ids[self._end] = pair['id']
                self.sigs[self._end] = np.array(pair['sig'])
                self._end += 1
            len_ = self.r.llen(self.key)  # TODO: find a way to avoid infinite loop
        return self._end - count

    def add(self, id_, sig):
        self._catch_up()
        if self._end == len(self.ids):
            self._extend()
        self.ids[self._end] = id_
        self.sigs[self._end] = sig
        self._end += 1
        i = self.r.rpush(self.key, json.dumps({"id": id_, "sig": sig.tolist()}))
        # TODO: _end should never be > i
        # TODO: handle _end < i, aka there was an addition between start of
        #       this method and the rpush need to catch up, and it's ok if
        #       local and remote lists are not in the same order, as long as
        #       they have the same contents


class DocNut:
    def __init__(self, model, key=None):
        seed = os.getenv('SEED', int(time.time()))
        random.seed(seed)
        print("using seed:", seed)
        self.model = model
        self.store = Store(key=(key or 'sigs:42:' + str(seed)),
                           signature_length=self.model.get_signature_length(),
                           signature_type=self.model.get_signature_type())
        self.model.set_store(self.store)

    def add_documents(self, body):
        accepted = []
        rejected = []
        for doc in body:
            if self.add_document(doc['id'], doc['content']):
                rejected.append(doc['id'])
            else:
                accepted.append(doc['id'])
        return {"accepted": accepted, "rejected": rejected}

    def add_document(self, id_, body):
        if len(id_) > 42:
            return 'Id too long', 400

        if id_ in self.store:
            return 'Document already exists', 409

        self.store.add(id_, self.model.calculate_signature(body))

    def _generate_similar_output(self, hits, scores):
        return [{"id": id_, "score": score}
                for id_, score in zip(self.store.ids[hits],
                                      (scores[hits]*100).astype(int).tolist())]

    def similar_by_id(self, id_):
        if id_ not in self.store:
            return 'Not Found', 404

        sig = self.store.sigs[self.store.ids == id_][0]
        scores = self.model.calculate_similarity(sig)
        hits = scores > self.model.get_threshold()

        return self._generate_similar_output(hits, scores)

    def similar_by_content(self, content):
        sig = self.model.calculate_signature(content)
        scores = self.model.calculate_similarity(sig)
        hits = scores > self.model.get_threshold()

        return self._generate_similar_output(hits, scores)

    def status(self):
        return {'_end': self.store._end,
                'redis': self.store.r.info(),
                'len(ids)': len(self.store.ids),
                'ids': self.store.ids.tolist(),
                'len(sigs)': len(self.store.sigs),
                'sigs': self.store.sigs.tolist()}


class TopicNut:
    def __init__(self):
        pass

    def get_topics(self, document):
        return ['bonus0', 'bogus topic', 'who_knows']
