import os

import redis

import api

class MockRedis:
    def __enter__(self):
        # mock up Redis
        fake_redis = api.NullRedis()
        self.real_redis = redis.Redis
        redis.Redis = lambda: fake_redis
    def __exit__(self, type, value, traceback):
        redis.Redis = self.real_redis

class TestSpace:
    def __enter__(self):
        self.r = redis.Redis()
        self.r.ping()
        self.key = "test-key"
        return self.key
    def __exit__(self, type, value, traceback):
        self.r.delete(self.key)

def test_get_unknown_document():
    with MockRedis():
        nut = api.Nut()
        _, code = nut.similarById("0")
        assert code == 404

def test_add_single_document():
    with MockRedis():
        nut = api.Nut()
        assert nut.addDocument("0", "a b c") is None
        assert len(nut.similarById("0")) == 1
        assert nut.status()["_end"] == 1

def test_add_duplicate_document():
    with MockRedis():
        nut = api.Nut()
        assert nut.addDocument("0", "a b c") is None
        assert len(nut.similarById("0")) == 1
        assert nut.status()["_end"] == 1
        _, code = nut.addDocument("0", "x y z")
        assert code == 409
        assert nut.status()["_end"] == 1

def test_parallel_add_single_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.Nut(key=key), api.Nut(key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.addDocument("0", "a b c") is None
        assert len(nut1.similarById("0")) == 1
        assert nut0.status()["_end"] == 1
        assert nut1.status()["_end"] == 1

def test_parallel_add_duplicate_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.Nut(key=key), api.Nut(key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.addDocument("0", "a b c") is None
        _, code = nut1.addDocument("0", "x y z")
        assert code == 409
        assert nut0.status()["_end"] == 1
        assert nut1.status()["_end"] == 1

def test_parallel_add_second_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.Nut(key=key), api.Nut(key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.addDocument("0", "a b c") is None
        assert nut1.addDocument("1", "x y z") is None
        assert len(nut0.similarById("1")) == 1
        assert len(nut1.similarById("0")) == 1
        assert nut0.status()["_end"] == 2
        assert nut1.status()["_end"] == 2
