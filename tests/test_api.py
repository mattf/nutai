import os

import pytest
import redis

from nutai import api
from nutai import minhash

# TODO: add tests for addDocuments


class NullModel(minhash.Model):
    pass


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
        try:
            self.r.ping()
        except:
            pytest.skip("redis not available")
        self.key = "test-key"
        return self.key

    def __exit__(self, type, value, traceback):
        self.r.delete(self.key)


def test_get_unknown_document():
    with MockRedis():
        nut = api.DocNut(NullModel())
        _, code = nut.similar_by_id("0")
        assert code == 404


def test_add_single_document():
    with MockRedis():
        nut = api.DocNut(NullModel())
        assert nut.add_document("0", "a b c") is None
        assert len(nut.similar_by_id("0")) == 1
        assert nut.status()["_end"] == 1


def test_add_duplicate_document():
    with MockRedis():
        nut = api.DocNut(NullModel())
        assert nut.add_document("0", "a b c") is None
        assert len(nut.similar_by_id("0")) == 1
        assert nut.status()["_end"] == 1
        _, code = nut.add_document("0", "x y z")
        assert code == 409
        assert nut.status()["_end"] == 1


def test_add_too_long_id():
    with MockRedis():
        nut = api.DocNut(NullModel())
        id_ = "0" * 128
        _, code = nut.add_document(id_, "a b c")
        assert code == 400
        assert nut.status()["_end"] == 0


def test_add_max_length_id():
    with MockRedis():
        nut = api.DocNut(NullModel())
        id_ = "0" * 42
        assert nut.add_document(id_, "a b c") is None
        assert len(nut.similar_by_id(id_)) == 1
        assert nut.status()["_end"] == 1


def test_get_too_long_id():
    with MockRedis():
        nut = api.DocNut(NullModel())
        id_ = "0" * 128
        _, code = nut.similar_by_id(id_)
        assert code == 404


def test_add_to_force_resize():
    with MockRedis():
        nut = api.DocNut(NullModel())
        size = nut.store.ids.shape[0]
        for i in range(size + 1):
            assert nut.add_document(str(i), "filler") is None
        assert size < nut.store.ids.shape[0]


def test_compare0():
    with MockRedis():
        nut = api.DocNut(NullModel())
        assert nut.add_document("0", "a b c") is None
        assert len(nut.similar_by_content("a b c")) == 1


def test_parallel_add_single_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.DocNut(NullModel(), key=key), api.DocNut(NullModel(), key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.add_document("0", "a b c") is None
        assert len(nut1.similar_by_id("0")) == 1
        assert nut0.status()["_end"] == 1
        assert nut1.status()["_end"] == 1


def test_parallel_add_duplicate_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.DocNut(NullModel(), key=key), api.DocNut(NullModel(), key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.add_document("0", "a b c") is None
        _, code = nut1.add_document("0", "x y z")
        assert code == 409
        assert nut0.status()["_end"] == 1
        assert nut1.status()["_end"] == 1


def test_parallel_add_second_document():
    # mock os.getenv("SEED")
    os.getenv = lambda key, default=None: "42"
    with TestSpace() as key:
        nut0, nut1 = api.DocNut(NullModel(), key=key), api.DocNut(NullModel(), key=key)
        assert nut0.store != nut1.store
        assert nut0.store.r != nut1.store.r
        assert nut0.store.key == nut1.store.key
        assert nut0.add_document("0", "a b c") is None
        assert nut1.add_document("1", "x y z") is None
        assert len(nut0.similar_by_id("1")) == 1
        assert len(nut1.similar_by_id("0")) == 1
        assert nut0.status()["_end"] == 2
        assert nut1.status()["_end"] == 2


def test_get_topics_exists():
    nut = api.TopicNut()
    assert type(nut.get_topics("nothing special")) is list
