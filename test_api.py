import redis

import api

def test_get_unknown_document():
    # mock up Redis
    fake_redis = api.NullRedis()
    redis.Redis = lambda: fake_redis

    nut = api.Nut()
    _, code = nut.similarById("0")
    assert code == 404

def test_add_single_document():
    # mock up Redis
    fake_redis = api.NullRedis()
    redis.Redis = lambda: fake_redis

    nut = api.Nut()
    assert nut.addDocument("0", "a b c") is None
    assert len(nut.similarById("0")) == 1
    assert nut.status()["_end"] == 1

def test_add_duplicate_document():
    # mock up Redis
    fake_redis = api.NullRedis()
    redis.Redis = lambda: fake_redis

    nut = api.Nut()
    assert nut.addDocument("0", "a b c") is None
    assert len(nut.similarById("0")) == 1
    assert nut.status()["_end"] == 1
    _, code = nut.addDocument("0", "x y z")
    assert code == 409
    assert nut.status()["_end"] == 1
