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
