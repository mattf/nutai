import redis

import api

def test_get_unknown_document():
    # mock up Redis
    fake_redis = api.NullRedis()
    redis.Redis = lambda: fake_redis

    nut = api.Nut()
    _, code = nut.similarById("0")
    assert code == 404
