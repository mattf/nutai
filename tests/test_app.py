import os

import json
import pytest

from nutai import minhash
from nutai.__main__ import create_app

# TODO: make this work when redis is available

@pytest.fixture
def client():
    app = create_app(minhash.Model())
    return app.app.test_client()


def test_get_unknown_document(client):
    assert client.get('/documents/0').status_code == 404


def test_add_single_document(client):
    response = client.post('/documents/0',
                           content_type='application/json',
                           data=json.dumps('a b c'))
    assert response.status_code == 204

    response = client.get('/documents/0')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert len(result) == 1
    assert result[0]['id'] == '0'


def test_add_duplicate_document(client):
    assert client.post('/documents/0',
                       content_type='application/json',
                       data=json.dumps('a b c')).status_code == 204
    assert client.post('/documents/0',
                       content_type='application/json',
                       data=json.dumps('a b c')).status_code == 409

    response = client.get('/documents/0')
    assert response.status_code == 200
    result = json.loads(response.data)
    assert len(result) == 1
    assert result[0]['id'] == '0'
