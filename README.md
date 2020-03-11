# Usage

Note: You can substitute `docker` for `podman` in these instructions.

## Run it

### Pre-built

```bash
podman build -t mattf/nutai:latest .
```

### Do it yourself

```bash
podman build -t nutai:latest .
podman run --name nutai -p 5000:5000 -d nutai
```

## Try it

```bash
curl 127.1:5000/documents/0
: "Not Found"

curl -d '"z e r o"' -H 'Content-Type: application/json' 127.1:5000/documents/0
: 200

curl 127.1:5000/documents/0
: '[{"id": "0", "score": 0}]'

curl -d '"o r e z"' -H 'Content-Type: application/json' 127.1:5000/documents/0
: "Document already exists"

curl -d '"o n e"' -H 'Content-Type: application/json' 127.1:5000/documents/1
: 200

curl 127.1:5000/documents/1
: '[{"id": "0", "score": 0}, {"id": "1", "score": 0}]'
```

## Advanced

### Doc2Vec

Assuming you have a model built and stored in your current directory as `d2v.model`

```bash
podman run --name nutai -p 5000:5000 -v .:/model:ro,Z -d nutai -- doc2vec /model/d2v.model
```

### Storage with Redis

Do this when you want to restart the service and not lose anything that was added. Of course, if you restart Redis you'll lose anything it hasn't persisted.

```bash
podman run --name nutai-redis -p 6379:6379 -d redis
```
