# How to use this...

### You can substitute `docker` for `podman` in these instructions.

## Build

```bash
podman build -t nut:latest .
```

## Run

```bash
podman run --name nut-redis -p 6379:6379 -d redis
podman run --name nut -p 5000:5000 -d nut
```

## Use

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
