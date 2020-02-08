# How to use this...

```bash
pip install -r requirements.txt

python -m nutai
```

```bash
curl 127.1:5000/documents/0
: "Not Found"
```

```bash
curl -d '"z e r o"' -H 'Content-Type: application/json' 127.1:5000/documents/0
: 200
```

```bash
curl 127.1:5000/documents/0
: '[{"id": "0", "score": 0}]'
```

```bash
curl -d '"o r e z"' -H 'Content-Type: application/json' 127.1:5000/documents/0
: "Document already exists"
```

```bash
curl -d '"o n e"' -H 'Content-Type: application/json' 127.1:5000/documents/1
: 200
```

```bash
curl 127.1:5000/documents/1
: '[{"id": "0", "score": 0}, {"id": "1", "score": 0}]'
```
