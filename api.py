documents = dict()

def findById(id):
    if id not in documents:
        return 'Not Found', 404
    return documents[id]

def addDocument(id, body):
    if id in documents:
        return 'Document already exists', 409
    documents[id] = body

def similarById(id):
    if id not in documents:
        return 'Not Found', 404
    return [{"id": id, "score": 0} for id in documents.keys()]
