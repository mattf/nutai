FROM registry.access.redhat.com/ubi8/python-36:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nutai/* nutai/

CMD [ "python", "-m", "nutai", "minhash" ]
