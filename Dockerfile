FROM registry.access.redhat.com/ubi8/python-36:1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nutai/* nutai/

ENTRYPOINT [ "python", "-m", "nutai" ]
CMD [ "minhash" ]
