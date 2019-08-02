FROM registry.access.redhat.com/ubi8/python-36

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nut.yaml nut.py api.py minhash.py timer.py .

CMD [ "python", "-m", "nut" ]
