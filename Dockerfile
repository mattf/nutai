FROM registry.access.redhat.com/ubi8/python-36

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nut/* nut/

CMD [ "python", "-m", "nut" ]
