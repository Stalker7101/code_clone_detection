FROM tensorflow/tensorflow:nightly-gpu-py3

RUN apt-get update

ADD requirements.txt /
RUN pip install -r /requirements.txt

WORKDIR /app

