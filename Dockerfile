FROM python:3.6

RUN apt update && apt install -y python-pip

WORKDIR /opt/neural-edu-seg/data/elmo
RUN curl -O https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

WORKDIR /opt/neural-edu-seg
ADD requirements.txt /opt/neural-edu-seg
RUN pip install -r requirements.txt

COPY . /opt/neural-edu-seg

RUN python setup.py install && python -m spacy download en

COPY data /usr/local/lib/python3.6/site-packages/neuralseg-0.1.0a0-py3.6.egg/data

WORKDIR /opt/neural-edu-seg

# TODO: remove after debug
#RUN pip install ipython pudb

ENTRYPOINT ["neuralseg/splitter.py"]

