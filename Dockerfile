FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
RUN apt-get update

# installing graphviz is a pain in the ass
RUN apt-get install -y graphviz python3-pydot libgraphviz-dev  python3-pygraphviz

# install requirements
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# install pytorch geometric and dependencies
RUN pip install torch-scatter
RUN pip install torch-sparse
RUN pip install torch-cluster
RUN pip install torch-spline-conv
RUN pip install torch-geometric
RUN pip install dgl

WORKDIR /app

