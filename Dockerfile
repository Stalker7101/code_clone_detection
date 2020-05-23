FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update

# installing graphviz is a pain in the ass
RUN apt-get install -y graphviz python3-pydot libgraphviz-dev  python3-pygraphviz

# install requirements
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

# install pytorch geometric and dependencies
RUN pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-geometric

WORKDIR /app

