# Code Clone Detection 

## Setup

- Dockerile <3

## Files 

A list of important notebooks are as follows:
* [notebooks/clone_detection_baseline.ipynb](notebooks/clone_detection_baseline.ipynb): Uses LSTM with code2vec(0.86)/fasttext(0.82)/random embeddings(0.83) for the task
* [notebooks/model_play-seasme.ipynb](notebooks/model_play-seasme.ipynb): Uses a Siamese Nework with base model of GrapConv+TopKPooling and node attributes assigned using code2vec(0.56)/fasttext(0.90)
* [notebooks/model_play.ipynb](notebooks/model_play.ipynb): Uses GrapConv+TopKPooling with code2vec(0.84)/fasttext(0.90).
* [notebooks/dgl_model_play.ipynb](notebooks/dgl_model_play.ipynb): Uses just GraphConv with code2vec(0.56)
* [notebooks/data_preprocssing_main.ipynb]: For making trying different kinds of processing on AST network, making vocab, training fasttext embbedings.
Other notebooks have experminents that we weren't able to execute succesfully due one or more errors.

A list of important code files:
* [src/code_parser.py](src/code_parser.py): Code for parsing a string java code, making an AST followed by making a networkx graph and combining it.
* [src/dataset.py](src/dataset.py): Make several kind of torch_geometric dataset
* [src/data_prep.py](src/data_prep.py): Data precrossing and data split script.




## References 

- The data was taken from: https://github.com/zhangj111/astnn/tree/master/clone/data/java
- Code to AST conversion done using: https://github.com/tree-sitter/py-tree-sitter/
- Training and making GNNs in PyTorch done using https://github.com/rusty1s/pytorch_geometric and https://github.com/dmlc/dgl
- Code2Vec embedding taken from https://github.com/tech-srl/code2vec


- RNN
    - code2vec: 0.86
    - normal: 0.83
- Code2Vec
    - GraphConv: 0.56
    - Siamese of (GraphConv+TopKPooling): 0.57
    - GraphConv(AST merged by tokes and thier location) + TopKPooling: 0.84
- FastText
    - GraphConv(AST merged by their structure)  + TopKPooling: 0.84
    - GraphConv(Disjoint AST)  + TopKPooling: 0.87
    - Siamese of (GraphConv+TopKPooling): 0.90
