# Code Clone Detection 


The data was taken from: https://github.com/zhangj111/astnn 


Reference links:
- https://github.com/JetBrains-Research/astminer/ : get .dot graph from source files. can be easily read in networkxx
- https://github.com/c2nes/javalang: get java ast inside javalang itself or use https://github.com/tree-sitter/py-tree-sitter/
- GNN in pytorch: https://github.com/rusty1s/pytorch_geometric
- Another GNNs in pytorch: https://github.com/dmlc/dgl
- TreeLSTM: https://github.com/dmlc/dgl/tree/master/examples/pytorch/tree_lstm
- GNN in TF2: https://github.com/microsoft/tf2-gnn/
- More  GNN in TF: https://github.com/deepmind/graph_nets
- Graph Matching: https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/graph_matching_networks/graph_matching_networks.ipynb



- RNN
    - code2vec: 0.86
    - normal: 0.83
- Code2Vec
    - GraphConv: 0.56
    - Siamese of (GraphConv+TopKPooling): 0.57
    - GraphConv(AST merged by tokes and thier location) + TopKPooling: 0.84
- FastText
    - GraphConv(AST merged by their structure)  + TopKPooling: 0.84
    - GraphConv(Disjoint AST)  + TopKPooling: underway
    - Siamese of (GraphConv+TopKPooling): 0.90
 