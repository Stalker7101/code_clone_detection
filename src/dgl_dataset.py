import glob
import os
import pickle
import random
from multiprocessing import Pool

import dgl
import networkx as nx
from dgl.data.utils import load_graphs, save_graphs
from torch.utils.data import Dataset


# def data_prep(orig, result):
#     xxxx = glob.glob(os.path.join(orig, "*"))
#
#     def apply(path):
#         with open(path, 'rb') as f:
#             g: nx.DiGraph = pickle.load(f)
#         g = nx.convert_node_labels_to_integers(g)
#         g_dgl = dgl.DGLGraph()
#         g_dgl.from_networkx(g, node_attrs=['data'])
#         save_graphs(os.path.join(result, path.split("/")[-1]), g_dgl)
#
#     with Pool(20) as p:
#         p.map(apply, xxxx)

class CloneDataset(Dataset):

    def __init__(self, functions_path, pairs_path):

        self.functions_path = functions_path
        self.pairs_path = pairs_path

        with open(self.pairs_path, 'rb') as f:
            self.pair_ids = pickle.load(f).to_numpy()

    def __len__(self):
        return len(self.pair_ids)

    def read_ast(self, id_):
        try:
            graph, _ = load_graphs(os.path.join(self.functions_path, str(id_)))
        except Exception as ex:
            return None

        return graph[0]

    def __getitem__(self, item):
        id1, id2, label = self.pair_ids[item, :]
        g1, g2, label = self.read_ast(id1), self.read_ast(id2), int(label)

        if g1 is None or g2 is None:
            return self.__getitem__(random.randint(0, len(self.pair_ids)))
        else:
            return g1, g2, label
