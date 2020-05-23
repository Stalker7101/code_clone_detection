from typing import List
import os

import torch
import tqdm as tqdm
from torch_geometric.data import Dataset

import pickle

from code_parser import *


class CloneDataset(Dataset):
    def __init__(self, root, functions_path, pairs_path, transform=None, pre_transform=None):

        self.functions_path = functions_path
        self.pairs_path = pairs_path
        with open(self.pairs_path, 'rb') as f:
            self.pair_ids = pickle.load(f).to_numpy()

        super(CloneDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.functions_path)

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self.pair_ids))]

    def process(self):

        with open(self.pairs_path, 'rb') as f:
            pair_ids = pickle.load(f).to_numpy()

        i = 0
        for id1, id2, label in tqdm.tqdm(pair_ids):
            try:
                with open(os.path.join(self.functions_path, str(id1)), 'rb') as f:
                    g1: nx.DiGraph = pickle.load(f)
                with open(os.path.join(self.functions_path, str(id2)), 'rb') as f:
                    g2: nx.DiGraph = pickle.load(f)
            except FileNotFoundError as ex:
                print(ex)
                continue

            g3 = nx.compose(g1, g2)

            y = int(label)

            data: Data = get_data_from_graph(g3, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            self.processed_file_names.append('data_{}.pt'.format(i))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
