import glob
import random
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

        processed_file_names = []

        for idx in range(len(self.pair_ids)):
            path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))
            if os.path.exists(path):
                processed_file_names.append('data_{}.pt'.format(idx))

        if len(processed_file_names) < 10:
            return ['data_{}.pt'.format(idx) for idx in range(len(self.pair_ids))]

        return processed_file_names

    def process(self):

        global apply

        with open(self.pairs_path, 'rb') as f:
            pair_ids = pickle.load(f).to_numpy()

        def apply(in_):
            i, (id1, id2, label) = in_
            try:
                with open(os.path.join(self.functions_path, str(id1)), 'rb') as f:
                    g1: nx.DiGraph = pickle.load(f)
                with open(os.path.join(self.functions_path, str(id2)), 'rb') as f:
                    g2: nx.DiGraph = pickle.load(f)
            except FileNotFoundError as ex:
                print(ex)
                return

            g3 = nx.compose(g1, g2)

            y = int(label)

            data: Data = get_data_from_graph(g3, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                return

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

        from multiprocessing import Pool
        with Pool(40) as p:
            p.map(apply, enumerate(pair_ids))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        path = os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))

        if not os.path.exists(path):
            path = random.choice(glob.glob(os.path.join(self.processed_dir, "data_*.pt")))

        data = torch.load(path)
        return data
