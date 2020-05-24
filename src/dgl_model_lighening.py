import os
from argparse import ArgumentParser

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.utils.data import DataLoader

from code_parser import *
from dgl_dataset import CloneDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graph1, graph2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graph1)
    batched_graph2 = dgl.batch(graph2)
    return batched_graph1, batched_graph2, torch.tensor(labels)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graph1, graph2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graph1)
    batched_graph2 = dgl.batch(graph2)
    return batched_graph1, batched_graph2, torch.tensor(labels)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graph1, graph2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graph1)
    batched_graph2 = dgl.batch(graph2)
    return batched_graph1, batched_graph2, torch.tensor(labels)


class NetBasic(nn.Module):

    def __init__(self, hparams):
        super(NetBasic, self).__init__()

        self.hparams = hparams

        self.build_model()

    def build_model(self):
        self.conv1 = GraphConv(self.hparams.num_features, self.hparams.hidden_dim)
        self.conv2 = GraphConv(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.conv3 = GraphConv(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.classify = nn.Linear(self.hparams.hidden_dim * 2, self.hparams.num_classes)

    def forward_(self, g1, g2):
        h1 = g1.ndata['data'].view(-1, self.hparams.num_features).float().to(device)
        h1 = F.relu(self.conv1(g1, h1))
        h1 = F.relu(self.conv2(g1, h1))
        h1 = F.relu(self.conv3(g1, h1))
        g1.ndata['h'] = h1

        h2 = g2.ndata['data'].view(-1, self.hparams.num_features).float().to(device)
        h2 = F.relu(self.conv1(g2, h2))
        h2 = F.relu(self.conv2(g2, h2))
        h2 = F.relu(self.conv3(g2, h2))
        g2.ndata['h'] = h2

        hg1 = dgl.mean_nodes(g1, 'h')
        hg2 = dgl.mean_nodes(g2, 'h')

        return F.log_softmax(self.classify(torch.cat([hg1, hg2], dim=-1)), dim=-1)

    def forward(self, g1, g2):
        return self.forward_(g1, g2)

    def training_step(self, data):
        g1, g2, label = data
        output = self.forward(g1, g2)
        loss = F.nll_loss(output, label.to(device))
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, data):
        g1, g2, label = data
        output = self.forward(g1, g2)
        loss = F.nll_loss(output, label.to(device))
        pred = output.max(dim=1)[1]
        acc = pred.eq(label).type(torch.float32).mean()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    #     def test_step(self, data, batch_idx):
    #         g1, g2, label = data
    #         output = self.forward(g1, g2)
    #         loss = F.cross_entropy(output, label)
    #         pred = torch.softmax(output, 1).max(dim=1)[1]
    #         acc = pred.eq(data.y).type(torch.float32).mean()
    #         return {'test_loss': loss, 'test_acc': acc}

    #     def test_epoch_end(self, outputs):
    #         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #         avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

    #         tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc}
    #         return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return optimizer, scheduler

    def prepare_data(self):
        dataset = CloneDataset(
            functions_path=os.path.join(self.hparams.root, "functions"),
            pairs_path=os.path.join(self.hparams.root, "bcb_pair_ids.pkl"),
        )

        n = len(dataset)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [n - 10000, 10000])

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          collate_fn=collate)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset,
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers,
                          collate_fn=collate)

    #     def test_dataloader(self):
    #         # OPTIONAL
    #         return DataLoader(self.test_dataset,
    #                           batch_size=self.hparams.batch_size,
    #                           num_workers=self.hparams.workers,
    #                           collate_fn=collate)

    @staticmethod
    def add_model_specific_args():
        parser = ArgumentParser(add_help=False)

        parser.add_argument('--learning_rate', default=0.0001, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--workers', default='8', type=int)
        parser.add_argument('--num_classes', default='6', type=int)
        parser.add_argument('--num_features', default='384', type=int)
        parser.add_argument('--hidden_dim', default='284', type=int)

        parser.add_argument('--root', type=str, required=True)

        # training specific (for this model)
        parser.add_argument('--gpus', type=int, default=1, help='how many gpus')

        return parser


params = dict(
    learning_rate=0.0001,
    batch_size=64,
    workers=128,
    num_classes=6,
    num_features=384,
    hidden_dim=284,
    gpu=1,
    root="../data/",
    max_nb_epochs=2
)
from argparse import Namespace

hparams = Namespace(**params)
model = NetBasic(hparams)

model.prepare_data()
train_loader = model.train_dataloader()
val_loader = model.val_dataloader()

optimizer, scheduler = model.configure_optimizers()

for epoch in range(1, 201):

    for data in train_loader:
        optimizer.zero_grad()
        logs = model.training_step(data)
        loss = logs['loss']
        loss.backward()
        print(f"loss = {loss.item()}", end="\r")

    for data in val_loader:
        logs = model.validation_step(data)

    model.validation_epoch_end()

    scheduler.step(epoch)