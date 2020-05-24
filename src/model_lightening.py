import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from code_parser import *
from dataset import CloneDataset


class NetBasic(pl.LightningModule):

    def __init__(self, hparams):
        super(NetBasic, self).__init__()

        self.hparams = hparams

        self.build_model()

    def build_model(self):
        self.conv1 = GraphConv(self.hparams.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.hparams.num_classes)

    def forward_(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

    def forward(self, data):
        return self.forward_(data)

    def training_step(self, data, batch_idx):
        output = self.forward(data)
        loss = F.nll_loss(output, data.y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, data, batch_idx):
        output = self.forward(data)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        acc = pred.eq(data.y).type(torch.float32).mean()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_step(self, data, batch_idx):
        output = self.forward(data)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        acc = pred.eq(data.y).type(torch.float32).mean()
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc}
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        dataset = CloneDataset(root=self.hparams.root,
                               functions_path=os.path.join(self.hparams.root, "functions"),
                               pairs_path=os.path.join(self.hparams.root, "bcb_pair_ids.pkl"),
                               transform=T.NormalizeFeatures())

        dataset = dataset.shuffle()
        n = (len(dataset) + 9) // 10
        self.test_dataset = dataset[:n]
        self.val_dataset = dataset[n:2 * n]
        self.train_dataset = dataset[2 * n:]

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--learning_rate', default=0.0001, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--workers', default='8', type=int)
        parser.add_argument('--num_classes', default='6', type=int)
        parser.add_argument('--num_features', default='384', type=int)

        parser.add_argument('--root', type=str, required=True)

        # training specific (for this model)
        parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
        parser.add_argument('--max_nb_epochs', default=2, type=int, required=True)

        return parser


if __name__ == '__main__':
    params = dict(
        learning_rate=0.0001,
        batch_size=8,
        workers=0,
        num_classes=6,
        num_features=384,
        gpu=1,
        root="../data/",
        max_nb_epochs=2
    )
    from argparse import Namespace
    hparams = Namespace(**params)

    model = NetBasic(hparams)

    from pytorch_lightning import Trainer

    # most basic trainer, uses good defaults
    trainer = Trainer()
    trainer.fit(
        model,
        gpus=hparams.gpus
    )
