"""Adapted from the PyTorch Lightning quickstart example.
Source: https://pytorchlightning.ai/ (2021/02/04)
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.params = []

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat

def load_data():
    # Training / validation set
    trainset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(trainset, [55000, 5000])
    train_loader = DataLoader(mnist_train, batch_size=24, shuffle=True, num_workers=4)
    val_loader = DataLoader(mnist_val, batch_size=24, shuffle=False, num_workers=4)

    # Test set
    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(testset, batch_size=24, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main() -> None:
    """Centralized training."""

    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Load model
    model = LitAutoEncoder()

    # Train
    trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()