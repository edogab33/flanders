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
from torchmetrics import Accuracy
import numpy as np


class LitMNIST(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, client=True):

        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.client = client
        self.params = []

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        #self.log("val_loss", loss, prog_bar=True)
        #self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        if self.client==False:
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)
        else:
            self.log("cl_test_loss", loss, prog_bar=True)
            self.log("cl_test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def load_data(client=True):
    # Training / validation set
    trainset = MNIST("", train=True, download=True, transform=transforms.ToTensor())

    if client:
        # Take a random subset of 600 samples
        ts = torch.randperm(len(trainset))[:1200]
        trainset = torch.utils.data.Subset(trainset, ts)
        mnist_train, mnist_val = random_split(trainset, [1000, 200])
    else:
        mnist_train, mnist_val = random_split(trainset, [55000, 5000])
        
    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True, num_workers=1)
    val_loader = DataLoader(mnist_val, batch_size=32, shuffle=False, num_workers=1)

    # Test set
    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader


def main() -> None:
    """Centralized training."""

    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Load model
    model = LitMNIST()

    # Train
    trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1)
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()