from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import AdamW

from .model import UNet
from .utils import weights_norm_init


class UnetTrainer(pl.LightningModule):
    """Trainer class for U-net model."""

    def __init__(self, learning_rate: float, num_classes: int, ladder_size: int):
        super(UnetTrainer, self).__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model = UNet(num_channels=self.num_classes, ladder_size=ladder_size)
        self.model.apply(weights_norm_init)
        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.crit(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.crit(outputs, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.crit(outputs, labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
