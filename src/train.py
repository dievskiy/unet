from __future__ import annotations

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .config import UnetConfig
from .data import get_datasets
from .unet.trainer import UnetTrainer

logger = logging.getLogger(__name__)


def start_training(config: UnetConfig):
    data_sizes = (config.train_size, config.val_size, config.test_size)

    train_dataset, val_dataset, test_dataset = get_datasets(
        config.data_input_dir, config.data_labels_dir, config.target_size, data_sizes
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.save_dir,
        filename="unet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=10,
        save_on_train_epoch_end=True,
    )

    wandb_logger = WandbLogger(project="Unet", log_model="all")

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        default_root_dir=config.log_dir,
        accelerator="cpu" if config.device.type == "cpu" else "gpu",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        logger=wandb_logger,
    )

    unet_trainer = UnetTrainer(
        learning_rate=config.lr,
        num_classes=config.num_classes,
        ladder_size=config.ladder_size,
    )

    trainer.fit(
        unet_trainer,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    trainer.test(
        unet_trainer,
        dataloaders=test_dataloader,
    )
