"""Configuration objects."""

import json
from typing import Tuple

import torch
from pydantic_settings import BaseSettings


class UnetConfig(BaseSettings):
    """Main class for U-net configuration."""

    name: str = "U-net"
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-3
    save_dir: str = ""
    log_dir: str = ""
    train_size: int = 70
    val_size: int = 15
    test_size: int = 15
    target_size: Tuple[int, int] = (256, 256)
    num_classes: int = 3
    num_workers: int = 5
    ladder_size: int = 4
    data_input_dir: str = "./data/images/"
    data_labels_dir: str = "./data/annotations/trimaps"
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
