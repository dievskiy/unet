from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import Resize


def get_input_paths(input_dir: str) -> List[str]:
    return sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])


def get_labels_paths(target_dir: str) -> List[str]:
    return sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )


def get_resize_transform(target_size: Tuple[int, int]):
    return Resize(target_size)


class PetDataset(Dataset):
    def __init__(self, input_images, label_images, target_size, transform=None):
        assert len(input_images) == len(label_images)

        # [N, 3, H, W]
        self.images = torch.zeros((len(input_images), 3, target_size[0], target_size[1]))
        # [N, H, W]
        self.labels = torch.zeros((len(input_images), target_size[0], target_size[1]))

        for i, input_image in enumerate(input_images):
            # Some images have alpha channel, so force RGB
            img = decode_image(input_image, mode=ImageReadMode.RGB)

            if transform is not None:
                img = transform(img)

            self.images[i] = img

        self.images /= 255

        for i, label_image in enumerate(label_images):
            # Some images have alpha channel, so force RGB
            img = decode_image(label_image, mode=ImageReadMode.RGB)
            # For labels all channel will have same value, but we only need one
            img = img[:1, :, :]

            if transform is not None:
                img = transform(img)

            self.labels[i] = img

        # subtract 1 to align with ground truth labels 0, 1, 2
        self.labels -= 1
        self.labels = self.labels.long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self.images[idx]
        labels = self.labels[idx]
        return images, labels


def get_datasets(
    input_dir: str, labels_dir: str, target_size: Tuple[int, int], data_sizes: Tuple[int, int, int]
) -> Tuple[PetDataset, PetDataset, PetDataset]:
    transform = get_resize_transform(target_size)

    train_size, val_size, test_size = data_sizes

    assert train_size + val_size + test_size == 100, "Dataset splits should sum up to 100%"

    input_paths = get_input_paths(input_dir)
    label_paths = get_labels_paths(labels_dir)

    train_len = int(len(input_paths) * train_size / 100)
    val_len = int(len(input_paths) * val_size / 100)

    train_input_paths = input_paths[:train_len]
    train_label_paths = label_paths[:train_len]
    val_input_paths = input_paths[train_len : train_len + val_len]
    val_label_paths = label_paths[train_len : train_len + val_len]
    test_input_paths = input_paths[train_len + val_len :]
    test_label_paths = label_paths[train_len + val_len :]

    train_dataset = PetDataset(
        train_input_paths,
        train_label_paths,
        target_size,
        transform,
    )
    val_dataset = PetDataset(
        val_input_paths,
        val_label_paths,
        target_size,
        transform,
    )
    test_dataset = PetDataset(
        test_input_paths,
        test_label_paths,
        target_size,
        transform,
    )

    return train_dataset, val_dataset, test_dataset
