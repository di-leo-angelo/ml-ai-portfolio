"""CIFAR-10 dataset loading, train/val split, and preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset

# Per-channel mean/std of the CIFAR-10 training set (standard normalization).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

IMAGE_SIZE = 32
NUM_CLASSES = 10
SEED = 42
VAL_SIZE = 5_000
QUICK_TRAIN_SIZE = 4_000
QUICK_VAL_SIZE = 1_000


def make_transforms(augment: bool) -> T.Compose:
    """Build the image transform pipeline for one of the splits.

    Training uses data augmentation (random crops and horizontal flips);
    validation and test use plain tensor conversion, no augmentation.
    """
    if augment:
        return T.Compose(
            [
                T.RandomCrop(IMAGE_SIZE, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def load_splits(
    data_dir: Path,
    *,
    seed: int = SEED,
    quick: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train, validation, test) datasets.

    The validation split is carved out of the official 50,000-image training
    split (5,000 images, fixed seed); the official 10,000-image test split is
    left untouched for the final evaluation.
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=make_transforms(augment=True),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=make_transforms(augment=False),
    )

    if quick:
        n_train, n_val = QUICK_TRAIN_SIZE, QUICK_VAL_SIZE
    else:
        n_val = VAL_SIZE
        n_train = len(train_dataset) - n_val

    indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    train_subset = Subset(train_dataset, indices[:n_train])
    val_subset = Subset(train_dataset, indices[n_train : n_train + n_val])
    return train_subset, val_subset, test_dataset
