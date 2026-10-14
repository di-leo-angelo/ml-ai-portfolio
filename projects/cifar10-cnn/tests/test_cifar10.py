"""Unit tests for the CIFAR-10 CNN components (no dataset download needed)."""

from __future__ import annotations

import torch
from cifar10.data import IMAGE_SIZE, NUM_CLASSES, make_transforms
from cifar10.model import CifarCNN
from cifar10.train import evaluate, train_epoch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


def _toy_loader(batch_size: int = 32) -> DataLoader:
    """Small synthetic image batch loader for fast plumbing tests."""
    images = torch.randn(64, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (64,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_model_output_shape() -> None:
    model = CifarCNN()
    logits = model(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (4, NUM_CLASSES)


def test_model_parameter_count() -> None:
    model = CifarCNN()
    assert 500_000 < model.count_parameters() < 5_000_000


def test_transforms_produce_normalized_tensor() -> None:
    image = Image.fromarray(
        torch.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8).numpy()
    )
    tensor = make_transforms(augment=True)(image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert abs(tensor.mean().item()) < 0.5
    assert 0.5 < tensor.std().item() < 2.0


def test_train_epoch_and_evaluate_smoke() -> None:
    model = CifarCNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _toy_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    loss = train_epoch(model, loader, optimizer, loss_fn, device)
    assert torch.isfinite(torch.tensor(loss))
    val_loss, val_acc = evaluate(model, loader, loss_fn, device)
    assert torch.isfinite(torch.tensor(val_loss))
    assert 0.0 <= val_acc <= 1.0
