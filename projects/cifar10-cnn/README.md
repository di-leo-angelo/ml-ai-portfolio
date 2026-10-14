# CIFAR-10 CNN Classification

## 1. Problem and motivation

Image classification on CIFAR-10, the standard introductory benchmark in
computer vision. The goal here is not the best published number, but to
demonstrate a complete, modern training protocol end-to-end on CPU-only
hardware:

- a compact CNN with regularization (weight decay, dropout, data augmentation),
- learning-rate scheduling that reacts to the validation curve,
- early stopping with best-checkpoint restoration,
- reproducible evaluation on the untouched official test set.

## 2. Dataset

**CIFAR-10** (Krizhevsky, 2009): 60,000 32x32 color images in 10 balanced
classes (5,000 each): airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck. 50,000 training images, 10,000 test images.

- Source: official archive via torchvision (<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>)
- Terms: free for research and educational use.
- Local layout: `data/` (not committed; re-download with `scripts/download_data.py`).

Training carves a 5,000-image validation split out of the training set
(fixed seed); the 10,000-image test set is used only for the final
evaluation, so the reported number is an honest hold-out estimate.

## 3. Methodology

| Component | Choice |
|---|---|
| Architecture | 4 conv blocks (32/64/128/256 ch, 3x3 conv, BatchNorm, MaxPool 2x2), Flatten -> Linear(1024, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 10); ~1.7M parameters |
| Regularization | weight decay 1e-4 (AdamW), Dropout2d(0.2) per block, FC dropout 0.3, augmentation: RandomCrop(32, pad=4) + RandomHorizontalFlip |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, batch size 128 |
| LR scheduling | ReduceLROnPlateau on val loss: factor 0.5, patience 5, min lr 1e-6 |
| Early stopping | patience 10 on val loss; best checkpoint restored at the end |
| Normalization | per-channel CIFAR-10 train mean/std |

Evaluation: overall accuracy plus per-class accuracy on the 10,000-image
test set.

## 4. Main results

Full run on CPU (Intel i5-8250U, no GPU), seed 42. Early stopping halted
training at epoch 64 after val loss plateaued (patience 10); the
`ReduceLROnPlateau` scheduler halved the LR twice (1e-3 → 5e-4 → 2.5e-4).

| Metric | Value |
|---|---|
| Test accuracy (10,000 images) | **89.03%** (8,903/10,000) |
| Best val loss (epoch 54) | 0.3233 (val acc 88.86%) |
| Training time | ~4.5 h (64 epochs, ~4.3 min/epoch on CPU) |

Per-class test accuracy: truck 95.4%, automobile 95.3%, horse 94.7%,
frog 94.1%, ship 92.9%, deer 90.6%, airplane 90.1%, bird 83.5%, dog 83.4%,
cat 70.3% — cat/dog confusion dominates the remaining errors.

![Learning curve](reports/figures/learning_curve_best.png)

## 5. Limitations and possible improvements

- CPU-only training: a GPU would cut the run to ~20-30 min.
- No hyperparameter search beyond chosen defaults (e.g. LR range test, width search).
- Compact CNN: a small ResNet with skip connections would push accuracy ~1-2 points higher.
- Possible extensions: cutout/mixup, label smoothing, test-time augmentation.

## 6. Reproduction

From the repository root (Windows PowerShell):

```powershell
. .venv\Scripts\Activate.ps1
cd projects\cifar10-cnn\src

python -m cifar10.train --quick    # pipeline smoke test: 4k-image subset, 15 epochs
python -m cifar10.train            # full run: up to 100 epochs, early stopping
python -m cifar10.evaluate         # final test-set report (uses models/best.pt)
```

Unit tests (from the repository root):

```powershell
python -m pytest projects/cifar10-cnn/tests -v
```

Outputs: `models/best.pt` (checkpoint), `models/best_history.json`
(per-epoch metrics), `reports/figures/learning_curve_best.png`.
