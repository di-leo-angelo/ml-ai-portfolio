# Dataset

This project uses **CIFAR-10**: 60,000 32x32 color images in 10 balanced
classes (5,000 each), 50,000 for training and 10,000 for testing.

- Source: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz> (fetched automatically by torchvision)
- Terms: free for research and educational use.

Download reproducibly from the repository root:

```powershell
python projects/cifar10-cnn/scripts/download_data.py
```

This creates (and validates) `data/cifar-10-batches-py/`. The archive is
~170 MB and is not committed. Training automatically carves a 5,000-image
validation split out of the training set (fixed seed); the test set is used
only for the final evaluation.
