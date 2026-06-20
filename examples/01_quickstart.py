"""
torchget — quick start
======================
Install:
    pip install git+https://github.com/jvidal86/safe-download-pythorch-models.git

`fetch` downloads (with resume + retries + MD5 verify), extracts, and returns a
ready-to-use torchvision dataset. If the data is already on disk and passes its
integrity check, it skips straight to loading — no network access.
"""

from torch.utils.data import DataLoader

from torchget import fetch, list_datasets

# What can I fetch?
print("Available:", list_datasets())
# ['cifar10', 'cifar100', 'fashion_mnist', 'mnist', 'stl10']

# Train / test splits (default transform = ToTensor + Normalize)
train_ds = fetch("cifar10", root="./data")
test_ds = fetch("cifar10", root="./data", train=False)
print(f"Train: {len(train_ds):,}   Test: {len(test_ds):,}")

# Plug straight into a DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

images, labels = next(iter(train_loader))
print("Batch:", images.shape, labels.shape)  # torch.Size([64, 3, 32, 32]) ...
