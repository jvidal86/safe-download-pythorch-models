"""
torchget — custom transforms & augmentation
============================================
`transform` and `target_transform` are passed straight through to the
underlying torchvision dataset, so anything torchvision supports works here.
"""

import torch
import torchvision.transforms as T

from torchget import fetch

# Typical training augmentation pipeline for CIFAR-10
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Evaluation: no augmentation, just normalize
eval_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_ds = fetch("cifar10", train=True, transform=train_tf)
test_ds = fetch("cifar10", train=False, transform=eval_tf)

# One-hot encode labels via target_transform
one_hot = lambda y: torch.nn.functional.one_hot(torch.tensor(y), num_classes=10)
train_onehot = fetch("cifar10", train=True, transform=train_tf, target_transform=one_hot)

_, label = train_onehot[0]
print("One-hot label:", label)
