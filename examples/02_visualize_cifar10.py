"""
torchget — visualize one sample per CIFAR-10 class
==================================================
Use a raw ToTensor() transform (values in [0, 1]) so the images display
correctly with matplotlib — the default fetch transform also normalizes,
which is great for training but looks washed out when plotted.
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T

from torchget import fetch

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# Raw [0,1] images for display (override the default Normalize transform)
raw_train = fetch("cifar10", "./data", train=True, transform=T.ToTensor())
print(f"Train: {len(raw_train):,}")

targets = np.array(raw_train.targets)
fig, axes = plt.subplots(1, 10, figsize=(16, 2.2))
for c in range(10):
    idx = np.where(targets == c)[0][0]
    img, _ = raw_train[idx]
    axes[c].imshow(img.permute(1, 2, 0).numpy())
    axes[c].set_title(CIFAR_CLASSES[c], fontsize=9)
    axes[c].axis("off")
plt.suptitle("CIFAR-10 — one sample per class (32x32 RGB)", fontweight="bold")
plt.tight_layout()
plt.show()
