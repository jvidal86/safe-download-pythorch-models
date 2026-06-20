"""
torchget.registry
~~~~~~~~~~~~~~~~~
Known dataset URLs and MD5 checksums.
Each entry is a dict with keys: url, md5, filename, extracted_dir.
"""

from __future__ import annotations

from typing import TypedDict


class DatasetInfo(TypedDict):
    url: str
    md5: str
    filename: str
    extracted_dir: str  # expected folder name after extraction


REGISTRY: dict[str, DatasetInfo] = {
    # ── Vision ────────────────────────────────────────────────────────────────
    "cifar10": DatasetInfo(
        # Fast mirror of the canonical Toronto tarball (byte-identical, MD5-verified).
        # Original (slow ~40 kB/s): https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        url="https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a",
        filename="cifar-10-python.tar.gz",
        extracted_dir="cifar-10-batches-py",
    ),
    "cifar100": DatasetInfo(
        url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
        filename="cifar-100-python.tar.gz",
        extracted_dir="cifar-100-python",
    ),
    "mnist": DatasetInfo(
        url="https://ossci-datasets.s3.amazonaws.com/mnist/MNIST/raw/train-images-idx3-ubyte.gz",
        md5="f68b3c2dcbeaaa9fbdd348bbdeb94873",
        filename="train-images-idx3-ubyte.gz",
        extracted_dir="MNIST/raw",
    ),
    "fashion_mnist": DatasetInfo(
        url="http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz",
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        filename="fashion-mnist-train-images.gz",
        extracted_dir="FashionMNIST/raw",
    ),
    "stl10": DatasetInfo(
        url="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
        md5="91f7769df0f17e558f3565bffb0c7dfb",
        filename="stl10_binary.tar.gz",
        extracted_dir="stl10_binary",
    ),
}


def get(name: str) -> DatasetInfo:
    """
    Retrieve dataset info by name (case-insensitive).

    Raises
    ------
    KeyError
        If *name* is not found in the registry.
    """
    key = name.lower().replace("-", "_")
    if key not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return REGISTRY[key]


def list_datasets() -> list[str]:
    """Return sorted list of all registered dataset names."""
    return sorted(REGISTRY.keys())
