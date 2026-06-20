"""
torchget
~~~~~~~~
Resilient dataset downloader for PyTorch / torchvision.

Quick start
-----------
>>> from torchget import fetch
>>> train_ds = fetch("cifar10")
>>> test_ds  = fetch("cifar10", train=False)

Low-level
---------
>>> from torchget import safe_download, extract
>>> safe_download("https://example.com/file.tar.gz", "data/file.tar.gz",
...               expected_md5="abc123")
>>> extract("data/file.tar.gz", "data/")

Registry
--------
>>> from torchget import list_datasets
>>> list_datasets()
['cifar10', 'cifar100', 'fashion_mnist', 'mnist', 'stl10']
"""

from .datasets import fetch
from .downloader import download, extract, safe_download
from .registry import list_datasets

__version__ = "0.1.1"
__all__ = [
    "fetch",
    "download",
    "safe_download",
    "extract",
    "list_datasets",
]
