"""
torchget.datasets
~~~~~~~~~~~~~~~~~
High-level API: fetch a registered dataset by name and return it as a
torchvision Dataset, ready for use in a DataLoader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torchvision.datasets as tvd
import torchvision.transforms as T

from .downloader import extract, safe_download
from .registry import DatasetInfo, get as registry_get

logger = logging.getLogger(__name__)

# Map registry key → torchvision class + sensible default transform
_TV_MAP: dict[str, type] = {
    "cifar10": tvd.CIFAR10,
    "cifar100": tvd.CIFAR100,
    "mnist": tvd.MNIST,
    "fashion_mnist": tvd.FashionMNIST,
    "stl10": tvd.STL10,
}

_DEFAULT_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])


def fetch(
    name: str,
    root: str | Path = "./data",
    *,
    train: bool = True,
    transform=_DEFAULT_TRANSFORM,
    target_transform=None,
    max_retries: int = 5,
    show_progress: bool = True,
    remove_archive: bool = False,
) -> tvd.VisionDataset:
    """
    Download (if needed) and return a torchvision dataset.

    Parameters
    ----------
    name:
        Dataset name as registered in ``torchget.registry`` (e.g. ``"cifar10"``).
    root:
        Local directory where datasets are stored.
    train:
        Whether to load the training split (``False`` loads test/val).
    transform:
        torchvision transform applied to each sample.
    target_transform:
        Transform applied to each label.
    max_retries:
        Download retry attempts on connection failure.
    show_progress:
        Show tqdm progress bar during download.
    remove_archive:
        Delete the downloaded archive after extraction.

    Returns
    -------
    torchvision.datasets.VisionDataset
        Ready to pass to ``torch.utils.data.DataLoader``.

    Examples
    --------
    >>> from torchget import fetch
    >>> train_ds = fetch("cifar10", root="./data")
    >>> test_ds  = fetch("cifar10", root="./data", train=False)
    """
    root = Path(root)
    info: DatasetInfo = registry_get(name)
    key = name.lower().replace("-", "_")

    tv_class = _TV_MAP.get(key)
    if tv_class is None:
        raise NotImplementedError(
            f"No torchvision mapping for '{name}'. "
            "Use torchget.downloader.safe_download() directly."
        )

    def _load() -> tvd.VisionDataset:
        """Construct the dataset with download=False.

        torchvision validates the MD5 of every required file in the extracted
        directory and raises RuntimeError if anything is missing or corrupt.
        """
        # STL10 uses 'split' instead of 'train'
        if key == "stl10":
            return tv_class(
                root=str(root),
                split="train" if train else "test",
                transform=transform,
                target_transform=target_transform,
                download=False,
            )
        return tv_class(
            root=str(root),
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

    # ── Already present AND valid? Verify contents, skip download ──────────────
    try:
        ds = _load()
        logger.info("Dataset verified on disk (%s) — skipping download.", key)
        return ds
    except (RuntimeError, FileNotFoundError):
        logger.info("Dataset missing or failed integrity check — downloading %s.", key)

    archive_path = root / info["filename"]

    # ── Step 1: Download archive (resume + retry + MD5 verify) ────────────────
    safe_download(
        info["url"],
        archive_path,
        expected_md5=info["md5"],
        max_retries=max_retries,
        show_progress=show_progress,
    )

    # ── Step 2: Extract ───────────────────────────────────────────────────────
    logger.info("Extracting %s…", archive_path.name)
    extract(archive_path, root, remove_after=remove_archive)

    # ── Step 3: Load (download=False — we did it ourselves) ───────────────────
    return _load()
