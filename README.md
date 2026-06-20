# torchget

Resilient dataset downloader for PyTorch.  
Downloads large files with **resume support**, **exponential-backoff retries**,  
**MD5 integrity verification**, and **path-traversal-safe extraction**.

## Install

```bash
pip install torchget          # from PyPI (once published)
pip install -e .              # local editable install
```

## Quick start

```python
from torchget import fetch

# Downloads, verifies MD5, extracts — and skips all of it if the data
# is already on disk and passes integrity checks
train_ds = fetch("cifar10", root="./data")
test_ds  = fetch("cifar10", root="./data", train=False)

# Plug straight into DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
```

## Available datasets

```python
from torchget import list_datasets
print(list_datasets())
# ['cifar10', 'cifar100', 'fashion_mnist', 'mnist', 'stl10']
```

## Low-level API

```python
from torchget import safe_download, extract

# Download any URL with retries + MD5 check
safe_download(
    url="https://example.com/dataset.tar.gz",
    dest="data/dataset.tar.gz",
    expected_md5="abc123...",   # omit to skip verification
    max_retries=5,
)

# Extract safely (blocks path traversal attacks)
extract("data/dataset.tar.gz", dest_dir="data/", remove_after=True)
```

## `fetch()`

`fetch(name, root="./data", *, train=True, ...)` is the high-level entry point.
It returns a ready-to-use `torchvision` dataset, doing the minimum work needed:

1. **Verify first.** It tries to load the dataset from `root` with no download.
   torchvision checks the MD5 of every required file, so a missing or corrupt
   file is detected — not just an empty/partial folder. If everything is valid,
   `fetch` returns immediately and never touches the network.
2. **Download if needed.** Otherwise it downloads the archive with resume,
   exponential-backoff retries, and an MD5 check (CIFAR-10 uses a fast mirror of
   the canonical tarball).
3. **Extract** safely (path-traversal-filtered) and load the dataset.

```python
# Raw [0,1] images (e.g. for plotting) — override the default transform
import torchvision.transforms as T
raw = fetch("cifar10", transform=T.ToTensor())

# Test split, drop the archive once extracted
test = fetch("cifar10", train=False, remove_archive=True)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `name` | required | Dataset name (`"cifar10"`, `"mnist"`, …) |
| `root` | `"./data"` | Local storage directory |
| `train` | `True` | Training vs test split (`split` for STL-10) |
| `transform` | `ToTensor + Normalize` | torchvision transform applied to each sample |
| `target_transform` | `None` | Transform applied to each label |
| `max_retries` | `5` | Retry attempts on connection error |
| `show_progress` | `True` | tqdm progress bar |
| `remove_archive` | `False` | Delete `.tar.gz` after extraction |

Returns a `torchvision.datasets.VisionDataset`, ready for `DataLoader`.

## Logging

```python
import logging
logging.basicConfig(level=logging.INFO)   # see download progress in logs
```

## How resume works

On a broken connection, the partially downloaded file stays on disk.  
The next call sends an HTTP `Range: bytes=N-` header, telling the server  
to continue from byte `N` — no data is re-downloaded.  
If MD5 fails after download, the file is deleted and a clean retry starts.
