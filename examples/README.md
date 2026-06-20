# torchget — usage examples

Runnable examples for [**torchget**](../), a resilient dataset downloader for
PyTorch: resume, retries, MD5 verification, and path-traversal-safe extraction.

## Install

```bash
pip install git+https://github.com/jvidal86/safe-download-pythorch-models.git
# or, from a checkout of this repo:
pip install -e .
```

## Examples

| File | What it shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Fetch CIFAR-10 and feed a `DataLoader` |
| [`02_visualize_cifar10.py`](02_visualize_cifar10.py) | Plot one sample per class with a raw `ToTensor` transform |
| [`03_custom_transforms.py`](03_custom_transforms.py) | Training augmentation, eval transform, one-hot labels |
| [`04_low_level_download.py`](04_low_level_download.py) | `safe_download` + `extract` for any URL |

> In Google Colab, prefix install/shell commands with `!`, and after
> (re)installing, **Runtime → Restart session** so the new module is loaded.
