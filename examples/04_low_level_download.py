"""
torchget — low-level downloader for arbitrary files
===================================================
Use `safe_download` + `extract` directly for any URL, not just registered
datasets. You get resume, exponential-backoff retries, optional MD5
verification, and path-traversal-safe extraction.
"""

from torchget import safe_download, extract

# Download any file with retries + (optional) integrity check.
# On a dropped connection it resumes via an HTTP Range request instead of
# starting over. If expected_md5 is given and fails, the file is deleted and
# the download is retried cleanly.
path = safe_download(
    url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    dest="data/cifar-10-python.tar.gz",
    expected_md5="c58f30108f718f92721af3b95e74349a",  # omit to skip verification
    max_retries=5,
    show_progress=True,
)
print("Downloaded to:", path)

# Extract .tar.gz / .tgz / .tar.bz2 / .tar.xz / .zip — unsafe members
# (absolute paths or '..') are filtered out.
extract(path, dest_dir="data/", remove_after=False)
print("Extracted into data/")

# See progress in logs:
#   import logging; logging.basicConfig(level=logging.INFO)
