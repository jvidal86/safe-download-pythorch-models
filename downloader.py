"""
torchget.downloader
~~~~~~~~~~~~~~~~~~~
Resilient file downloader with resume support, MD5 verification,
exponential-backoff retries, and path-traversal-safe extraction.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Default tunables ──────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 65_536          # 64 KB — granular progress on flaky links
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_CONNECT_TIMEOUT: float = 10.0    # seconds to establish TCP connection
DEFAULT_READ_TIMEOUT: float = 30.0       # seconds between received chunks


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _md5(filepath: Path) -> str:
    """Return the hex MD5 digest of *filepath*."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8_192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Filter tar members to prevent path-traversal attacks."""
    safe = []
    for m in tf.getmembers():
        if os.path.isabs(m.name) or ".." in Path(m.name).parts:
            logger.warning("Skipping unsafe tar member: %s", m.name)
            continue
        safe.append(m)
    return safe


# ── Core download ─────────────────────────────────────────────────────────────

def download(
    url: str,
    dest: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    show_progress: bool = True,
) -> None:
    """
    Download *url* to *dest*, resuming from an existing partial file if present.

    Raises
    ------
    requests.HTTPError
        On non-2xx responses that cannot be resumed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    logger.debug("GET %s (offset=%d)", url, existing)
    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )

    # 416 = Range Not Satisfiable → file is already complete on disk
    if response.status_code == 416:
        logger.info("Server confirmed file is already fully downloaded.")
        return

    response.raise_for_status()

    remote_length = int(response.headers.get("content-length", 0))
    total = existing + remote_length
    mode = "ab" if existing else "wb"

    if existing:
        logger.info("Resuming from %.1f MB", existing / 1e6)
    else:
        logger.info("Starting download: %s", dest.name)

    with open(dest, mode) as fh, tqdm(
        total=total or None,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
        disable=not show_progress,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))


# ── Safe download (retries + integrity check) ─────────────────────────────────

def safe_download(
    url: str,
    dest: Path | str,
    *,
    expected_md5: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    show_progress: bool = True,
) -> Path:
    """
    Download *url* to *dest* with retries, resume support, and optional MD5
    verification.

    Parameters
    ----------
    url:
        Remote URL to download.
    dest:
        Local destination path.
    expected_md5:
        Hex MD5 digest to verify after download.  Skip verification if *None*.
    max_retries:
        How many times to retry on connection errors.
    chunk_size:
        Bytes per read chunk (default 64 KB).
    connect_timeout / read_timeout:
        TCP connect and per-chunk read timeouts in seconds.
    show_progress:
        Show a tqdm progress bar.

    Returns
    -------
    Path
        The local path of the downloaded file.

    Raises
    ------
    RuntimeError
        If the download fails after *max_retries* attempts.
    ValueError
        If MD5 verification fails after a successful download.
    """
    dest = Path(dest)

    # ── Already present? ──────────────────────────────────────────────────────
    if dest.exists():
        if expected_md5 is None:
            logger.info("File already present (no MD5 check): %s", dest)
            return dest

        logger.info("File found — verifying MD5...")
        if _md5(dest) == expected_md5:
            logger.info("✓ MD5 OK — skipping download.")
            return dest

        logger.warning("✗ MD5 mismatch — removing corrupt file and re-downloading.")
        dest.unlink()

    # ── Download loop ─────────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        try:
            download(
                url,
                dest,
                chunk_size=chunk_size,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                show_progress=show_progress,
            )

            if expected_md5 is not None:
                if _md5(dest) != expected_md5:
                    logger.warning("✗ MD5 mismatch after download — removing and retrying.")
                    dest.unlink()
                    continue
                logger.info("✓ MD5 verified.")

            return dest

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts: {url}"
                ) from exc

            wait = 2 ** attempt          # 2 s, 4 s, 8 s, 16 s …
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %ds…",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"Download failed after {max_retries} attempts: {url}")


# ── Archive extraction ────────────────────────────────────────────────────────

def extract(
    archive: Path | str,
    dest_dir: Path | str,
    *,
    remove_after: bool = False,
) -> Path:
    """
    Extract a .tar.gz, .tgz, or .zip archive to *dest_dir*.

    Parameters
    ----------
    archive:
        Path to the archive file.
    dest_dir:
        Directory to extract into (created if absent).
    remove_after:
        Delete the archive after successful extraction.

    Returns
    -------
    Path
        The extraction directory.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    name = archive.name.lower()

    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        logger.info("Extracting tar archive: %s → %s", archive.name, dest_dir)
        with tarfile.open(archive) as tf:
            tf.extractall(path=dest_dir, members=_safe_members(tf))

    elif name.endswith(".zip"):
        logger.info("Extracting zip archive: %s → %s", archive.name, dest_dir)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(path=dest_dir)

    else:
        raise ValueError(f"Unsupported archive format: {archive.suffix}")

    if remove_after:
        archive.unlink()
        logger.info("Removed archive: %s", archive)

    return dest_dir
