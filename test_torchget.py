"""
tests/test_torchget.py
~~~~~~~~~~~~~~~~~~~~~~
Unit tests for torchget using pytest + responses (mock HTTP).
Run: pytest tests/ -v
"""

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path

import pytest
import responses as rsps_lib

from torchget.downloader import _md5, _safe_members, download, extract, safe_download
from torchget.registry import get as registry_get, list_datasets


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tar(tmp_path: Path, filename: str = "archive.tar.gz") -> tuple[Path, bytes]:
    """Create a small .tar.gz and return (path, raw_bytes)."""
    inner = tmp_path / "inner.txt"
    inner.write_text("hello torchget")

    archive = tmp_path / filename
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(inner, arcname="inner.txt")

    return archive, archive.read_bytes()


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── MD5 ───────────────────────────────────────────────────────────────────────

def test_md5_correct(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc123")
    assert _md5(f) == hashlib.md5(b"abc123").hexdigest()


def test_md5_wrong(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc123")
    assert _md5(f) != "deadbeef"


# ── Safe tar members ──────────────────────────────────────────────────────────

def test_safe_members_filters_traversal(tmp_path):
    archive, _ = _make_tar(tmp_path)

    with tarfile.open(archive) as tf:
        members = _safe_members(tf)

    # All normal members should pass
    assert all(not os.path.isabs(m.name) for m in members)
    assert all(".." not in Path(m.name).parts for m in members)


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_get_cifar10():
    info = registry_get("cifar10")
    assert "url" in info
    assert "md5" in info
    assert "filename" in info


def test_registry_case_insensitive():
    assert registry_get("CIFAR10") == registry_get("cifar10")


def test_registry_unknown_raises():
    with pytest.raises(KeyError, match="Unknown dataset"):
        registry_get("nonexistent_dataset")


def test_list_datasets_returns_sorted():
    names = list_datasets()
    assert names == sorted(names)
    assert "cifar10" in names


# ── download() ────────────────────────────────────────────────────────────────

@rsps_lib.activate
def test_download_fresh(tmp_path):
    content = b"x" * 1024
    rsps_lib.add(rsps_lib.GET, "https://example.com/file.bin", body=content, status=200)

    dest = tmp_path / "file.bin"
    download("https://example.com/file.bin", dest, show_progress=False)

    assert dest.read_bytes() == content


@rsps_lib.activate
def test_download_resumes(tmp_path):
    partial = b"AAA"
    remainder = b"BBB"
    dest = tmp_path / "file.bin"
    dest.write_bytes(partial)

    rsps_lib.add(
        rsps_lib.GET,
        "https://example.com/file.bin",
        body=remainder,
        status=206,
        headers={"content-length": str(len(remainder))},
    )

    download("https://example.com/file.bin", dest, show_progress=False)
    assert dest.read_bytes() == partial + remainder


@rsps_lib.activate
def test_download_skips_when_416(tmp_path):
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"complete")

    rsps_lib.add(rsps_lib.GET, "https://example.com/file.bin", status=416)
    download("https://example.com/file.bin", dest, show_progress=False)

    # File must be untouched
    assert dest.read_bytes() == b"complete"


# ── safe_download() ───────────────────────────────────────────────────────────

@rsps_lib.activate
def test_safe_download_skips_if_md5_ok(tmp_path):
    content = b"valid content"
    dest = tmp_path / "file.bin"
    dest.write_bytes(content)

    md5 = _md5_bytes(content)
    # No HTTP mock registered — should not hit the network
    result = safe_download("https://example.com/file.bin", dest,
                           expected_md5=md5, show_progress=False)
    assert result == dest


@rsps_lib.activate
def test_safe_download_removes_corrupt_and_redownloads(tmp_path):
    content = b"good content"
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"corrupt data")

    rsps_lib.add(rsps_lib.GET, "https://example.com/file.bin",
                 body=content, status=200)

    md5 = _md5_bytes(content)
    safe_download("https://example.com/file.bin", dest,
                  expected_md5=md5, show_progress=False)

    assert dest.read_bytes() == content


# ── extract() ────────────────────────────────────────────────────────────────

def test_extract_tar_gz(tmp_path):
    archive, _ = _make_tar(tmp_path)
    out_dir = tmp_path / "out"
    extract(archive, out_dir)
    assert (out_dir / "inner.txt").exists()


def test_extract_removes_archive(tmp_path):
    archive, _ = _make_tar(tmp_path)
    out_dir = tmp_path / "out"
    extract(archive, out_dir, remove_after=True)
    assert not archive.exists()


def test_extract_unsupported_format(tmp_path):
    bad = tmp_path / "file.rar"
    bad.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported archive"):
        extract(bad, tmp_path / "out")
