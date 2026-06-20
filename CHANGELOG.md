# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `examples/` folder with runnable scripts: quickstart, CIFAR-10 visualization,
  custom transforms / one-hot labels, and low-level `safe_download` + `extract`.
- Expanded README documentation for `fetch()` (verify-first behavior,
  `target_transform`, return type, transform-override examples).

### Changed
- `fetch()` now verifies dataset contents before skipping a download: it tries
  to load with `download=False` and relies on torchvision's per-file MD5 check,
  so a partial or corrupt extraction triggers a clean re-download instead of
  being silently trusted.
- CIFAR-10 downloads from a fast, byte-identical mirror (~887 kB/s vs the
  original ~40 kB/s host); the MD5 pin is unchanged so integrity is still
  verified.

### Removed
- Unused `extracted_dir` field from `DatasetInfo` and all registry entries.

## [0.1.0]

### Added
- Initial release of **torchget**.
- `fetch(name, ...)` high-level API returning a ready-to-use torchvision dataset.
- `safe_download()` with resume (HTTP `Range`), exponential-backoff retries, and
  optional MD5 verification.
- `extract()` supporting `.tar.gz` / `.tgz` / `.tar.bz2` / `.tar.xz` / `.zip`
  with path-traversal-safe member filtering.
- Dataset registry: `cifar10`, `cifar100`, `mnist`, `fashion_mnist`, `stl10`.
- `list_datasets()` helper.

[Unreleased]: https://github.com/jvidal86/safe-download-pythorch-models/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jvidal86/safe-download-pythorch-models/releases/tag/v0.1.0
