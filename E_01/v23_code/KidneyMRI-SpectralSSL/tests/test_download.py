from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import requests

from kidney_ssl.data.download import DownloadSpec, _download, _safe_extract
from kidney_ssl.utils import md5_file


def test_safe_extract_rejects_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        _safe_extract(archive, tmp_path / "output")


def test_existing_file_is_verified_and_not_overwritten(tmp_path: Path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"expected")
    spec = DownloadSpec("file.bin", "https://invalid.example", md5_file(target), False)
    assert _download(spec, target) == "verified-existing"
    assert target.read_bytes() == b"expected"


def test_existing_wrong_md5_stops(tmp_path: Path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"wrong")
    spec = DownloadSpec("file.bin", "https://invalid.example", "0" * 32, False)
    with pytest.raises(RuntimeError, match="not overwritten"):
        _download(spec, target)


def test_http_failure_leaves_no_partial_file(tmp_path: Path, monkeypatch):
    class FailedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            raise requests.HTTPError("synthetic HTTP failure")

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FailedResponse())
    target = tmp_path / "failed.bin"
    spec = DownloadSpec("failed.bin", "https://invalid.example", "0" * 32, False)
    with pytest.raises(requests.HTTPError, match="synthetic"):
        _download(spec, target)
    assert not target.exists()
    assert not target.with_suffix(".bin.part").exists()
