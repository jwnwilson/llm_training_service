from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from pathlib import Path

from adapters.storage.local import LocalStorageAdapter
from adapters.storage.s3 import S3StorageAdapter

from adapters.storage.paths import standalone_model_key, run_model_key

__all__ = [
    "LocalStorageAdapter",
    "S3StorageAdapter",
    "upload_model",
    "download_model",
    "standalone_model_key",
    "run_model_key",
]


def upload_model(storage, local_path: Path, key: str) -> str:
    """Gzip-compress a GGUF and upload it to storage.

    Appends `.gz` to the key if not already present.
    Returns the actual storage key used (always ends in `.gz`).
    """
    if not key.endswith(".gz"):
        key = key + ".gz"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".gguf.gz")
    gz_path = Path(tmp_name)
    try:
        os.close(tmp_fd)
        with open(local_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
        storage.upload(gz_path, key)
    finally:
        gz_path.unlink(missing_ok=True)
    return key


def download_model(storage, key: str, dest: Path) -> None:
    """Download a model from storage to *dest*, decompressing `.gz` keys automatically."""
    if key.endswith(".gz"):
        gz_path = dest.parent / (dest.name + ".gz")
        storage.download(key, gz_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)
    else:
        storage.download(key, dest)
