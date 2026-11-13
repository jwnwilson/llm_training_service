"""Dataset management endpoints.

Provides full CRUD for named dataset records backed by a DatasetStorePort,
plus legacy upload endpoints (POST /train, POST /eval) retained for
backwards compatibility.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from adapters.storage.paths import dataset_eval_key, dataset_train_key
from domain.models import DatasetConfig, DatasetRecord, DatasetType, PaginatedResponse, UserContext
from domain.ports import DatasetStorePort, StoragePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_dataset_store, get_storage

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_approved)],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 64 * 1024  # 64 KB read buffer
DATASET_KEY_BY_TYPE = {
    DatasetType.TRAIN: dataset_train_key,
    DatasetType.EVAL: dataset_eval_key,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DatasetUploadResult(BaseModel):
    key: str


async def _stream_to_tmp(file: UploadFile) -> tuple[Path, int]:
    """Stream *file* into a temp file; return (path, byte_count).

    Raises HTTPException 400 on empty file, 413 on oversized file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp_path = Path(tmp.name)
        size = 0
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds maximum upload size (50 MB)",
                )
            tmp.write(chunk)

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return tmp_path, size


async def _upload_to_storage(file: UploadFile, storage: StoragePort, key: str) -> int:
    """Upload *file* to *storage* under *key*; return byte count."""
    tmp_path, size = await _stream_to_tmp(file)
    try:
        storage.upload(tmp_path, key)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Storage upload failed: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    log.info("Uploaded dataset: key=%s bytes=%d", key, size)
    return size


# ---------------------------------------------------------------------------
# Named dataset CRUD  (fixed paths must come before /{dataset_id} parameter)
# ---------------------------------------------------------------------------

@router.post("/train", status_code=201, response_model=DatasetUploadResult)
async def upload_train_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
    user: UserContext = Depends(require_approved),
) -> DatasetUploadResult:
    """Legacy fixed-key train upload (backwards compat)."""
    key = "dataset/train.jsonl"
    await _upload_to_storage(file, storage, key)
    return DatasetUploadResult(key=key)


@router.post("/eval", status_code=201, response_model=DatasetUploadResult)
async def upload_eval_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
    user: UserContext = Depends(require_approved),
) -> DatasetUploadResult:
    """Legacy fixed-key eval upload (backwards compat)."""
    key = "dataset/eval.jsonl"
    await _upload_to_storage(file, storage, key)
    return DatasetUploadResult(key=key)


@router.get("", response_model=PaginatedResponse[DatasetRecord])
def list_datasets(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    user: UserContext = Depends(require_approved),
) -> PaginatedResponse[DatasetRecord]:
    offset = (page - 1) * limit
    items = dataset_store.list(owner_id=user.user_id, offset=offset, limit=limit)
    total = dataset_store.count(owner_id=user.user_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)


@router.post("", status_code=201, response_model=DatasetRecord)
async def create_dataset(
    name: str = Form(...),
    dataset_type: DatasetType = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    user: UserContext = Depends(require_approved),
) -> DatasetRecord:
    """Upload a file and create a named dataset record."""
    import uuid as _uuid

    dataset_id = str(_uuid.uuid4())
    key = DATASET_KEY_BY_TYPE[dataset_type](dataset_id)
    await _upload_to_storage(file, storage, key)

    config = DatasetConfig(
        name=name,
        description=description,
        dataset_type=dataset_type,
        key=key,
        owner_id=user.user_id,
    )
    return dataset_store.create(config)


@router.get("/{dataset_id}", response_model=DatasetRecord)
def get_dataset(
    dataset_id: str,
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    user: UserContext = Depends(require_approved),
) -> DatasetRecord:
    record = dataset_store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if record.owner_id is not None and record.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return record


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str,
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    storage: StoragePort = Depends(get_storage),
    user: UserContext = Depends(require_approved),
) -> None:
    record = dataset_store.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if record.owner_id is not None and record.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Remove the file from storage (best-effort — don't fail if already gone)
    try:
        storage.delete(record.key)
    except Exception:
        log.warning("Could not delete storage key %s for dataset %s", record.key, dataset_id)

    dataset_store.delete(dataset_id)
