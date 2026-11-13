"""Integration tests for named dataset CRUD endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.orm import Session
from adapters.database.dataset_store import SQLAlchemyDatasetStore
from adapters.database.uow import SQLAlchemyUnitOfWork
from interactors.api.app import app
from interactors.api.deps import clear_storage, configure_storage, get_uow

VALID_JSONL = b'{"prompt": "hello", "completion": "world"}\n'


@pytest_asyncio.fixture
async def client(db_engine):
    storage = MagicMock()
    storage.upload = MagicMock()
    storage.delete = MagicMock()
    configure_storage(storage)

    def override_get_uow():
        uow = SQLAlchemyUnitOfWork(db_engine)
        with uow.transaction():
            yield uow

    app.dependency_overrides[get_uow] = override_get_uow

    session = Session(db_engine)
    dataset_store = SQLAlchemyDatasetStore(session)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, storage, dataset_store

    session.close()
    app.dependency_overrides.pop(get_uow, None)
    clear_storage()


class TestListDatasets:
    @pytest.mark.asyncio
    async def test_returns_empty_list_initially(self, client):
        c, _, _ = client
        resp = await c.get("/api/datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_created_datasets(self, client):
        c, _, _ = client
        await c.post(
            "/api/datasets",
            data={"name": "my-train", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        resp = await c.get("/api/datasets")
        assert resp.status_code == 200
        body = resp.json()["items"]
        assert len(body) == 1
        assert body[0]["name"] == "my-train"
        assert body[0]["dataset_type"] == "train"


class TestCreateDataset:
    @pytest.mark.asyncio
    async def test_creates_record_and_returns_201(self, client):
        c, _, _ = client
        resp = await c.post(
            "/api/datasets",
            data={"name": "train-v1", "dataset_type": "train", "description": "first version"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "train-v1"
        assert body["dataset_type"] == "train"
        assert body["description"] == "first version"
        assert body["id"]

    @pytest.mark.asyncio
    async def test_uploads_to_storage(self, client):
        c, storage, _ = client
        await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "eval"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert storage.upload.called

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("dataset_type", "filename"),
        [("train", "train.jsonl"), ("eval", "eval.jsonl")],
    )
    async def test_storage_key_filename_matches_dataset_type(self, client, dataset_type, filename):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": dataset_type},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        uploaded_key = storage.upload.call_args[0][1]
        assert uploaded_key.startswith("dataset/")
        assert uploaded_key.endswith(f"/{filename}")
        # The stored record must point at the object that was actually uploaded.
        assert resp.json()["key"] == uploaded_key

    @pytest.mark.asyncio
    async def test_storage_key_is_unique_per_dataset(self, client):
        c, storage, _ = client
        for _ in range(2):
            await c.post(
                "/api/datasets",
                data={"name": "ds", "dataset_type": "eval"},
                files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
            )
        first_key, second_key = (call.args[1] for call in storage.upload.call_args_list)
        assert first_key != second_key

    @pytest.mark.asyncio
    async def test_empty_file_returns_400(self, client):
        c, _, _ = client
        resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(b""), "application/octet-stream")},
        )
        assert resp.status_code == 400


class TestGetDataset:
    @pytest.mark.asyncio
    async def test_returns_created_dataset(self, client):
        c, _, _ = client
        create_resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        dataset_id = create_resp.json()["id"]
        get_resp = await c.get(f"/api/datasets/{dataset_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == dataset_id

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_id(self, client):
        c, _, _ = client
        resp = await c.get("/api/datasets/no-such-id")
        assert resp.status_code == 404


class TestDeleteDataset:
    @pytest.mark.asyncio
    async def test_returns_204(self, client):
        c, _, _ = client
        create_resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        dataset_id = create_resp.json()["id"]
        del_resp = await c.delete(f"/api/datasets/{dataset_id}")
        assert del_resp.status_code == 204

    @pytest.mark.asyncio
    async def test_removes_from_store(self, client):
        c, _, _ = client
        create_resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        dataset_id = create_resp.json()["id"]
        await c.delete(f"/api/datasets/{dataset_id}")
        get_resp = await c.get(f"/api/datasets/{dataset_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_calls_storage_delete(self, client):
        c, storage, _ = client
        create_resp = await c.post(
            "/api/datasets",
            data={"name": "ds", "dataset_type": "train"},
            files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        dataset_id = create_resp.json()["id"]
        await c.delete(f"/api/datasets/{dataset_id}")
        assert storage.delete.called

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_id(self, client):
        c, _, _ = client
        resp = await c.delete("/api/datasets/no-such-id")
        assert resp.status_code == 404
