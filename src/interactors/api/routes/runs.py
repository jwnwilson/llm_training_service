"""Run management and training trigger endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from domain.models import EvaluationData, PaginatedResponse, QualityReport, RunConfig, RunRecord, RunStatus, UserContext
from domain.ports import DatasetStorePort, ModelStorePort, RunStorePort, StoragePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_dataset_store, get_model_store, get_run_store, get_storage

log = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/runs",
    tags=["runs"],
    dependencies=[Depends(require_approved)],
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TriggerRunRequest(BaseModel):
    model_id: str
    name: str | None = None
    epochs: int | None = None
    patience: int | None = None
    warmup_ratio: float | None = None
    skip_generate: bool | None = None
    remote_backend: str | None = None
    base_model: str | None = None
    num_train_samples: int | None = None
    num_eval_samples: int | None = None
    train_dataset_id: str | None = None
    eval_dataset_id: str | None = None


class EvaluateRequest(BaseModel):
    remote_backend: str = ""
    remote_run_id: str = ""


class ExportRequest(BaseModel):
    remote_backend: str = ""
    remote_run_id: str = ""


class RunLogsResponse(BaseModel):
    logs: str | None
    source: str | None


# ---------------------------------------------------------------------------
# Diagnostics response schemas
# ---------------------------------------------------------------------------

class TemporalDetails(BaseModel):
    workflow_id: str
    temporal_run_id: str
    status: str
    start_time: str | None
    close_time: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=PaginatedResponse[RunRecord])
def list_runs(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> PaginatedResponse[RunRecord]:
    offset = (page - 1) * limit
    items = run_store.list(owner_id=user.user_id, offset=offset, limit=limit)
    total = run_store.count(owner_id=user.user_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)


@router.get("/{run_id}", response_model=RunRecord)
def get_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> RunRecord:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/evaluation", response_model=EvaluationData)
def get_run_evaluation(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> EvaluationData:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    quality_report: QualityReport | None = None
    report_path = Path(f"data/workflow/{run_id}/quality_report.json")
    if report_path.exists():
        try:
            quality_report = QualityReport.model_validate_json(report_path.read_text())
        except Exception:
            log.warning("Failed to parse quality report for run %s", run_id)

    return EvaluationData(
        run_id=run.id,
        status=run.status,
        eval_valid_pct=run.eval_valid_pct,
        quality_report=quality_report,
    )


_NO_CACHE_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}


@router.get("/{run_id}/logs")
def get_run_logs(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    storage: StoragePort = Depends(get_storage),
    user: UserContext = Depends(require_approved),
) -> JSONResponse:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    logs = storage.read_text(f"workflow/{run_id}/logs.txt")
    payload = RunLogsResponse(logs=logs or None, source="s3" if logs else None)
    return JSONResponse(content=payload.model_dump(), headers=_NO_CACHE_HEADERS)


@router.get("/{run_id}/logs/stream")
async def stream_run_logs(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    storage: StoragePort = Depends(get_storage),
    user: UserContext = Depends(require_approved),
) -> StreamingResponse:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    log_key = f"workflow/{run_id}/logs.txt"
    loop = asyncio.get_running_loop()

    async def _event_generator():
        sent_bytes = 0
        consecutive_empty = 0
        while True:
            current_run = await loop.run_in_executor(None, lambda: run_store.get(run_id))
            is_terminal = current_run is None or current_run.status in _TERMINAL_STATUSES

            offset = sent_bytes
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: storage.read_bytes_from(log_key, offset)
                )
            except Exception:
                log.warning("stream_run_logs: S3 read failed  run_id=%s  offset=%d", run_id, offset, exc_info=True)
                chunk = b""

            if chunk:
                consecutive_empty = 0
                sent_bytes += len(chunk)
                new_text = chunk.decode("utf-8", errors="replace")
                for line in new_text.splitlines():
                    yield f"data: {line}\n\n"
            else:
                consecutive_empty += 1
                if consecutive_empty == 1:
                    log.info(
                        "stream_run_logs: S3 returned empty  run_id=%s  key=%s  offset=%d  is_terminal=%s",
                        run_id, log_key, offset, is_terminal,
                    )

            if is_terminal:
                yield "event: done\ndata: stream closed\n\n"
                return

            try:
                await asyncio.sleep(3.0)
            except asyncio.CancelledError:
                return

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{run_id}/progress/stream")
async def stream_run_progress(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> StreamingResponse:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    loop = asyncio.get_running_loop()

    async def _event_generator():
        last_fraction: float | None = None
        while True:
            current_run = await loop.run_in_executor(None, lambda: run_store.get(run_id))
            is_terminal = current_run is None or current_run.status in _TERMINAL_STATUSES

            fraction = current_run.progress if current_run else None
            detail = (current_run.progress_detail or "") if current_run else ""

            if fraction is not None and fraction != last_fraction:
                last_fraction = fraction
                yield f"data: {json.dumps({'fraction': fraction, 'detail': detail})}\n\n"

            if is_terminal:
                yield "event: done\ndata: stream closed\n\n"
                return

            try:
                await asyncio.sleep(3.0)
            except asyncio.CancelledError:
                return

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/{run_id}", status_code=204)
def delete_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> None:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    run_store.delete(run_id)


_CANCELLABLE_STATUSES = frozenset({
    RunStatus.PENDING,
    RunStatus.GENERATING,
    RunStatus.TRAINING,
    RunStatus.EVALUATING,
    RunStatus.EXPORTING,
    RunStatus.RUNNING,
})

_TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


@router.post("/{run_id}/cancel", status_code=204)
async def cancel_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> None:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in _CANCELLABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Run cannot be cancelled (status={run.status.value})",
        )

    try:
        from temporalio.client import Client

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)
        handle = client.get_workflow_handle(run.workflow_id)
        await handle.cancel()
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to cancel Temporal workflow %s", run.workflow_id)
        raise HTTPException(status_code=500, detail="Failed to cancel workflow")

    run_store.update_status(run_id, RunStatus.CANCELLED)


@router.get("/{run_id}/temporal", response_model=TemporalDetails)
async def get_run_temporal(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> TemporalDetails:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        from temporalio.client import Client

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)
        handle = client.get_workflow_handle(run.workflow_id)
        desc = await handle.describe()
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to describe Temporal workflow %s", run.workflow_id)
        raise HTTPException(status_code=502, detail="Temporal unreachable")

    return TemporalDetails(
        workflow_id=desc.id,
        temporal_run_id=desc.run_id,
        status=desc.status.name,
        start_time=desc.start_time.isoformat() if desc.start_time else None,
        close_time=desc.close_time.isoformat() if desc.close_time else None,
    )


@router.post("/trigger", status_code=202)
async def trigger_run(
    body: TriggerRunRequest,
    store: ModelStorePort = Depends(get_model_store),
    run_store: RunStorePort = Depends(get_run_store),
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    user: UserContext = Depends(require_approved),
) -> dict[str, str]:
    model = store.get(body.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.owner_id is not None and model.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Model not found")

    # Resolve dataset storage keys if dataset IDs provided
    train_data = model.train_data
    eval_data = model.eval_data
    if body.train_dataset_id is not None:
        train_ds = dataset_store.get(body.train_dataset_id)
        if train_ds is None:
            raise HTTPException(status_code=404, detail="Training dataset not found")
        if train_ds.owner_id is not None and train_ds.owner_id != user.user_id:
            raise HTTPException(status_code=404, detail="Training dataset not found")
        train_data = train_ds.key
    if body.eval_dataset_id is not None:
        eval_ds = dataset_store.get(body.eval_dataset_id)
        if eval_ds is None:
            raise HTTPException(status_code=404, detail="Eval dataset not found")
        if eval_ds.owner_id is not None and eval_ds.owner_id != user.user_id:
            raise HTTPException(status_code=404, detail="Eval dataset not found")
        eval_data = eval_ds.key
    elif body.train_dataset_id is not None:
        # No explicit eval dataset — reuse the train dataset S3 key so remote
        # backends (K8s, RunPod, etc.) can download a valid eval file.
        # The model's eval_data field often holds a stale local path that
        # doesn't exist in S3, making this fallback essential for remote runs.
        eval_data = train_data

    epochs = body.epochs if body.epochs is not None else model.epochs
    patience = body.patience if body.patience is not None else model.patience
    warmup_ratio = body.warmup_ratio if body.warmup_ratio is not None else model.warmup_ratio
    skip_generate = body.skip_generate if body.skip_generate is not None else model.skip_generate
    remote_backend = body.remote_backend if body.remote_backend is not None else model.remote_backend
    base_model = body.base_model if body.base_model is not None else model.base_model
    num_train_samples = body.num_train_samples
    num_eval_samples = body.num_eval_samples
    if remote_backend == "local":
        remote_backend = ""

    # Remote backends cannot fall back to a local data dir — require an explicit
    # dataset when skip_generate=True and train_data is a local-only default path.
    if remote_backend and skip_generate and body.train_dataset_id is None:
        is_s3_key = (
            train_data.startswith("dataset/")
            or train_data.startswith("workflow/")
        )
        if not is_s3_key:
            raise HTTPException(
                status_code=422,
                detail="Remote training with skip_generate=True requires a train_dataset_id "
                       f"(model train_data '{train_data}' is a local path, not an S3 key).",
            )

    log.info(
        "Trigger run: model=%s epochs=%s patience=%s warmup_ratio=%s "
        "skip_generate=%s remote_backend=%s base_model=%s "
        "num_train_samples=%s num_eval_samples=%s",
        body.model_id, epochs, patience, warmup_ratio,
        skip_generate, remote_backend, base_model,
        num_train_samples, num_eval_samples,
    )

    # Build config blob — stored on the run record for auditability
    run_training_config = {
        "epochs": epochs,
        "patience": patience,
        "warmup_ratio": warmup_ratio,
        "skip_generate": skip_generate,
        "remote_backend": remote_backend or "local",
        "base_model": base_model,
        "num_train_samples": num_train_samples,
        "num_eval_samples": num_eval_samples,
        "train_data": train_data,
        "eval_data": eval_data,
    }

    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)

        workflow_id = f"training-{model.id}-{uuid.uuid4().hex[:8]}"
        run = run_store.create(RunConfig(
            model_id=model.id,
            workflow_id=workflow_id,
            name=body.name,
            training_config=run_training_config,
            train_dataset_id=body.train_dataset_id,
            eval_dataset_id=body.eval_dataset_id,
            owner_id=user.user_id,
        ))
        run_id = run.id

        Path(f"data/workflow/{run_id}").mkdir(parents=True, exist_ok=True)

        config = ExperimentConfig(
            experiment_name=model.name,
            model_id=model.id,
            model_name=model.name,
            run_id=run_id,
            epochs=epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            skip_generate=skip_generate,
            remote_backend=remote_backend,
            model=base_model,
            data_dir=f"data/workflow/{run_id}",
            output_dir=f"data/workflow/{run_id}/checkpoint",
            gguf_output=f"data/workflow/{run_id}/model.gguf",
            # Forward resolved S3 keys so the workflow (and remote jobs) use the correct
            # storage paths rather than deriving a local-only data_dir+/train.jsonl path.
            train_data=train_data or "",
            eval_data=eval_data or "",
            **({"train_size": num_train_samples} if num_train_samples is not None else {}),
            **({"eval_size": num_eval_samples} if num_eval_samples is not None else {}),
        )

        await client.start_workflow(
            TrainingPipelineWorkflow.run,
            config,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        log.info(
            "Training triggered: model=%s run_id=%s workflow_id=%s",
            body.model_id, run_id, workflow_id,
        )
        return {"workflow_id": workflow_id, "run_id": run_id}
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to trigger training workflow for model %s", body.model_id)
        raise HTTPException(status_code=500, detail="Failed to start training workflow")


@router.post("/{run_id}/activate", response_model=RunRecord)
def activate_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> RunRecord:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Run has not completed successfully (status={run.status.value})",
        )

    from adapters.inference import LlamaCppInferenceAdapter
    from adapters.storage import LocalStorageAdapter, download_model
    from adapters.storage.paths import run_model_key
    from interactors.api.deps import configure
    from interactors.temporal.activities import _get_storage

    try:
        storage = _get_storage()
    except RuntimeError:
        storage = LocalStorageAdapter()

    gguf_key = run_model_key(run_id)
    local_path = Path(f"data/workflow/{run_id}/model.gguf")
    try:
        download_model(storage, gguf_key, local_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load run model from storage: {exc}") from exc

    configure(LlamaCppInferenceAdapter(model_path=str(local_path)))
    log.info("Activated run %s — gguf=%s", run_id, local_path)
    return run


@router.post("/{run_id}/evaluate", status_code=202)
async def evaluate_run(
    run_id: str,
    body: EvaluateRequest = EvaluateRequest(),
    run_store: RunStorePort = Depends(get_run_store),
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> dict[str, str]:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    model = store.get(run.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found for this run")

    remote_backend = body.remote_backend or model.remote_backend
    if remote_backend == "local":
        remote_backend = ""

    workflow_id = f"evaluate-{run_id}-{uuid.uuid4().hex[:8]}"
    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import EvaluateWorkflow, EvaluateWorkflowConfig

        client = await Client.connect(os.getenv("TEMPORAL_HOST", "localhost:7233"))
        run_store.update(run_id, RunConfig(model_id=run.model_id, workflow_id=workflow_id))
        run_store.update_status(run_id, RunStatus.RUNNING)

        await client.start_workflow(
            EvaluateWorkflow.run,
            EvaluateWorkflowConfig(
                run_id=run_id,
                remote_backend=remote_backend,
                remote_run_id=body.remote_run_id,
                eval_data=model.eval_data,
                checkpoint_path=f"data/workflow/{run_id}/checkpoint",
                output_dir=f"data/workflow/{run_id}",
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        log.info("Eval workflow started: run_id=%s workflow_id=%s", run_id, workflow_id)
        return {"run_id": run_id, "workflow_id": workflow_id}
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to start evaluate workflow for run %s", run_id)
        raise HTTPException(status_code=500, detail="Failed to start evaluation workflow")


@router.post("/{run_id}/export", status_code=202)
async def export_run(
    run_id: str,
    body: ExportRequest = ExportRequest(),
    run_store: RunStorePort = Depends(get_run_store),
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> dict[str, str]:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.owner_id is not None and run.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    model = store.get(run.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found for this run")

    remote_backend = body.remote_backend or model.remote_backend
    if remote_backend == "local":
        remote_backend = ""

    workflow_id = f"export-{run_id}-{uuid.uuid4().hex[:8]}"
    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import ExportWorkflow, ExportWorkflowConfig

        client = await Client.connect(os.getenv("TEMPORAL_HOST", "localhost:7233"))
        run_store.update(run_id, RunConfig(model_id=run.model_id, workflow_id=workflow_id))
        run_store.update_status(run_id, RunStatus.RUNNING)

        await client.start_workflow(
            ExportWorkflow.run,
            ExportWorkflowConfig(
                run_id=run_id,
                model_id=model.id,
                remote_backend=remote_backend,
                remote_run_id=body.remote_run_id,
                checkpoint_path=f"data/workflow/{run_id}/checkpoint",
                gguf_output=f"data/workflow/{run_id}/model.gguf",
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        log.info("Export workflow started: run_id=%s workflow_id=%s", run_id, workflow_id)
        return {"run_id": run_id, "workflow_id": workflow_id}
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to start export workflow for run %s", run_id)
        raise HTTPException(status_code=500, detail="Failed to start export workflow")
