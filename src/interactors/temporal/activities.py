"""Temporal activities — one per pipeline stage, each wrapping a domain function."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from temporalio import activity

log = logging.getLogger(__name__)
from temporalio.exceptions import ApplicationError

from sqlalchemy.engine import Engine as _Engine
from domain.ports import RemoteJobPort, RemoteTrainingPort, StoragePort, SubmitRetryConfig
from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
from domain.train.config import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO


# ---------------------------------------------------------------------------
# Module-level singletons — injected by the worker (or tests)
# ---------------------------------------------------------------------------

_engine: _Engine | None = None
_storage: StoragePort | None = None

# Max length of the progress `detail` string allowed into a Temporal heartbeat.
# Heartbeat details must stay well under the server's ~2 MB blob limit; a short
# status line is all we ever need here.
_HEARTBEAT_DETAIL_MAX = 2000


def configure_engine(engine: _Engine) -> None:
    global _engine
    _engine = engine


def configure_storage(storage: StoragePort) -> None:
    global _storage
    _storage = storage


def _create_uow():
    """Return a fresh UoW — create one per activity call, never share."""
    from adapters.database.uow import SQLAlchemyUnitOfWork
    if _engine is None:
        raise RuntimeError("Engine has not been configured in activities.")
    return SQLAlchemyUnitOfWork(_engine)


def _get_storage() -> StoragePort:
    if _storage is None:
        raise RuntimeError("StoragePort has not been configured in activities.")
    return _storage


def _build_log_prefix(run_id: str, section_name: str) -> str:
    """Read existing S3 log content once and return a cached prefix with a section separator.

    Called once before each activity's polling loop. Every subsequent write in the loop
    becomes write_bytes(key, (prefix + latest_cumulative_logs).encode()), so the file
    grows monotonically and the streaming endpoint's byte-range offset tracking works.
    """
    log_key = f"workflow/{run_id}/logs.txt"
    try:
        existing = _get_storage().read_text(log_key)
    except Exception:
        existing = ""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sep = f"\n{'=' * 80}\n=== {section_name} — Started {ts} ===\n{'=' * 80}\n\n"
    return existing + sep


# ---------------------------------------------------------------------------
# Config / result dataclasses (Temporal serialises these as JSON)
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED
    # Set to True when using a remote training backend (k8s, kaggle, ssh, …).
    # The activity will upload the generated files to the configured StoragePort so
    # the remote job can download them — without this the pod gets a 404 from S3.
    upload_to_storage: bool = False
    # DB run ID — used to derive the canonical S3 key: dataset/runs/{run_id}/train.jsonl
    run_id: str = ""


@dataclass
class DatasetPaths:
    train: str = ""
    eval: str = ""


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    train_data: str = "data/train.jsonl"
    output_dir: str = DEFAULT_OUTPUT_DIR
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    dry_run: bool = False
    # Remote backend: "" or "local" → run locally; "kaggle" or "ssh" → remote.
    remote_backend: str = ""
    experiment_name: str = ""
    run_id: str = ""  # DB RunRecord.id — also used as the S3 key prefix for this run
    # None = auto-detect based on model size; True = always QLoRA; False = never QLoRA.
    force_qlora: bool | None = None


@dataclass
class CheckpointPath:
    path: str = ""            # local path; empty when checkpoint is still on the remote
    run_id: str = ""          # opaque id from adapter.submit(); non-empty for remote runs
    remote_backend: str = ""  # "kaggle", "ssh", etc.; "" means local


@dataclass
class EvalConfig:
    checkpoint: str = ""
    eval_data: str = "data/eval.jsonl"
    artifact_run_id: str = ""  # training adapter run_id (S3 prefix / kernel slug)
    remote_backend: str = ""  # non-empty → dispatch eval as EvalJobSpec to the remote GPU
    output_dir: str = ""      # local dir for downloaded eval_results.json
    run_id: str = ""       # DB RunRecord.id for progress updates; "" = no tracking


@dataclass
class EvalResult:
    valid_pct: float = 0.0
    passed: bool = False


@dataclass
class ExportConfig:
    checkpoint_path: str = ""
    gguf_output: str = "models/model.gguf"
    run_id: str = ""           # non-empty → download checkpoint from remote before export
    remote_backend: str = ""
    model_id: str = ""         # fallback storage key when pipeline_run_id is unset
    pipeline_run_id: str = ""  # pipeline UUID; drives storage key model/runs/{id}/{name}.gguf
    model_name: str = ""       # human-readable name; drives model/{model_id}/{name}.gguf key


@dataclass
class GGUFPath:
    path: str = ""            # storage key (e.g. "model/{model_id}/model.gguf")


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


async def _heartbeat_loop(stage: str, interval: int = 30) -> None:
    """Send a liveness heartbeat every `interval` seconds while a blocking call runs."""
    while True:
        activity.heartbeat({"stage": stage})
        await asyncio.sleep(interval)


def _is_retryable_submit_error(exc: Exception, config: SubmitRetryConfig) -> bool:
    msg = str(exc).lower()
    return any(pattern in msg for pattern in config.retryable_errors)


async def _retryable_submit(
    adapter: RemoteJobPort,
    remote_config: object,
    loop: asyncio.AbstractEventLoop,
) -> str:
    """Submit with exponential-backoff retries for transient errors.

    Retry constraints come from adapter.submit_retry_config() — each backend
    declares its own retryable strings, max attempts, and base delay.
    The heartbeat task stays alive across all backoff sleeps.
    """
    config = adapter.submit_retry_config()
    errors: list[str] = []
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("train_submit"))
    try:
        for attempt in range(1, config.max_retries + 2):  # 1..max+1 inclusive
            try:
                return await loop.run_in_executor(None, lambda: adapter.submit(remote_config))
            except Exception as exc:
                if not _is_retryable_submit_error(exc, config):
                    raise ApplicationError(
                        f"Remote submit failed (non-retryable): {exc}",
                        non_retryable=True,
                    ) from exc

                errors.append(f"attempt {attempt}: {exc}")
                log.warning(
                    "Remote submit attempt %d/%d failed (retryable): %s",
                    attempt, config.max_retries + 1, exc,
                )

                if attempt > config.max_retries:
                    raise ApplicationError(
                        f"Remote submit failed after {attempt} attempt(s): {'; '.join(errors)}",
                        non_retryable=True,
                    ) from exc

                delay = config.base_delay_s * (2 ** (attempt - 1))
                log.info("Backing off %.1fs before submit retry %d", delay, attempt + 1)
                await asyncio.sleep(delay)
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

    raise RuntimeError("unreachable")


async def _poll_local_progress(run_id: str, output_dir: str, interval: int = 30) -> None:
    """Heartbeat loop for local training: also polls progress.json and persists it."""
    import json as _json
    progress_path = Path(output_dir) / "progress.json"
    while True:
        activity.heartbeat({"stage": "train_local"})
        if run_id:
            try:
                data = _json.loads(progress_path.read_text())
                step = data.get("step", 0)
                max_steps = data.get("max_steps", 1)
                frac = step / max_steps if max_steps else 0.0
                epoch = data.get("epoch", "?")
                parts = [f"epoch={epoch}"]
                for key in ("loss", "eval_loss"):
                    if key in data:
                        parts.append(f"{key}={data[key]:.4f}")
                with _create_uow().transaction() as _uow:
                    _uow.run_store.update_progress(run_id, frac, "  ".join(parts))
            except Exception:
                pass
        await asyncio.sleep(interval)


@activity.defn
async def generate_dataset_activity(config: DatasetConfig) -> DatasetPaths:
    from domain.train.dataset import generate

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("generate_dataset"))
    try:
        ok = await loop.run_in_executor(
            None,
            lambda: generate(
                data_dir=Path(config.data_dir),
                train_size=config.train_size,
                eval_size=config.eval_size,
                seed=config.seed,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"generate_dataset failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    if not ok:
        raise ApplicationError("Dataset generation failed: invalid examples or distribution out of bounds")

    train_path = str(Path(config.data_dir) / "train.jsonl")
    eval_path = str(Path(config.data_dir) / "eval.jsonl")

    if config.upload_to_storage:
        # Remote backends (K8s, Kaggle, SSH, …) fetch training data from S3.
        # Upload the generated files now so the remote job can call storage.download().
        # Store under dataset/runs/{run_id}/ — never workflow/, which a lifecycle rule
        # expires — so generated data survives for audits and retraining.
        from adapters.storage.paths import run_dataset_eval_key, run_dataset_train_key

        storage = _get_storage()
        if config.run_id:
            train_s3_key = run_dataset_train_key(config.run_id)
            eval_s3_key = run_dataset_eval_key(config.run_id)
        else:
            train_s3_key = train_path
            eval_s3_key = eval_path
        await loop.run_in_executor(
            None,
            lambda: (
                storage.upload(Path(train_path), train_s3_key),
                storage.upload(Path(eval_path), eval_s3_key),
            ),
        )
        log.info("Dataset uploaded to storage: train=%s eval=%s", train_s3_key, eval_s3_key)
        return DatasetPaths(train=train_s3_key, eval=eval_s3_key)

    return DatasetPaths(train=train_path, eval=eval_path)


def _make_remote_adapter(backend: str) -> RemoteJobPort:
    # Storage is only needed for S3-backed backends; fetch lazily to avoid
    # failing when storage is not configured (e.g. Kaggle tests).
    if backend == "k8s":
        from adapters.compute.k8s.adapter import K8sTrainingAdapter
        return K8sTrainingAdapter(storage=_get_storage())
    if backend == "kaggle":
        from adapters.compute.kaggle import KaggleTrainingAdapter
        return KaggleTrainingAdapter()          # Kaggle uses its own file-based staging
    if backend == "ssh":
        from adapters.compute.ssh import SshTrainingAdapter
        return SshTrainingAdapter()             # SSH has no StoragePort injection yet
    if backend == "colab":
        from adapters.compute.colab.adapter import ColabTrainingAdapter
        return ColabTrainingAdapter()           # Colab has no StoragePort injection yet
    if backend == "runpod":
        from adapters.compute.runpod import RunPodTrainingAdapter
        return RunPodTrainingAdapter(storage=_get_storage())
    if backend == "vastai":
        from adapters.compute.vastai.adapter import VastAiTrainingAdapter
        return VastAiTrainingAdapter(storage=_get_storage())
    raise ApplicationError(f"Unknown remote_backend: {backend!r}")


@activity.defn
async def train_activity(config: TrainConfig) -> CheckpointPath:
    backend = config.remote_backend or "local"

    if backend == "local":
        return await _train_local(config)

    adapter = _make_remote_adapter(backend)
    return await _train_remote(config, adapter)


async def _train_local(config: TrainConfig) -> CheckpointPath:
    from domain.train.trainer import train

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(
        _poll_local_progress(config.run_id, config.output_dir)
    )
    try:
        await loop.run_in_executor(
            None,
            lambda: train(
                model=config.model,
                train_data=config.train_data,
                output_dir=config.output_dir,
                epochs=config.epochs,
                patience=config.patience,
                warmup_ratio=config.warmup_ratio,
                dry_run=config.dry_run,
                force_qlora=config.force_qlora,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"train failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return CheckpointPath(path=config.output_dir)


# Backends that stage training data from the local filesystem before uploading it
# to their own compute environment.  These need train/eval files materialised
# locally even when skip_generate=True provides an S3 key as train_data.
#
# S3-backed backends (k8s, runpod) pass the S3 key directly to the
# remote job; the pod downloads from S3 itself.  Replacing the key with a local
# path would point the pod at a key that doesn't exist → DO NOT download for them.
# Note: Kaggle is intentionally excluded here — it accesses training data from S3
# directly (enable_internet=True) rather than requiring local staging.
_FILE_BASED_BACKENDS = frozenset({"colab", "ssh"})


async def _resolve_training_data(
    config: TrainConfig, loop: asyncio.AbstractEventLoop
) -> str:
    """Return the train key ready for use in ``TrainJobSpec``.

    For **file-based backends** (Colab, SSH) the adapter stages data from the
    local filesystem.  When ``skip_generate=True`` the workflow forwards an S3
    storage key as ``train_data`` — the file does not exist locally.  This
    function downloads it to ``data/train.jsonl`` and returns the local path.

    For **S3-backed backends** (K8s, RunPod, VastAI) the remote job downloads
    from S3 directly; the original S3 key is returned unchanged.
    """
    train_key = config.train_data

    # S3-backed backends: pass key straight through — no local download needed.
    if config.remote_backend not in _FILE_BASED_BACKENDS:
        return train_key

    def _is_local(key: str) -> bool:
        p = Path(key)
        return (p if p.is_absolute() else Path.cwd() / p).exists()

    if _is_local(train_key):
        return train_key

    activity.logger.info(
        "Training data not found locally (train=%r cwd=%s); "
        "downloading from storage for local staging.",
        train_key, Path.cwd(),
    )
    storage = _get_storage()
    dest = Path("data")
    dest.mkdir(parents=True, exist_ok=True)
    local_train = dest / "train.jsonl"

    if not _is_local(train_key):
        await loop.run_in_executor(None, lambda: storage.download(train_key, local_train))
        activity.logger.info("Downloaded train data → %s", local_train)

    return str(local_train)


async def _train_remote(config: TrainConfig, adapter: RemoteJobPort) -> CheckpointPath:
    from domain.models import TrainJobSpec

    loop = asyncio.get_event_loop()

    # If train path is an S3 key, download locally for file-based backends
    # (Colab, SSH) that stage data into their compute environment.
    train_data = await _resolve_training_data(config, loop)

    remote_config = TrainJobSpec(
        model=config.model,
        train_data=train_data,
        epochs=config.epochs,
        patience=config.patience,
        warmup_ratio=config.warmup_ratio,
        experiment_name=config.experiment_name or "llm-api",
        run_id=config.run_id,
        # Original S3 key for backends that access storage directly (e.g. Kaggle).
        train_s3_key=config.train_data,
    )

    # Resume from a prior attempt if the remote job was already submitted.
    info = activity.info()
    prior_run_id: str | None = None
    if info.heartbeat_details:
        prev = info.heartbeat_details[0]
        if isinstance(prev, dict) and prev.get("run_id"):
            prior_run_id = prev["run_id"]

    if prior_run_id:
        activity.logger.info(
            "Resuming poll for existing remote job: adapter=%s run_id=%s",
            type(adapter).__name__, prior_run_id,
        )
        run_id = prior_run_id
    else:
        run_id = await _retryable_submit(adapter, remote_config, loop)

        activity.logger.info("Remote job submitted: adapter=%s run_id=%s", type(adapter).__name__, run_id)

        # Give the remote backend time to register the job before the first status
        # poll.  Kaggle's GetKernelSessionStatus returns 500 if the session does not
        # exist yet (race condition between kernel push and session initialisation).
        await asyncio.sleep(10)

    started_at = time.time()

    # Read existing S3 log content once so subsequent writes append rather than overwrite.
    train_log_prefix = _build_log_prefix(config.run_id, "TRAINING") if config.run_id else ""
    last_logs: str = ""  # only re-upload to S3 when the cumulative log actually changes

    # Background heartbeat keeps the activity alive while executor calls block.
    # status() + logs() can take >2 min on slow VastAI/S3 paths; without this
    # the heartbeat_timeout fires before the inline heartbeat arrives.
    poll_heartbeat = asyncio.ensure_future(_heartbeat_loop("train_poll", interval=30))
    try:
        while True:
            # status/logs/progress all run subprocess — keep them off the event loop.
            try:
                status = await loop.run_in_executor(None, lambda: adapter.status(run_id))
            except Exception as exc:
                raise ApplicationError(f"Remote status check failed: {exc}") from exc

            elapsed_s = int(time.time() - started_at)
            logs = await loop.run_in_executor(None, lambda: adapter.logs(run_id))

            activity.logger.info(
                "Remote status: adapter=%s run_id=%s status=%s elapsed=%ds",
                type(adapter).__name__, run_id, status, elapsed_s,
            )
            log.info(
                "Remote status: adapter=%s run_id=%s status=%s elapsed=%ds",
                type(adapter).__name__, run_id, status, elapsed_s,
            )
            if logs:
                activity.logger.info("Instance output:\n%s", logs)
                log.info("Instance output (run_id=%s):\n%s", run_id, logs)

            if logs and logs != last_logs and config.run_id:
                # Re-upload only when the cumulative log changed — adapter.logs()
                # returns the full log every poll, so unchanged content would mean
                # re-uploading the same (growing) blob every 30s for nothing.
                # Run off the event loop so a slow S3 put can't block the
                # background heartbeat task and trip the heartbeat timeout.
                payload = (train_log_prefix + logs).encode()
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: _get_storage().write_bytes(
                            f"workflow/{config.run_id}/logs.txt", payload
                        ),
                    )
                    last_logs = logs
                except Exception:
                    log.warning("Failed to persist training logs for run %s", config.run_id)

            # Fetch progress (a small fraction + short detail string) before the heartbeat
            # so it can ride along.  Unlike logs, this stays bounded — safe for heartbeat
            # details.  Only fetched when a DB run_id exists (matches DB-persist guard below).
            frac, detail = 0.0, ""
            if config.run_id:
                try:
                    frac, detail = await loop.run_in_executor(None, lambda: adapter.progress(run_id))
                except Exception:
                    frac, detail = 0.0, ""

            # Detailed heartbeat after each poll (background loop covers gaps between polls).
            # run_id is included so a restarted worker can resume polling without resubmitting.
            # NOTE: never put `logs` here — cumulative training logs grow unbounded and blow
            # past Temporal's ~2 MB heartbeat-details limit on long runs, killing the activity
            # with a non-retryable "Heartbeat details exceed size limit." error. Logs are
            # already persisted to S3 above; the live-log UI streams them from there.
            # `detail` is truncated defensively so a runaway progress string can't reintroduce
            # the same size failure.
            activity.heartbeat({
                "status": status,
                "elapsed_s": elapsed_s,
                "run_id": run_id,
                "progress": frac,
                "detail": detail[:_HEARTBEAT_DETAIL_MAX],
            })

            if config.run_id and frac > 0:
                try:
                    with _create_uow().transaction() as _uow:
                        _uow.run_store.update_progress(config.run_id, frac, detail)
                except Exception:
                    pass

            if status == "done":
                # Defer download — eval may run on the remote, so we avoid pulling
                # gigabytes of checkpoint data until we know the model actually passes.
                return CheckpointPath(
                    run_id=run_id,
                    remote_backend=config.remote_backend,
                )

            if status == "failed":
                tail = f"\n--- pod logs ---\n{logs}" if logs else " (no logs captured)"
                raise ApplicationError(
                    f"Remote training failed (adapter={type(adapter).__name__}, run_id={run_id}){tail}"
                )

            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                activity.logger.warning(
                    "train_activity cancelled while polling (adapter=%s, run_id=%s, elapsed=%ds)",
                    type(adapter).__name__,
                    run_id,
                    int(time.time() - started_at),
                )
                raise
    finally:
        poll_heartbeat.cancel()
        await asyncio.gather(poll_heartbeat, return_exceptions=True)


@activity.defn
async def evaluate_activity(config: EvalConfig) -> EvalResult:
    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("evaluate"))
    try:
        if config.remote_backend:
            # Dispatch eval to the remote GPU that ran training.
            # The EvalJobSpec downloads checkpoint + eval data on the remote machine;
            # the Temporal worker only polls status and retrieves the tiny eval_results.json.
            result = await _evaluate_via_remote_job(config, loop)
        else:
            result = await _evaluate_local(config, loop)
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"evaluate failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    activity.logger.info(
        "Eval complete: valid_pct=%.1f%% passed=%s",
        result.valid_pct * 100,
        result.passed,
    )
    return result


async def _evaluate_via_remote_job(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    """Submit an EvalJobSpec to the remote backend and poll until done.

    Mirrors the _train_remote polling loop so heartbeats are sent throughout.
    """
    import json as _json
    from domain.models import EvalJobSpec

    adapter = _make_remote_adapter(config.remote_backend)
    spec = EvalJobSpec(
        experiment_name=f"eval-{config.run_id or 'standalone'}",
        training_artifact_ref=config.artifact_run_id,
        eval_data=config.eval_data,
        run_id=config.run_id,
    )
    eval_run_id = await loop.run_in_executor(None, lambda: adapter.submit(spec))
    activity.logger.info(
        "Remote eval submitted: backend=%s eval_run_id=%s", config.remote_backend, eval_run_id
    )

    # Read existing S3 log content once so eval logs append after training logs.
    eval_log_prefix = _build_log_prefix(config.run_id, "EVALUATION") if config.run_id else ""

    poll_hb = asyncio.ensure_future(_heartbeat_loop("eval_poll", interval=30))
    poll_start = loop.time()
    last_logs: str = ""
    try:
        while True:
            status = await loop.run_in_executor(None, lambda: adapter.status(eval_run_id))
            logs = await loop.run_in_executor(None, lambda: adapter.logs(eval_run_id))
            elapsed = loop.time() - poll_start
            activity.heartbeat({"status": status, "eval_run_id": eval_run_id, "elapsed_s": int(elapsed)})
            activity.logger.info(
                "Remote eval poll: backend=%s eval_run_id=%s status=%s elapsed=%ds",
                config.remote_backend, eval_run_id, status, int(elapsed),
            )
            if logs and logs != last_logs:
                activity.logger.info("Remote eval output (eval_run_id=%s):\n%s", eval_run_id, logs)
                if config.run_id:
                    try:
                        _get_storage().write_bytes(f"workflow/{config.run_id}/logs.txt", (eval_log_prefix + logs).encode())
                    except Exception:
                        log.warning("Failed to persist eval logs for run %s", config.run_id)
                last_logs = logs
            if status == "done":
                # Final fetch: eval_script.py flushes logs.txt before writing status.txt="done",
                # but re-fetch to capture any content that arrived in the same poll iteration.
                try:
                    final_logs = await loop.run_in_executor(None, lambda: adapter.logs(eval_run_id))
                    if final_logs and config.run_id:
                        _get_storage().write_bytes(f"workflow/{config.run_id}/logs.txt", (eval_log_prefix + final_logs).encode())
                except Exception:
                    log.warning("Failed to persist final eval logs for run %s", config.run_id)
                break
            if status == "failed":
                raise ApplicationError(
                    f"Remote eval failed (backend={config.remote_backend}, "
                    f"eval_run_id={eval_run_id}, elapsed={int(elapsed)}s)\n{logs}"
                )
            await asyncio.sleep(30)
    finally:
        poll_hb.cancel()
        await asyncio.gather(poll_hb, return_exceptions=True)

    dest = Path(config.output_dir or f"models/eval_tmp/{eval_run_id.replace('/', '_')}")
    try:
        result_path = await loop.run_in_executor(None, lambda: adapter.download(eval_run_id, dest))
        data = _json.loads(Path(result_path).read_text())
        return EvalResult(valid_pct=data["valid_pct"], passed=data["passed"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ApplicationError(
            f"Remote eval result malformed "
            f"(backend={config.remote_backend}, eval_run_id={eval_run_id}): {exc}"
        ) from exc
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(
            f"Remote eval result download failed "
            f"(backend={config.remote_backend}, eval_run_id={eval_run_id}): {exc}"
        ) from exc


def _normalise_report_keys(obj: object) -> object:
    """Recursively rename the JSON key 'pass' → 'passed' (pass is a Python keyword)."""
    if isinstance(obj, dict):
        return {("passed" if k == "pass" else k): _normalise_report_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_report_keys(v) for v in obj]
    return obj


async def _evaluate_local(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline

    pipe = load_hf_pipeline(config.checkpoint)
    infer_fn_raw = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

    exit_code, valid_pct = await loop.run_in_executor(
        None, lambda: evaluate(Path(config.eval_data), infer_fn_raw)
    )

    if config.run_id:
        from adapters.prompt import build_prompt, parse_response
        from domain.train.quality_report import run_quality_report

        infer_fn_rich = lambda req: parse_response(infer_fn_raw(build_prompt(req)))  # noqa: E731
        report = await loop.run_in_executor(
            None, lambda: run_quality_report(infer_fn_rich)
        )
        report = _normalise_report_keys(report)
        report_path = Path(f"data/workflow/{config.run_id}/quality_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report))
        activity.logger.info("Quality report saved: %s", report_path)

    return EvalResult(valid_pct=valid_pct, passed=exit_code == 0)


@activity.defn
async def export_activity(config: ExportConfig) -> GGUFPath:
    # Determine the destination GGUF storage key upfront — shared by all paths.
    from adapters.storage.paths import standalone_model_key, run_model_key

    if config.pipeline_run_id:
        name = config.model_name or "model"
        gguf_key = run_model_key(config.pipeline_run_id, name)
    elif config.model_name or config.model_id:
        key_id = config.model_id or config.model_name
        name = config.model_name or "model"
        gguf_key = standalone_model_key(key_id, name)
    else:
        gguf_key = config.gguf_output

    # ── K8s path: dispatch a dedicated export Job ─────────────────────────
    # The Temporal worker image does NOT have llama.cpp installed.  When the
    # backend is "k8s" we submit an ExportJobSpec so the conversion runs inside
    # the export Docker image (which has llama.cpp pre-built).
    # The Job downloads the checkpoint from S3, converts it, and uploads the
    # GGUF back to S3 — the worker only needs to submit + poll.
    if config.remote_backend == "k8s":
        from domain.models import ExportJobSpec
        run_id = config.pipeline_run_id
        if not run_id:
            raise ApplicationError(
                "export_activity: pipeline_run_id is required for K8s export "
                "(it is the DB run_id used as the S3 key prefix)."
            )
        spec = ExportJobSpec(
            experiment_name=run_id,
            checkpoint_s3_prefix=f"workflow/{run_id}/checkpoint/",
            gguf_s3_key=gguf_key,
        )
        adapter = _make_remote_adapter("k8s")
        loop = asyncio.get_event_loop()
        heartbeat_task = asyncio.ensure_future(_heartbeat_loop("export"))
        try:
            export_run_id = await loop.run_in_executor(None, lambda: adapter.submit(spec))
            activity.logger.info(
                "K8s export Job submitted: %s (run_id=%s, gguf_key=%s)",
                export_run_id,
                run_id,
                gguf_key,
            )
            while True:
                status = await loop.run_in_executor(
                    None, lambda: adapter.status(export_run_id)
                )
                if status == "done":
                    break
                if status == "failed":
                    logs = adapter.logs(export_run_id)
                    raise ApplicationError(
                        f"K8s export Job {export_run_id} failed.\nLogs:\n{logs}"
                    )
                await asyncio.sleep(30)
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(f"K8s export failed: {exc}") from exc
        finally:
            heartbeat_task.cancel()
        return GGUFPath(path=gguf_key)

    # ── Local / other-remote path: download checkpoint then export inline ──
    # Non-k8s backends (kaggle, ssh, colab, runpod) and local runs
    # still download the checkpoint to the worker and export inline.
    from domain.train.export import export as export_gguf

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("export"))
    try:
        if config.remote_backend:
            adapter = _make_remote_adapter(config.remote_backend)
            dest = Path(config.gguf_output).parent / "checkpoint"
            try:
                checkpoint_path = await loop.run_in_executor(
                    None, lambda: adapter.download(config.run_id, dest)
                )
            except Exception as exc:
                raise ApplicationError(f"Remote download failed: {exc}") from exc
        else:
            checkpoint_path = config.checkpoint_path

        local_gguf = Path(config.gguf_output)
        await loop.run_in_executor(
            None,
            lambda: export_gguf(checkpoint=Path(checkpoint_path), output=local_gguf),
        )

        # Upload to storage so the API can retrieve it by key.
        from adapters.storage import upload_model
        storage = _get_storage()
        gguf_key = upload_model(storage, local_gguf, gguf_key)

    except SystemExit as exc:
        raise ApplicationError(f"export failed: llama.cpp setup issue (exit {exc.code})") from exc
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"export failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return GGUFPath(path=gguf_key)


@activity.defn
async def finalise_run_activity(run_id: str, passed: bool, valid_pct: float) -> None:
    """Mark the run as completed or failed.

    ``passed`` reflects whether *training* succeeded (not eval) — the workflow
    always calls this with ``passed=True`` after a successful train.  Eval outcome
    is recorded separately via ``record_eval_result_activity``.
    The ``valid_pct`` parameter is kept for backward-compat but is no longer
    persisted here (``record_eval_result_activity`` owns that).
    """
    from domain.models import RunStatus

    with _create_uow().transaction() as uow:
        uow.run_store.update_status(run_id, RunStatus.COMPLETED if passed else RunStatus.FAILED)
    activity.logger.info(
        "Run %s finalised: status=%s",
        run_id,
        RunStatus.COMPLETED.value if passed else RunStatus.FAILED.value,
    )


@activity.defn
async def record_eval_result_activity(
    run_id: str, valid_pct: float, outcome_value: str
) -> None:
    """Persist the eval score and SUCCEEDED/FAILED outcome atomically.

    Called by the workflow after evaluate_activity finishes (or raises).
    Keeping this as a separate activity means eval failure never blocks
    the run from being marked COMPLETED.
    """
    from domain.models import EvalOutcome

    with _create_uow().transaction() as uow:
        uow.run_store.update_eval_result(run_id, valid_pct, EvalOutcome(outcome_value))
    activity.logger.info(
        "Eval result recorded: run=%s pct=%.1f%% outcome=%s",
        run_id, valid_pct * 100, outcome_value,
    )


@activity.defn
async def update_run_status_activity(run_id: str, status_value: str) -> None:
    """Set run status without touching eval_valid_pct (used by export-only workflows)."""
    from domain.models import RunStatus
    with _create_uow().transaction() as uow:
        uow.run_store.update_status(run_id, RunStatus(status_value))


@activity.defn
async def fail_run_activity(run_id: str, reason: str, status_value: str = "failed") -> None:
    """Mark a run as failed or cancelled, persisting the reason in progress_detail.

    Called by the workflow's exception handlers so that any activity failure or
    Temporal cancellation is reflected in the run record.
    """
    from domain.models import RunStatus
    with _create_uow().transaction() as uow:
        uow.run_store.fail_run(run_id, reason, RunStatus(status_value))
    activity.logger.info(
        "Run %s marked %s: %s",
        run_id,
        status_value,
        reason,
    )


@activity.defn
async def save_gguf_path_activity(model_id: str, gguf_path: str) -> None:
    """Persist the storage key of the exported GGUF back to the model record."""
    from domain.models import TrainingModelConfig
    with _create_uow().transaction() as uow:
        model = uow.model_store.get(model_id)
        if model is None:
            activity.logger.warning("save_gguf_path: model %s not found — skipping", model_id)
            return
        config = TrainingModelConfig(
            name=model.name,
            description=model.description,
            base_model=model.base_model,
            train_data=model.train_data,
            eval_data=model.eval_data,
            epochs=model.epochs,
            patience=model.patience,
            warmup_ratio=model.warmup_ratio,
            remote_backend=model.remote_backend,
            skip_generate=model.skip_generate,
            gguf_path=gguf_path,
            is_active=model.is_active,
        )
        uow.model_store.update(model_id, config)
    activity.logger.info("Saved gguf_path=%s for model %s", gguf_path, model_id)


@activity.defn
async def create_inference_activity(model_id: str, model_path: str = "", run_id: str = "") -> str:
    """Create an InferenceInstance record for an exported model.
    Returns the new instance id."""
    from domain.models import InferenceInstanceConfig
    config = InferenceInstanceConfig(model_id=model_id, model_path=model_path, run_id=run_id or None)
    with _create_uow().transaction() as uow:
        instance = uow.inference_store.create(config)
    return instance.id
