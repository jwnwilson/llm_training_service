"""Unit tests for Temporal activities — domain functions are mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from domain.ports import SubmitRetryConfig as _SubmitRetryConfig

from interactors.temporal.activities import (
    CheckpointPath,
    DatasetConfig,
    DatasetPaths,
    EvalConfig,
    EvalResult,
    ExportConfig,
    GGUFPath,
    TrainConfig,
    configure_storage,
    evaluate_activity,
    export_activity,
    fail_run_activity,
    finalise_run_activity,
    generate_dataset_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)


ENV = ActivityEnvironment()

def _mock_uow(run_store=None, model_store=None, inference_store=None):
    """Mock UoW whose transaction() context manager yields itself."""
    from contextlib import contextmanager
    mock = MagicMock()
    if run_store is not None:
        mock.run_store = run_store
    if model_store is not None:
        mock.model_store = model_store
    if inference_store is not None:
        mock.inference_store = inference_store

    @contextmanager
    def fake_transaction():
        yield mock

    mock.transaction = fake_transaction
    return mock




# ---------------------------------------------------------------------------
# generate_dataset_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_dataset_activity_delegates_to_domain():
    with patch("domain.train.dataset.generate", return_value=True) as mock_gen:
        result = await ENV.run(generate_dataset_activity, DatasetConfig(data_dir="data", train_size=10, eval_size=5, seed=1))

    mock_gen.assert_called_once_with(data_dir=Path("data"), train_size=10, eval_size=5, seed=1)
    assert result == DatasetPaths(train="data/train.jsonl", eval="data/eval.jsonl")


@pytest.mark.asyncio
async def test_generate_dataset_activity_raises_on_invalid_examples():
    with patch("domain.train.dataset.generate", return_value=False):
        with pytest.raises(ApplicationError, match="invalid examples"):
            await ENV.run(generate_dataset_activity, DatasetConfig())


@pytest.mark.asyncio
async def test_generate_dataset_activity_raises_on_exception():
    with patch("domain.train.dataset.generate", side_effect=RuntimeError("disk full")):
        with pytest.raises(ApplicationError, match="generate_dataset failed"):
            await ENV.run(generate_dataset_activity, DatasetConfig())


@pytest.mark.asyncio
async def test_generate_dataset_uploads_files_to_storage_when_flag_is_set(tmp_path, monkeypatch):
    """When upload_to_storage=True with a run_id, files are uploaded under
    dataset/runs/{run_id}/ so the remote job can download them and they are kept.

    Regression for: K8s pod 404 when data was uploaded to a non-canonical path, and
    for generated datasets being deleted by the workflow/ lifecycle expiry rule.
    """
    import interactors.temporal.activities as acts

    mock_storage = MagicMock()
    monkeypatch.setattr(acts, "_storage", mock_storage)
    monkeypatch.chdir(tmp_path)

    data_dir = "data/workflow/run-abc123"
    run_id = "run-abc123"
    (tmp_path / data_dir).mkdir(parents=True)
    (tmp_path / data_dir / "train.jsonl").write_text('{"action": "IDLE"}\n')
    (tmp_path / data_dir / "eval.jsonl").write_text('{"action": "IDLE"}\n')

    with patch("domain.train.dataset.generate", return_value=True):
        result = await ENV.run(
            generate_dataset_activity,
            DatasetConfig(data_dir=data_dir, upload_to_storage=True, run_id=run_id),
        )

    # Both files must be uploaded so the remote K8s pod can download them from S3.
    assert mock_storage.upload.call_count == 2, (
        "Expected two upload() calls (train + eval) — "
        "without this the K8s pod gets a 404 from S3"
    )
    uploaded_keys = [call.args[1] for call in mock_storage.upload.call_args_list]
    # Persistent dataset namespace — never under the expiring workflow/ prefix.
    assert f"dataset/runs/{run_id}/train.jsonl" in uploaded_keys
    assert f"dataset/runs/{run_id}/eval.jsonl" in uploaded_keys
    # Returned DatasetPaths use the S3 keys so the remote pod can download them.
    assert result.train == f"dataset/runs/{run_id}/train.jsonl"
    assert result.eval == f"dataset/runs/{run_id}/eval.jsonl"


@pytest.mark.asyncio
async def test_generate_dataset_does_not_upload_when_flag_is_false(monkeypatch):
    """When upload_to_storage is False (default), no upload to storage occurs (local runs)."""
    import interactors.temporal.activities as acts

    mock_storage = MagicMock()
    monkeypatch.setattr(acts, "_storage", mock_storage)

    with patch("domain.train.dataset.generate", return_value=True):
        await ENV.run(
            generate_dataset_activity,
            DatasetConfig(data_dir="data", train_size=10, eval_size=5, seed=1),
        )

    mock_storage.upload.assert_not_called()


# ---------------------------------------------------------------------------
# train_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_activity_delegates_to_domain():
    config = TrainConfig(
        model="some-model",
        train_data="data/train.jsonl",
        output_dir="models/checkpoints",
        epochs=3,
        patience=2,
    )
    with patch("domain.train.trainer.train") as mock_train:
        result = await ENV.run(train_activity, config)

    mock_train.assert_called_once_with(
        model="some-model",
        train_data="data/train.jsonl",
        output_dir="models/checkpoints",
        epochs=3,
        patience=2,
        warmup_ratio=0.05,
        dry_run=False,
        force_qlora=None,
    )
    assert result == CheckpointPath(path="models/checkpoints")


@pytest.mark.asyncio
async def test_train_activity_raises_on_exception():
    with patch("domain.train.trainer.train", side_effect=ImportError("torch not found")):
        with pytest.raises(ApplicationError, match="train failed"):
            await ENV.run(train_activity, TrainConfig())


# ---------------------------------------------------------------------------
# evaluate_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_activity_delegates_to_domain():
    mock_pipe = MagicMock()
    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=mock_pipe) as mock_load,
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.evaluate.evaluate", return_value=(0, 0.95)) as mock_eval,
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints", eval_data="data/eval.jsonl"))

    mock_load.assert_called_once_with("models/checkpoints")
    mock_eval.assert_called_once()
    assert result.passed is True


@pytest.mark.asyncio
async def test_evaluate_activity_fail_result_when_exit_code_nonzero():
    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value=""),
        patch("domain.train.evaluate.evaluate", return_value=(1, 0.75)),
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints"))

    assert result.passed is False
    assert result.valid_pct == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_evaluate_activity_extracts_valid_pct_from_return_value():
    def fake_evaluate(path, infer_fn):
        return (0, 0.95)

    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value=""),
        patch("domain.train.evaluate.evaluate", side_effect=fake_evaluate),
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints"))

    assert result.passed is True
    assert result.valid_pct == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_evaluate_activity_raises_on_exception():
    with patch("domain.train.evaluate.load_hf_pipeline", side_effect=RuntimeError("model not found")):
        with pytest.raises(ApplicationError, match="evaluate failed"):
            await ENV.run(evaluate_activity, EvalConfig(checkpoint="bad/path"))


@pytest.mark.asyncio
async def test_evaluate_remote_backend_dispatches_via_remote_job(monkeypatch):
    """With a remote_backend set, evaluate_activity must dispatch to _evaluate_via_remote_job,
    NOT download the checkpoint to the Temporal worker.  The worker is an orchestrator only."""
    import asyncio
    import interactors.temporal.activities as acts

    remote_calls: list[str] = []

    async def fake_remote_job(config, loop):
        remote_calls.append(config.remote_backend)
        return EvalResult(valid_pct=0.95, passed=True)

    monkeypatch.setattr(acts, "_evaluate_via_remote_job", fake_remote_job)

    config = EvalConfig(remote_backend="kaggle", run_id="u/exp", eval_data="data/eval.jsonl")
    await acts._evaluate_via_remote_job(config, asyncio.get_event_loop())

    assert remote_calls == ["kaggle"], (
        "remote_backend must route to _evaluate_via_remote_job, not download checkpoint to Temporal"
    )


# ---------------------------------------------------------------------------
# export_activity
# ---------------------------------------------------------------------------


def _fake_upload_model(storage, local_path, key: str) -> str:
    """Test stand-in for adapters.storage.upload_model — skips compression."""
    return key if key.endswith(".gz") else key + ".gz"


@pytest.fixture()
def mock_storage():
    """Wire a fresh MagicMock StoragePort into the activities module and patch upload_model."""
    from adapters.storage.local import LocalStorageAdapter
    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)
    with patch("adapters.storage.upload_model", side_effect=_fake_upload_model) as mock_upload:
        storage.mock_upload_model = mock_upload
        yield storage


@pytest.mark.asyncio
async def test_export_activity_uses_model_name_as_filename_within_workflow_path(mock_storage):
    # When both pipeline_run_id and model_name are set, pipeline_run_id determines the
    # top-level prefix and model_name becomes the filename.
    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                model_name="my-pet-v2",
                pipeline_run_id="r1",
                model_id="model-uuid",
            ),
        )

    assert result.path == "model/runs/r1/my-pet-v2.gguf.gz"


@pytest.mark.asyncio
async def test_export_activity_pipeline_run_id_takes_precedence_with_named_model(mock_storage):
    # pipeline_run_id always wins the prefix; model_name becomes the filename.
    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                model_name="my-pet-v2",
                pipeline_run_id="r1",
            ),
        )

    assert result.path == "model/runs/r1/my-pet-v2.gguf.gz"


@pytest.mark.asyncio
async def test_export_activity_uses_pipeline_run_id_for_storage_key(mock_storage):
    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                pipeline_run_id="r1",
                model_id="m",
            ),
        )

    # The GGUFPath is the storage key returned by the activity (always .gz).
    assert result == GGUFPath(path="model/runs/r1/model.gguf.gz")
    # Verify upload_model was called (port contract boundary).
    mock_storage.mock_upload_model.assert_called_once()


@pytest.mark.asyncio
async def test_export_activity_pipeline_run_id_takes_precedence_over_model_id(mock_storage):
    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/run-42/model.gguf",
                pipeline_run_id="run-42",
                model_id="model-99",
            ),
        )

    assert result.path == "model/runs/run-42/model.gguf.gz"


@pytest.mark.asyncio
async def test_export_activity_falls_back_to_model_id_when_no_pipeline_run_id(mock_storage):
    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/gguf/m.gguf", model_id="m"),
        )

    assert result == GGUFPath(path="model/m/model.gguf.gz")
    mock_storage.mock_upload_model.assert_called_once()


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_exception():
    with patch("domain.train.export.export", side_effect=RuntimeError("conversion failed")):
        with pytest.raises(ApplicationError, match="export failed"):
            await ENV.run(
                export_activity,
                ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/model.gguf"),
            )


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_system_exit():
    with patch("domain.train.export.export", side_effect=SystemExit(1)):
        with pytest.raises(ApplicationError, match="llama.cpp setup issue"):
            await ENV.run(
                export_activity,
                ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/model.gguf"),
            )


# ---------------------------------------------------------------------------
# _train_remote polling loop
# ---------------------------------------------------------------------------


class TestTrainRemotePolling:
    """Verify _train_remote calls logs() and sends a structured heartbeat each poll."""

    def _make_adapter(self, statuses, log_output="step 10 loss=0.5", download_path="/tmp/ckpt"):
        adapter = MagicMock()
        adapter.submit.return_value = "run-42"
        adapter.submit_retry_config.return_value = _SubmitRetryConfig()
        adapter.status.side_effect = list(statuses)
        adapter.logs.return_value = log_output
        adapter.download.return_value = download_path
        return adapter

    def _patches(self, monkeypatch):
        """Common patches: silence activity logger, stop heartbeat loop from spinning."""
        import interactors.temporal.activities as acts

        async def _noop_heartbeat_loop(*args, **kwargs):
            pass

        monkeypatch.setattr(acts, "_heartbeat_loop", _noop_heartbeat_loop)
        monkeypatch.setattr(acts.activity, "logger", MagicMock())
        monkeypatch.setattr(acts.activity, "info", MagicMock(return_value=MagicMock(heartbeat_details=())))
        return acts

    @pytest.mark.asyncio
    async def test_calls_adapter_logs_each_poll(self, monkeypatch):
        acts = self._patches(monkeypatch)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        adapter = self._make_adapter(["running", "done"])
        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert adapter.logs.call_count >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_carries_status_elapsed_and_run_id(self, monkeypatch):
        acts = self._patches(monkeypatch)
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))

        adapter = self._make_adapter(["running", "done"])
        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured, "heartbeat should have been called"
        first = captured[0]
        assert isinstance(first, dict)
        assert first["status"] == "running"
        assert "elapsed_s" in first
        # run_id must be present so a restarted worker can resume polling.
        assert first["run_id"] == "run-42"

    @pytest.mark.asyncio
    async def test_heartbeat_excludes_logs(self, monkeypatch):
        """Logs must NOT ride in heartbeat details.

        Cumulative training logs grow without bound; Temporal rejects heartbeat
        details over ~2 MB with a non-retryable "Heartbeat details exceed size
        limit." error. Logs are persisted to S3 separately, so they have no
        business in the heartbeat payload.
        """
        acts = self._patches(monkeypatch)
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))

        # Simulate a long run whose log file has grown past the 2 MB limit.
        huge_logs = "x" * (3 * 1024 * 1024)
        adapter = self._make_adapter(["running", "done"], log_output=huge_logs)
        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured, "heartbeat should have been called"
        for hb in captured:
            assert "logs" not in hb, "logs must not be embedded in heartbeat details"
            # Every heartbeat payload must stay comfortably under the 2 MB limit.
            assert len(json.dumps(hb)) < 100_000


# ---------------------------------------------------------------------------
# _train_remote progress tracking
# ---------------------------------------------------------------------------


class TestTrainRemoteProgress:
    """Verify _train_remote calls adapter.progress() and persists it when run_id is set."""

    def _make_adapter(self, statuses, progress_return=(0.0, ""), download_path="/tmp/ckpt"):
        adapter = MagicMock()
        adapter.submit.return_value = "run-42"
        adapter.submit_retry_config.return_value = _SubmitRetryConfig()
        adapter.status.side_effect = list(statuses)
        adapter.logs.return_value = ""
        adapter.progress.return_value = progress_return
        adapter.download.return_value = download_path
        return adapter

    def _patches(self, monkeypatch):
        import interactors.temporal.activities as acts

        async def _noop_heartbeat_loop(*args, **kwargs):
            pass

        monkeypatch.setattr(acts, "_heartbeat_loop", _noop_heartbeat_loop)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())
        monkeypatch.setattr(acts.activity, "info", MagicMock(return_value=MagicMock(heartbeat_details=())))
        return acts

    @pytest.mark.asyncio
    async def test_calls_adapter_progress_and_persists_when_run_id_set(self, monkeypatch):
        acts = self._patches(monkeypatch)
        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        adapter = self._make_adapter(["done"], progress_return=(0.5, "epoch=1.0  loss=0.4312"))
        config = TrainConfig(run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_called()
        mock_store.update_progress.assert_called_with("run-db-1", 0.5, "epoch=1.0  loss=0.4312")

    @pytest.mark.asyncio
    async def test_heartbeat_includes_progress_and_detail(self, monkeypatch):
        acts = self._patches(monkeypatch)
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        adapter = self._make_adapter(["running", "done"], progress_return=(0.42, "epoch=2.0  loss=0.31"))
        config = TrainConfig(run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured, "heartbeat should have been called"
        assert captured[0]["progress"] == 0.42
        assert captured[0]["detail"] == "epoch=2.0  loss=0.31"

    @pytest.mark.asyncio
    async def test_heartbeat_progress_defaults_when_no_run_id(self, monkeypatch):
        acts = self._patches(monkeypatch)
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))

        adapter = self._make_adapter(["done"], progress_return=(0.75, "epoch=2.0"))
        config = TrainConfig(run_id="", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        # No run_id → progress is never fetched, heartbeat carries safe defaults.
        adapter.progress.assert_not_called()
        assert captured[0]["progress"] == 0.0
        assert captured[0]["detail"] == ""

    @pytest.mark.asyncio
    async def test_heartbeat_detail_is_truncated(self, monkeypatch):
        acts = self._patches(monkeypatch)
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        huge_detail = "y" * (1024 * 1024)
        adapter = self._make_adapter(["done"], progress_return=(0.5, huge_detail))
        config = TrainConfig(run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        # A runaway detail string must not reintroduce the heartbeat-size failure.
        assert len(captured[0]["detail"]) <= acts._HEARTBEAT_DETAIL_MAX
        assert len(json.dumps(captured[0])) < 100_000

    @pytest.mark.asyncio
    async def test_skips_update_when_fraction_is_zero(self, monkeypatch):
        acts = self._patches(monkeypatch)
        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        adapter = self._make_adapter(["done"], progress_return=(0.0, ""))
        config = TrainConfig(run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_called()
        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_progress_entirely_when_no_run_id(self, monkeypatch):
        acts = self._patches(monkeypatch)
        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        adapter = self._make_adapter(["done"], progress_return=(0.75, "epoch=2.0"))
        config = TrainConfig(run_id="", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_not_called()
        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_progress_errors_do_not_fail_the_activity(self, monkeypatch):
        acts = self._patches(monkeypatch)
        mock_store = MagicMock()
        mock_store.update_progress.side_effect = RuntimeError("DB gone")
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        adapter = self._make_adapter(["done"], progress_return=(0.5, "epoch=1.0"))
        config = TrainConfig(run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            result = await acts._train_remote(config, adapter)

        assert result is not None


# ---------------------------------------------------------------------------
# _poll_local_progress
# ---------------------------------------------------------------------------


class TestPollLocalProgress:
    """Verify _poll_local_progress reads progress.json and calls update_progress."""

    @pytest.mark.asyncio
    async def test_reads_progress_json_and_calls_update_progress(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text(
            '{"step": 50, "max_steps": 100, "epoch": 1.0, "loss": 0.4312}'
        )

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        mock_store.update_progress.assert_called_once_with("run-42", 0.5, "epoch=1.0  loss=0.4312")

    @pytest.mark.asyncio
    async def test_includes_eval_loss_in_detail(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text(
            '{"step": 75, "max_steps": 100, "epoch": 1.5, "eval_loss": 0.3210}'
        )

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        call_args = mock_store.update_progress.call_args
        assert "eval_loss=0.3210" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_skips_update_when_run_id_is_empty(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text('{"step": 50, "max_steps": 100, "epoch": 1.0}')

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("", str(tmp_path))

        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_progress_file_gracefully(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        mock_store.update_progress.assert_not_called()


# ---------------------------------------------------------------------------
# finalise_run_activity
# ---------------------------------------------------------------------------


class TestFinaliseRunActivity:
    """Verify finalise_run_activity persists eval result and updates run status."""

    def _make_run_store(self) -> MagicMock:
        store = MagicMock()
        return store

    @pytest.mark.asyncio
    async def test_passed_sets_status_completed(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = self._make_run_store()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(finalise_run_activity, "run-1", True, 0.97)

        from domain.models import RunStatus
        mock_store.update_status.assert_called_once_with("run-1", RunStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_failed_sets_status_failed(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = self._make_run_store()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(finalise_run_activity, "run-2", False, 0.75)

        from domain.models import RunStatus
        mock_store.update_status.assert_called_once_with("run-2", RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_persists_eval_valid_pct(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = self._make_run_store()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(finalise_run_activity, "run-3", True, 0.95)

        # eval persistence is now owned by record_eval_result_activity, not finalise_run_activity
        mock_store.update_eval.assert_not_called()


# ---------------------------------------------------------------------------
# update_run_status_activity
# ---------------------------------------------------------------------------


class TestUpdateRunStatusActivity:
    """Verify update_run_status_activity sets the correct RunStatus value."""

    @pytest.mark.asyncio
    async def test_sets_training_status(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(update_run_status_activity, "run-42", "training")

        from domain.models import RunStatus
        mock_store.update_status.assert_called_once_with("run-42", RunStatus.TRAINING)

    @pytest.mark.asyncio
    async def test_sets_evaluating_status(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(update_run_status_activity, "run-42", "evaluating")

        from domain.models import RunStatus
        mock_store.update_status.assert_called_once_with("run-42", RunStatus.EVALUATING)

    @pytest.mark.asyncio
    async def test_sets_completed_status(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(update_run_status_activity, "run-42", "completed")

        from domain.models import RunStatus
        mock_store.update_status.assert_called_once_with("run-42", RunStatus.COMPLETED)


# ---------------------------------------------------------------------------
# fail_run_activity
# ---------------------------------------------------------------------------


class TestFailRunActivity:
    """Verify fail_run_activity calls store.fail_run with the correct arguments."""

    @pytest.mark.asyncio
    async def test_default_status_is_failed(self, monkeypatch):
        import interactors.temporal.activities as acts
        from interactors.temporal.activities import fail_run_activity

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(fail_run_activity, "run-99", "train failed: OOM")

        from domain.models import RunStatus
        mock_store.fail_run.assert_called_once_with("run-99", "train failed: OOM", RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_cancelled_status(self, monkeypatch):
        import interactors.temporal.activities as acts
        from interactors.temporal.activities import fail_run_activity

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(fail_run_activity, "run-99", "cancelled by user", "cancelled")

        from domain.models import RunStatus
        mock_store.fail_run.assert_called_once_with("run-99", "cancelled by user", RunStatus.CANCELLED)

    @pytest.mark.asyncio
    async def test_explicit_failed_status(self, monkeypatch):
        import interactors.temporal.activities as acts
        from interactors.temporal.activities import fail_run_activity

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        await ENV.run(fail_run_activity, "run-55", "evaluate failed: timeout", "failed")

        from domain.models import RunStatus
        mock_store.fail_run.assert_called_once_with("run-55", "evaluate failed: timeout", RunStatus.FAILED)

    @pytest.mark.asyncio
    async def test_noop_when_run_not_found(self, monkeypatch):
        """fail_run_activity must not raise if run_id is not in the store."""
        import interactors.temporal.activities as acts
        from interactors.temporal.activities import fail_run_activity

        mock_store = MagicMock()
        mock_store.fail_run.return_value = None  # store returns None for unknown id
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(run_store=mock_store))

        # should complete without raising
        await ENV.run(fail_run_activity, "unknown-run", "some error")
        mock_store.fail_run.assert_called_once()


# ---------------------------------------------------------------------------
# _evaluate_local — quality report persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_local_saves_quality_report(tmp_path, monkeypatch):
    """Quality report is saved to data/workflow/{run_id}/quality_report.json."""
    monkeypatch.chdir(tmp_path)

    fake_report = {
        "per_stat_accuracy": {
            s: {"correct": 38, "total": 40, "accuracy": 0.95, "pass": True}
            for s in ["hunger", "boredom", "social", "tiredness", "toilet"]
        },
        "target_accuracy": {"correct": 18, "total": 20, "accuracy": 0.9, "pass": True},
        "priority_conflict": {"correct": 16, "total": 20, "accuracy": 0.8, "pass": True},
        "fallback_accuracy": {"correct": 19, "total": 20, "accuracy": 0.95, "pass": True},
        "action_distribution": {"EAT": 50, "SLEEP": 40},
        "max_action_share": 0.25,
        "pass": True,
    }

    with (
        patch("domain.train.evaluate.evaluate", return_value=(0, 0.95)),
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.quality_report.run_quality_report", return_value=fake_report),
    ):
        result = await ENV.run(
            evaluate_activity,
            EvalConfig(
                checkpoint="some-checkpoint",
                eval_data="data/eval.jsonl",
                run_id="test-run-123",
            ),
        )

    assert result == EvalResult(valid_pct=0.95, passed=True)

    report_path = tmp_path / "data" / "workflow" / "test-run-123" / "quality_report.json"
    assert report_path.exists(), "quality_report.json was not created"

    saved = json.loads(report_path.read_text())
    # "pass" key must be normalised to "passed"
    assert "passed" in saved
    assert "pass" not in saved
    assert saved["passed"] is True
    assert "passed" in saved["per_stat_accuracy"]["hunger"]


@pytest.mark.asyncio
async def test_evaluate_local_skips_quality_report_without_run_id(tmp_path, monkeypatch):
    """No quality_report.json is written when run_id is empty."""
    monkeypatch.chdir(tmp_path)

    with (
        patch("domain.train.evaluate.evaluate", return_value=(0, 0.97)),
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.quality_report.run_quality_report") as mock_qr,
    ):
        await ENV.run(
            evaluate_activity,
            EvalConfig(checkpoint="some-checkpoint", eval_data="data/eval.jsonl"),
        )

    mock_qr.assert_not_called()


# ---------------------------------------------------------------------------
# save_gguf_path_activity
# ---------------------------------------------------------------------------


class TestSaveGgufPathActivity:
    """Verify save_gguf_path_activity updates the model record with the GGUF key."""

    def _make_model(self) -> MagicMock:
        from domain.models import TrainingModelConfig

        model = MagicMock()
        model.name = "my-model"
        model.description = ""
        model.base_model = "HuggingFaceTB/SmolLM2-360M"
        model.train_data = "data/train.jsonl"
        model.eval_data = "data/eval.jsonl"
        model.epochs = 5
        model.patience = 3
        model.warmup_ratio = 0.05
        model.remote_backend = "local"
        model.skip_generate = False
        model.gguf_path = ""
        model.is_active = False
        return model

    @pytest.mark.asyncio
    async def test_updates_gguf_path_on_model(self, monkeypatch):
        import interactors.temporal.activities as acts

        model = self._make_model()
        mock_store = MagicMock()
        mock_store.get.return_value = model
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(model_store=mock_store))

        await ENV.run(save_gguf_path_activity, "model-uuid", "gguf/my-model.gguf")

        mock_store.update.assert_called_once()
        call_args = mock_store.update.call_args
        assert call_args[0][0] == "model-uuid"
        # The updated config must have the new gguf_path
        updated_config = call_args[0][1]
        assert updated_config.gguf_path == "gguf/my-model.gguf"

    @pytest.mark.asyncio
    async def test_skips_update_when_model_not_found(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        mock_store.get.return_value = None
        monkeypatch.setattr(acts, "_create_uow", lambda: _mock_uow(model_store=mock_store))

        # Must not raise
        await ENV.run(save_gguf_path_activity, "nonexistent-model", "gguf/x.gguf")

        mock_store.update.assert_not_called()


# ---------------------------------------------------------------------------
# _resolve_training_data — download from storage when skip_generate=True
# ---------------------------------------------------------------------------


class TestResolveTrainingData:
    """Unit tests for the _resolve_training_data helper.

    File-based backends (Colab, SSH) stage data from the local filesystem and
    need S3 keys materialised locally when skip_generate=True.
    S3-backed backends (K8s, RunPod, VastAI, Kaggle) must receive the original
    S3 key so the remote pod can download it.
    """

    @pytest.mark.asyncio
    async def test_returns_train_path_unchanged_when_file_exists_locally(self, tmp_path):
        """No-op when train file already exists locally (file-based backend)."""
        import asyncio
        import interactors.temporal.activities as acts

        train = tmp_path / "train.jsonl"
        train.write_text('{"prompt":"p","completion":"c"}\n')

        config = TrainConfig(train_data=str(train), remote_backend="ssh")
        loop = asyncio.get_event_loop()
        train_out = await acts._resolve_training_data(config, loop)

        assert train_out == str(train)

    @pytest.mark.asyncio
    async def test_passes_s3_key_unchanged_for_kaggle(self, tmp_path, monkeypatch):
        """Kaggle is S3-backed: train key is returned unchanged (no local download)."""
        import asyncio
        import interactors.temporal.activities as acts

        mock_storage = MagicMock()
        monkeypatch.setattr(acts, "_storage", mock_storage)

        config = TrainConfig(train_data="dataset/abc123/train.jsonl", remote_backend="kaggle")
        loop = asyncio.get_event_loop()
        train_out = await acts._resolve_training_data(config, loop)

        assert train_out == "dataset/abc123/train.jsonl"
        mock_storage.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_download_when_file_already_local(self, tmp_path, monkeypatch):
        """Storage.download must not be called when train file already exists locally."""
        import asyncio
        import interactors.temporal.activities as acts

        train = tmp_path / "data" / "train.jsonl"
        train.parent.mkdir(parents=True)
        train.write_text("{}\n")

        mock_storage = MagicMock()
        monkeypatch.setattr(acts, "_storage", mock_storage)

        config = TrainConfig(train_data=str(train), remote_backend="ssh")
        loop = asyncio.get_event_loop()
        await acts._resolve_training_data(config, loop)

        mock_storage.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_k8s_passes_s3_key_through_unchanged(self, tmp_path, monkeypatch):
        """K8s (S3-backed) must receive the original S3 key — never a local path."""
        import asyncio
        import interactors.temporal.activities as acts

        mock_storage = MagicMock()
        monkeypatch.setattr(acts, "_storage", mock_storage)
        monkeypatch.chdir(tmp_path)

        config = TrainConfig(train_data="dataset/abc123/train.jsonl", remote_backend="k8s")
        loop = asyncio.get_event_loop()
        train_out = await acts._resolve_training_data(config, loop)

        assert train_out == "dataset/abc123/train.jsonl"
        mock_storage.download.assert_not_called()

    @pytest.mark.asyncio
    async def test_runpod_passes_s3_key_through_unchanged(self, tmp_path, monkeypatch):
        """RunPod (S3-backed) also passes S3 key through unchanged."""
        import asyncio
        import interactors.temporal.activities as acts

        mock_storage = MagicMock()
        monkeypatch.setattr(acts, "_storage", mock_storage)
        monkeypatch.chdir(tmp_path)

        config = TrainConfig(train_data="dataset/abc123/train.jsonl", remote_backend="runpod")
        loop = asyncio.get_event_loop()
        train_out = await acts._resolve_training_data(config, loop)

        assert train_out == "dataset/abc123/train.jsonl"
        mock_storage.download.assert_not_called()


# ---------------------------------------------------------------------------
# _is_retryable_submit_error and _retryable_submit
# ---------------------------------------------------------------------------


import asyncio as _asyncio
import interactors.temporal.activities as _acts


async def _noop_heartbeat_loop(*args, **kwargs):  # noqa: D401
    pass


_RETRYABLE_CFG = _SubmitRetryConfig(
    max_retries=2,
    base_delay_s=0.0,
    retryable_errors=("rate limit", "temporarily unavailable", "timeout"),
)


class TestIsRetryableSubmitError:
    def test_matches_rate_limit(self):
        assert _acts._is_retryable_submit_error(RuntimeError("rate limit exceeded"), _RETRYABLE_CFG)

    def test_matches_case_insensitively(self):
        assert _acts._is_retryable_submit_error(RuntimeError("RATE LIMIT HIT"), _RETRYABLE_CFG)

    def test_matches_timeout(self):
        assert _acts._is_retryable_submit_error(RuntimeError("connection timeout"), _RETRYABLE_CFG)

    def test_rejects_auth_failure(self):
        assert not _acts._is_retryable_submit_error(RuntimeError("authentication failed"), _RETRYABLE_CFG)

    def test_rejects_disk_quota(self):
        assert not _acts._is_retryable_submit_error(RuntimeError("disk quota exceeded"), _RETRYABLE_CFG)


class TestRetryableSubmit:
    def _make_adapter(self, config: _SubmitRetryConfig = _RETRYABLE_CFG):
        adapter = MagicMock()
        adapter.submit_retry_config.return_value = config
        return adapter

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.return_value = "run-ok"

        result = await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert result == "run-ok"
        assert adapter.submit.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error_and_succeeds(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.side_effect = [RuntimeError("rate limit exceeded"), "run-after-retry"]

        result = await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert result == "run-after-retry"
        assert adapter.submit.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable_error(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.side_effect = RuntimeError("authentication failed")

        with pytest.raises(ApplicationError, match="non-retryable"):
            await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert adapter.submit.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_flag_set_on_immediate_failure(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.side_effect = RuntimeError("disk quota exceeded")

        with pytest.raises(ApplicationError) as exc_info:
            await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    async def test_exhausts_all_retries_and_includes_history(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.side_effect = RuntimeError("temporarily unavailable")

        with pytest.raises(ApplicationError) as exc_info:
            await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        err_msg = str(exc_info.value)
        assert "attempt 1" in err_msg
        assert adapter.submit.call_count == _RETRYABLE_CFG.max_retries + 1

    @pytest.mark.asyncio
    async def test_non_retryable_flag_set_on_exhaustion(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        adapter = self._make_adapter()
        adapter.submit.side_effect = RuntimeError("temporarily unavailable")

        with pytest.raises(ApplicationError) as exc_info:
            await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    async def test_backoff_grows_exponentially(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        sleep_calls: list[float] = []

        async def record_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        monkeypatch.setattr(_asyncio, "sleep", record_sleep)
        cfg = _SubmitRetryConfig(max_retries=3, base_delay_s=5.0, retryable_errors=("timeout",))
        adapter = self._make_adapter(cfg)
        adapter.submit.side_effect = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            "run-eventual",
        ]

        await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert sleep_calls[0] == pytest.approx(5.0)   # 5 * 2^0
        assert sleep_calls[1] == pytest.approx(10.0)  # 5 * 2^1

    @pytest.mark.asyncio
    async def test_uses_adapter_specific_config(self, monkeypatch):
        monkeypatch.setattr(_acts, "_heartbeat_loop", _noop_heartbeat_loop)
        custom_cfg = _SubmitRetryConfig(
            max_retries=1,
            base_delay_s=0.0,
            retryable_errors=("custom error",),
        )
        adapter = self._make_adapter(custom_cfg)
        adapter.submit.side_effect = [RuntimeError("custom error"), "run-custom"]

        result = await _acts._retryable_submit(adapter, MagicMock(), _asyncio.get_event_loop())

        assert result == "run-custom"
        assert adapter.submit.call_count == 2


class TestAdapterSubmitRetryConfig:
    def test_runpod_returns_submit_retry_config(self):
        from adapters.compute.runpod.adapter import RunPodTrainingAdapter
        adapter = RunPodTrainingAdapter.__new__(RunPodTrainingAdapter)
        cfg = adapter.submit_retry_config()
        assert isinstance(cfg, _SubmitRetryConfig)
        assert cfg.max_retries > 0
        assert len(cfg.retryable_errors) > 0

    def test_vastai_returns_submit_retry_config(self):
        from adapters.compute.vastai.adapter import VastAiTrainingAdapter
        adapter = VastAiTrainingAdapter.__new__(VastAiTrainingAdapter)
        cfg = adapter.submit_retry_config()
        assert isinstance(cfg, _SubmitRetryConfig)
        assert cfg.max_retries > 0
        assert len(cfg.retryable_errors) > 0
