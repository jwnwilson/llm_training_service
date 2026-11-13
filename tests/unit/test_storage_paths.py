"""Tests for canonical S3 key constructors and their retention safety.

Finished GGUF models and datasets must never be written under an S3 prefix
that a bucket lifecycle rule expires. A rule expiring ``workflow/`` after 30 days once
silently deleted every pipeline-trained model, because run models lived at
``workflow/{run_id}/model/``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from adapters.storage.paths import (
    dataset_eval_key,
    dataset_train_key,
    run_dataset_eval_key,
    run_dataset_train_key,
    run_model_key,
    standalone_model_key,
)

TERRAFORM_MAIN = Path(__file__).resolve().parents[2] / "infra" / "terraform" / "main.tf"
RULE_BLOCK = re.compile(r"rule\s*\{(?P<body>.*?)\n  \}", re.DOTALL)
PREFIX = re.compile(r'prefix\s*=\s*"(?P<prefix>[^"]*)"')
ENABLED = re.compile(r'status\s*=\s*"Enabled"')
# Matches `expiration { days = N }` but not `noncurrent_version_expiration`, and
# not `expired_object_delete_marker`, which only removes empty delete markers.
CURRENT_EXPIRY = re.compile(r"(?<![\w])expiration\s*\{[^}]*\bdays\s*=")


def _expiring_prefixes() -> list[str]:
    """Return prefixes of enabled lifecycle rules that expire current objects."""
    text = TERRAFORM_MAIN.read_text()
    prefixes: list[str] = []
    for match in RULE_BLOCK.finditer(text):
        body = match.group("body")
        if not CURRENT_EXPIRY.search(body) or not ENABLED.search(body):
            continue
        prefix_match = PREFIX.search(body)
        # A rule with no prefix filter applies to the whole bucket.
        prefixes.append(prefix_match.group("prefix") if prefix_match else "")
    return prefixes


def test_run_model_key_is_scoped_by_run_under_model_namespace():
    assert run_model_key("r1", "my-pet") == "model/runs/r1/my-pet.gguf"


def test_run_model_key_defaults_filename_to_model():
    assert run_model_key("r1") == "model/runs/r1/model.gguf"


def test_run_model_key_is_not_under_workflow_prefix():
    assert not run_model_key("r1").startswith("workflow/")


def test_run_dataset_keys_are_scoped_by_run_under_dataset_namespace():
    assert run_dataset_train_key("r1") == "dataset/runs/r1/train.jsonl"
    assert run_dataset_eval_key("r1") == "dataset/runs/r1/eval.jsonl"


def test_run_dataset_keys_are_not_under_workflow_prefix():
    assert not run_dataset_train_key("r1").startswith("workflow/")
    assert not run_dataset_eval_key("r1").startswith("workflow/")


def test_terraform_lifecycle_rules_are_parseable():
    # Guard against the regex silently matching nothing if main.tf is restructured.
    text = TERRAFORM_MAIN.read_text()
    assert "aws_s3_bucket_lifecycle_configuration" in text
    assert RULE_BLOCK.search(text) is not None


@pytest.mark.parametrize(
    "key",
    [
        run_model_key("r1"),
        run_model_key("r1", "named"),
        standalone_model_key("m1", "named"),
        run_dataset_train_key("r1"),
        run_dataset_eval_key("r1"),
        dataset_train_key("d1"),
        dataset_eval_key("d1"),
    ],
)
def test_persistent_keys_are_never_under_an_expiring_lifecycle_prefix(key):
    expiring = _expiring_prefixes()
    offending = [p for p in expiring if key.startswith(p)]
    assert not offending, f"{key!r} would be deleted by lifecycle rule(s) for {offending}"
