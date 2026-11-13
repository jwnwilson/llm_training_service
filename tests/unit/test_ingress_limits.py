"""Guard the llm-api ingress against rejecting uploads the API accepts.

ingress-nginx defaults ``client_max_body_size`` to 1 MB, so without an explicit
``proxy-body-size`` annotation a normal 5,000-example train.jsonl (~3 MB) is
rejected with an nginx 413 before it ever reaches ``POST /api/datasets``.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from interactors.api.routes.datasets import MAX_UPLOAD_BYTES

INGRESS_YAML = Path(__file__).resolve().parents[2] / "infra" / "k8s" / "llm-api" / "ingress.yaml"
BODY_SIZE_ANNOTATION = "nginx.ingress.kubernetes.io/proxy-body-size"
NGINX_SIZE = re.compile(r"^(?P<num>\d+)(?P<unit>[kKmMgG]?)$")
UNIT_BYTES = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def _parse_nginx_size(value: str) -> int:
    match = NGINX_SIZE.match(value.strip())
    assert match, f"unparseable nginx size {value!r}"
    return int(match.group("num")) * UNIT_BYTES[match.group("unit").lower()]


def _ingress_annotations() -> dict[str, str]:
    ingress = yaml.safe_load(INGRESS_YAML.read_text())
    return ingress["metadata"].get("annotations", {})


def test_parse_nginx_size_handles_units():
    assert _parse_nginx_size("512") == 512
    assert _parse_nginx_size("8k") == 8 * 1024
    assert _parse_nginx_size("60m") == 60 * 1024**2
    assert _parse_nginx_size("1G") == 1024**3


def test_ingress_sets_explicit_proxy_body_size():
    assert BODY_SIZE_ANNOTATION in _ingress_annotations(), (
        "Without proxy-body-size, ingress-nginx caps request bodies at 1 MB"
    )


def test_ingress_body_size_allows_max_dataset_upload():
    limit = _parse_nginx_size(_ingress_annotations()[BODY_SIZE_ANNOTATION])
    # Strictly greater: a MAX_UPLOAD_BYTES file plus multipart framing must pass,
    # so the API (not nginx) is the component that rejects oversized files.
    assert limit > MAX_UPLOAD_BYTES
