"""Tests for the signed read client (TenantClient) and verify_transfer.

Offline via httpx.MockTransport, mirroring test_upload.py — a handler routes by
path/query and returns canned coordinator responses. Asserts the requests are
RFC 9421-signed, pagination follows next_cursor, errors map to CoordinatorError,
and the export bundle's custody signature verifies.
"""

from __future__ import annotations

import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from auspexai_tenant.client import CoordinatorError, TenantClient, verify_transfer
from auspexai_tenant.signing import MaintainerKey

COORD = "https://coord.test"


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _tenant(handler) -> TenantClient:
    return TenantClient(COORD, MaintainerKey.generate(), client=_client_for(handler))


def test_requests_are_rfc9421_signed() -> None:
    seen: list[httpx.Request] = []
    key = MaintainerKey.generate()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"experiments": []})

    TenantClient(COORD, key, client=_client_for(handler)).list_experiments()
    assert "Signature" in seen[0].headers
    assert f'keyid="{key.pubkey_hex}"' in seen[0].headers["Signature-Input"]


def test_list_experiments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/experiments"
        return httpx.Response(200, json={"experiments": [{"experiment_id": "exp-1"}]})

    assert _tenant(handler).list_experiments() == [{"experiment_id": "exp-1"}]


def test_get_attestation_parses_response() -> None:
    body = {
        "attestation_id": "att-1",
        "experiment_id": "exp-label",
        "tenant_id": "tenant-a",
        "merkle_root": "abc123",
        "algorithm": "sha256-merkle-v0",
        "unit_count": 2,
        "units": [
            {"unit_id": "u1", "consensus_result_hash": "h1", "receipt_id": "rcpt-1"},
            {"unit_id": "u2", "consensus_result_hash": "h2", "receipt_id": "rcpt-2"},
        ],
        "cose_b64": "AAAA",
        "signing_key_pubkey_hex": "ab" * 32,
        "rekor_log_index": 0,
        "rekor_entry_uuid": "lab-mode-no-rekor",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/experiments/exp-1/attestation"
        return httpx.Response(200, json=body)

    att = _tenant(handler).get_attestation("exp-1")
    assert att.attestation_id == "att-1"
    assert att.merkle_root == "abc123"
    assert att.unit_count == 2
    assert [u["unit_id"] for u in att.units] == ["u1", "u2"]
    assert att.partial is False  # no partial key in response → False


def test_get_attestation_checkpoint_passes_query_and_parses_partial() -> None:
    """M9 leg 2: checkpoint=True sends ?checkpoint=true and surfaces partial."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/experiments/exp-1/attestation"
        assert request.url.params.get("checkpoint") == "true"
        return httpx.Response(
            200,
            json={
                "attestation_id": "att-1",
                "experiment_id": "exp-label",
                "tenant_id": "tenant-a",
                "merkle_root": "abc123",
                "algorithm": "sha256-merkle-v0",
                "unit_count": 1,
                "units": [{"unit_id": "u1", "consensus_result_hash": "h1", "receipt_id": "rcpt-1"}],
                "cose_b64": "AAAA",
                "signing_key_pubkey_hex": "ab" * 32,
                "rekor_log_index": 0,
                "rekor_entry_uuid": "lab-mode-no-rekor",
                "partial": True,
            },
        )

    att = _tenant(handler).get_attestation("exp-1", checkpoint=True)
    assert att.partial is True

    # default (no checkpoint) must NOT send the query param
    def handler_no_cp(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("checkpoint") is None
        return httpx.Response(
            200,
            json={
                "attestation_id": "att-2",
                "experiment_id": "exp-label",
                "tenant_id": "tenant-a",
                "merkle_root": "abc123",
                "algorithm": "sha256-merkle-v0",
                "unit_count": 0,
                "units": [],
                "cose_b64": "AAAA",
                "signing_key_pubkey_hex": "ab" * 32,
                "rekor_log_index": 0,
                "rekor_entry_uuid": "lab-mode-no-rekor",
            },
        )

    assert _tenant(handler_no_cp).get_attestation("exp-1").partial is False


def test_get_results_passes_include_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("include") == "raw"
        return httpx.Response(200, json={"results": [{"result_id": "r1"}], "next_cursor": None})

    page = _tenant(handler).get_results("exp-1", include="raw")
    assert page["results"] == [{"result_id": "r1"}]


def test_iter_results_follows_cursor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={"results": [{"result_id": "r1"}], "next_cursor": "c1"})
        assert cursor == "c1"
        return httpx.Response(200, json={"results": [{"result_id": "r2"}], "next_cursor": None})

    got = [r["result_id"] for r in _tenant(handler).iter_results("exp-1")]
    assert got == ["r1", "r2"]


def test_get_receipts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/experiments/exp-1/receipts"
        return httpx.Response(200, json={"receipts": [{"receipt_id": "rcpt-1"}]})

    assert _tenant(handler).get_receipts("exp-1") == [{"receipt_id": "rcpt-1"}]


def test_non_2xx_raises_coordinator_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"code": "forbidden"}})

    with pytest.raises(CoordinatorError) as exc:
        _tenant(handler).list_experiments()
    assert exc.value.status_code == 403


def _signed_bundle(*, tamper: bool = False) -> dict:
    """An export bundle whose transfer record is signed by a coordinator key."""
    coord = Ed25519PrivateKey.generate()
    coord_pub = coord.public_key().public_bytes_raw().hex()
    root = "deadbeef"
    collected_by, collected_at, manifest_hash = "ab" * 32, "2026-05-31T00:00:00+00:00", "mh"
    message = f"{root}|{collected_by}|{collected_at}|{manifest_hash}".encode()
    sig = coord.sign(message).hex()
    if tamper:
        root = " feedface"  # change the signed content after signing → invalid
    return {
        "consensus_results": [{"result_id": "r1"}],
        "receipts": [{"receipt_id": "rcpt-1", "cose_b64": "AA=="}],
        "transfer": {
            "transfer_id": "xfer-1",
            "result_set_root": root,
            "collected_by_pubkey": collected_by,
            "collected_at": collected_at,
            "manifest_hash": manifest_hash,
            "coordinator_signature": sig,
            "coordinator_pubkey_hex": coord_pub,
        },
    }


def test_export_returns_bundle_and_transfer_verifies() -> None:
    bundle = _signed_bundle()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/experiments/exp-1/results/export"
        return httpx.Response(200, json=bundle)

    fetched = _tenant(handler).export("exp-1")
    assert fetched["consensus_results"] == [{"result_id": "r1"}]
    v = verify_transfer(fetched)
    assert v.valid is True
    assert v.transfer_id == "xfer-1"


def test_verify_transfer_detects_tampering() -> None:
    assert verify_transfer(_signed_bundle(tamper=True)).valid is False


def test_export_bundle_round_trips_as_json() -> None:
    # A saved bundle must be plain JSON (the CLI writes it to disk).
    bundle = _signed_bundle()
    assert json.loads(json.dumps(bundle))["transfer"]["transfer_id"] == "xfer-1"


# ---- demand-board (model requests, M2) -------------------------------------


def test_get_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v0/models/catalog"
        return httpx.Response(
            200,
            json={"models": [{"model_id": "m-x", "worker_count": 2}], "total_active_workers": 3},
        )

    cat = _tenant(handler).get_catalog()
    assert cat["models"][0]["model_id"] == "m-x"
    assert cat["total_active_workers"] == 3


def test_request_model_posts_signed_body() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["signed"] = "Signature" in request.headers
        return httpx.Response(
            201, json={"request_id": "mrq-1", "status": "pending", "model_id": "qwen3-q4"}
        )

    out = _tenant(handler).request_model("qwen3-q4", reason="need it", hf_repo="Qwen/X-GGUF")
    assert out["status"] == "pending"
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/v0/model-requests"
    assert seen["body"] == {"model_id": "qwen3-q4", "reason": "need it", "hf_repo": "Qwen/X-GGUF"}
    assert seen["signed"]  # RFC 9421 signed


def test_request_model_omits_empty_hf_repo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "hf_repo" not in body
        return httpx.Response(201, json={"request_id": "mrq-2", "status": "available"})

    assert _tenant(handler).request_model("m-y", reason="x")["status"] == "available"
