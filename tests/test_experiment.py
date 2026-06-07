"""Tests for the stateful per-experiment client (`Experiment`, M8 §3.1).

Offline via httpx.MockTransport, mirroring test_client.py / test_upload.py — a
handler routes by method+path and returns canned coordinator responses. Asserts
envelopes are built from the experiment metadata, requests are RFC 9421-signed,
the 409 family maps to typed errors (the crash-resume idempotency contract),
pagination follows the cursor, and progress() derives reduce_ready from status.
"""

from __future__ import annotations

import json

import httpx
import pytest

from auspexai_tenant.client import CoordinatorError
from auspexai_tenant.experiment import (
    ROUND_PAYLOAD_KEY,
    Experiment,
    LifecycleConflictError,
    MaxUnitsExceededError,
    Progress,
    ResultPage,
    SubmissionsFinalizedError,
    Unit,
    UnitsAlreadySubmittedError,
)
from auspexai_tenant.signing import MaintainerKey

COORD = "https://coord.test"
EXP_ID = "exp-test"  # coordinator id (URL path)
TENANT = "tenant-a"
LABEL = "doubler-001"  # tenant_experiment_label (goes in the unit envelope)
MANIFEST = "ab" * 32

_META = {
    "experiment_id": EXP_ID,
    "tenant_id": TENANT,
    "tenant_experiment_label": LABEL,
    "manifest_hash": MANIFEST,
    "status": "approved",
    "submissions_finalized": False,
}


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _experiment(handler, *, key: MaintainerKey | None = None) -> Experiment:
    return Experiment(COORD, key or MaintainerKey.generate(), EXP_ID, client=_client_for(handler))


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content) if request.content else {}


# ---- envelope construction + signing --------------------------------------


def test_submit_units_builds_envelopes_from_metadata_and_signs() -> None:
    seen: list[httpx.Request] = []
    key = MaintainerKey.generate()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_META)
        return httpx.Response(201, json={"submitted_unit_ids": ["u1"], "count": 1})

    result = _experiment(handler, key=key).submit_units([Unit("u1", {"value": 21})])

    assert result.submitted_unit_ids == ["u1"]
    assert result.count == 1
    post = seen[-1]
    assert post.method == "POST"
    assert post.url.path == f"/api/v0/experiments/{EXP_ID}/work-units"
    # RFC 9421-signed with the tenant key.
    assert "Signature" in post.headers
    assert f'keyid="{key.pubkey_hex}"' in post.headers["Signature-Input"]
    # Envelope carries the LABEL + manifest hash from metadata, not the coord id.
    env = _body(post)["work_units"][0]
    assert env["unit_id"] == "u1"
    assert env["tenant_id"] == TENANT
    assert env["experiment_id"] == LABEL
    assert env["manifest_sha256"] == MANIFEST
    assert env["payload"] == {"value": 21}


def test_round_rides_in_payload_under_reserved_key() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_META)
        captured.update(_body(request))
        return httpx.Response(201, json={"submitted_unit_ids": ["u1"], "count": 1})

    _experiment(handler).submit_units([Unit("u1", {"value": 1})], round=7)
    payload = captured["work_units"][0]["payload"]
    assert payload[ROUND_PAYLOAD_KEY] == 7
    assert payload["value"] == 1  # original payload preserved


def test_submit_units_does_not_mutate_caller_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_META)
        return httpx.Response(201, json={"submitted_unit_ids": ["u1"], "count": 1})

    original = {"value": 1}
    _experiment(handler).submit_units([Unit("u1", original)], round=3)
    assert original == {"value": 1}  # round embedding used a copy


def test_metadata_is_cached_across_submits() -> None:
    meta_gets = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/api/v0/experiments/{EXP_ID}":
            meta_gets["n"] += 1
            return httpx.Response(200, json=_META)
        return httpx.Response(201, json={"submitted_unit_ids": ["u"], "count": 1})

    exp = _experiment(handler)
    exp.submit_units([Unit("u1", {})])
    exp.submit_units([Unit("u2", {})])
    assert meta_gets["n"] == 1  # immutable metadata fetched once


def test_submit_units_empty_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_META)

    with pytest.raises(ValueError, match="at least one unit"):
        _experiment(handler).submit_units([])


# ---- the 409 idempotency family -------------------------------------------


def _conflict(code: str) -> httpx.Response:
    return httpx.Response(409, json={"error": {"code": code, "message": code}})


@pytest.mark.parametrize(
    ("code", "exc"),
    [
        ("unit_id_already_submitted", UnitsAlreadySubmittedError),
        ("max_units_exceeded", MaxUnitsExceededError),
        ("submissions_finalized", SubmissionsFinalizedError),
    ],
)
def test_submit_units_409_maps_to_typed_error(code: str, exc: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_META)
        return _conflict(code)

    with pytest.raises(exc) as ei:
        _experiment(handler).submit_units([Unit("u1", {})])
    assert ei.value.status_code == 409


def test_submit_units_other_error_is_generic_coordinator_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_META)
        return httpx.Response(403, json={"error": {"code": "researcher_required"}})

    with pytest.raises(CoordinatorError) as ei:
        _experiment(handler).submit_units([Unit("u1", {})])
    assert ei.value.status_code == 403
    assert not isinstance(ei.value, UnitsAlreadySubmittedError)


# ---- lifecycle actions -----------------------------------------------------


def test_finalize_posts_empty_body_to_action() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={**_META, "submissions_finalized": True})

    out = _experiment(handler).finalize()
    assert out["submissions_finalized"] is True
    assert seen[-1].url.path == f"/api/v0/experiments/{EXP_ID}/actions/finalize-submissions"
    assert seen[-1].content == b""  # finalize carries no body


def test_abort_posts_to_action() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_META, "status": "aborted"})

    assert _experiment(handler).abort()["status"] == "aborted"


def test_action_409_raises_lifecycle_conflict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _conflict("finalize_not_applicable")

    with pytest.raises(LifecycleConflictError):
        _experiment(handler).finalize()


# ---- progress (reads = source of truth) ------------------------------------


def test_progress_combines_experiment_and_activity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/activity"):
            return httpx.Response(
                200,
                json={
                    "total_work_units": 10,
                    "work_unit_counts": {"completed": 4, "pending": 6},
                    "completions_total": 12,
                    "replication_target_total": 30,
                    "active_contributor_count": 3,
                    "network_active_workers": 5,
                },
            )
        return httpx.Response(200, json=_META)

    p = _experiment(handler).progress()
    assert isinstance(p, Progress)
    assert p.status == "approved"
    assert p.reduce_ready is False
    assert p.is_terminal is False
    assert p.total_work_units == 10
    assert p.work_unit_counts == {"completed": 4, "pending": 6}
    assert p.completions_total == 12
    assert p.network_active_workers == 5


def test_progress_reduce_ready_when_completed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/activity"):
            return httpx.Response(200, json={})
        return httpx.Response(200, json={**_META, "status": "completed"})

    p = _experiment(handler).progress()
    assert p.reduce_ready is True
    assert p.is_terminal is True


# ---- results pagination ----------------------------------------------------


def test_results_page_returns_cursor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("include") == "consensus"
        return httpx.Response(200, json={"results": [{"unit_id": "u1"}], "next_cursor": "c2"})

    page = _experiment(handler).results_page()
    assert isinstance(page, ResultPage)
    assert page.results == [{"unit_id": "u1"}]
    assert page.next_cursor == "c2"


def test_results_follows_pagination_from_since() -> None:
    pages = {
        None: {"results": [{"unit_id": "u1"}], "next_cursor": "c2"},
        "c2": {"results": [{"unit_id": "u2"}], "next_cursor": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        return httpx.Response(200, json=pages[cursor])

    got = list(_experiment(handler).results())
    assert [r["unit_id"] for r in got] == ["u1", "u2"]


def test_results_resumes_from_given_cursor() -> None:
    seen_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_cursors.append(request.url.params.get("cursor"))
        return httpx.Response(200, json={"results": [], "next_cursor": None})

    list(_experiment(handler).results(since="c5"))
    assert seen_cursors == ["c5"]


# ---- attestation delegation ------------------------------------------------


def test_attestation_checkpoint_passes_through() -> None:
    seen: list[httpx.Request] = []
    att_body = {
        "attestation_id": "att-1",
        "experiment_id": LABEL,
        "tenant_id": TENANT,
        "merkle_root": "root",
        "algorithm": "sha256-merkle-v0",
        "unit_count": 1,
        "units": [{"unit_id": "u1", "consensus_result_hash": "h1", "receipt_id": "r1"}],
        "cose_b64": "AAAA",
        "signing_key_pubkey_hex": "cd" * 32,
        "rekor_log_index": 0,
        "rekor_entry_uuid": "lab-mode-no-rekor",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=att_body)

    att = _experiment(handler).attestation(checkpoint=True)
    assert att.merkle_root == "root"
    assert seen[-1].url.params.get("checkpoint") == "true"
