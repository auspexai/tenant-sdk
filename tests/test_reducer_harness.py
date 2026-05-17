"""Tests for ReducerHarness — the SDK helper for writing tenant reducers.

Covers the reducer-invocation contract (--results dir, --output file) per
sdk_license_boundary_position.md §6.5 and the exit-code conventions
(0 = success; 1 = tenant code failure; 2 = harness/IO failure).
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest

from auspexai_tenant.reducer import ReducerDecision, ReducerHarness
from auspexai_tenant.schemas import load_schema
from auspexai_tenant.workunits import Result

# ---- helpers ----------------------------------------------------------------


def _make_result(unit_id: str, worker_id: int, payload: dict) -> dict:
    """Build a Result-shaped dict (worker_pubkey + signature are placeholder
    but valid-format for schema purposes)."""
    pubkey_hex = f"{worker_id:064x}"
    return {
        "schema_version": "0.1",
        "unit_id": unit_id,
        "worker_pubkey": pubkey_hex,
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "exit_code": 0,
        "payload": payload,
        "worker_signature": base64.b64encode(b"\x00" * 64).decode("ascii"),
    }


def _stage_results(tmp_path: Path, results: list[dict]) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    for i, r in enumerate(results):
        (d / f"result_{i:04d}.json").write_text(json.dumps(r))
    return d


def _args(results_dir: Path, output_path: Path) -> list[str]:
    return ["--results", str(results_dir), "--output", str(output_path)]


# ---- happy paths ------------------------------------------------------------


def test_reducer_happy_path_agree(tmp_path: Path) -> None:
    results_dir = _stage_results(
        tmp_path,
        [
            _make_result("u1", 1, {"score": 0.42}),
            _make_result("u1", 2, {"score": 0.42}),
            _make_result("u1", 3, {"score": 0.42}),
        ],
    )
    output_path = tmp_path / "decision.json"

    def reduce(results: list[Result]) -> ReducerDecision:
        assert len(results) == 3
        assert all(r.unit_id == "u1" for r in results)
        first = results[0].payload
        if all(r.payload == first for r in results):
            return ReducerDecision(schema_version="0.1", verdict="agree", merged=first)
        return ReducerDecision(schema_version="0.1", verdict="disagree")

    rc = ReducerHarness(reduce).main(_args(results_dir, output_path))
    assert rc == 0

    decision = ReducerDecision.model_validate(json.loads(output_path.read_text()))
    assert decision.verdict == "agree"
    assert decision.merged == {"score": 0.42}


def test_reducer_happy_path_disagree(tmp_path: Path) -> None:
    results_dir = _stage_results(
        tmp_path,
        [
            _make_result("u1", 1, {"score": 0.42}),
            _make_result("u1", 2, {"score": 0.99}),
        ],
    )
    output_path = tmp_path / "decision.json"

    def reduce(results: list[Result]) -> ReducerDecision:
        return ReducerDecision(schema_version="0.1", verdict="disagree")

    rc = ReducerHarness(reduce).main(_args(results_dir, output_path))
    assert rc == 0
    decision = ReducerDecision.model_validate(json.loads(output_path.read_text()))
    assert decision.verdict == "disagree"
    assert decision.merged is None


# ---- harness/IO failures (exit 2) -------------------------------------------


def test_reducer_missing_results_dir(tmp_path: Path) -> None:
    output_path = tmp_path / "decision.json"
    rc = ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={})
    ).main(_args(tmp_path / "nonexistent", output_path))
    assert rc == 2


def test_reducer_empty_results_dir(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={})
    ).main(_args(results_dir, output_path))
    assert rc == 2


def test_reducer_malformed_result_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "result_0001.json").write_text("{not valid json")
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={})
    ).main(_args(results_dir, output_path))
    assert rc == 2


def test_reducer_result_fails_schema(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "result_0001.json").write_text(json.dumps({"not_a_result": True}))
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={})
    ).main(_args(results_dir, output_path))
    assert rc == 2


# ---- tenant-code failures (exit 1) ------------------------------------------


def test_reducer_tenant_raises(tmp_path: Path) -> None:
    results_dir = _stage_results(tmp_path, [_make_result("u1", 1, {"x": 1})])
    output_path = tmp_path / "decision.json"

    def boom(_r: list[Result]) -> ReducerDecision:
        raise RuntimeError("tenant reducer exploded")

    rc = ReducerHarness(boom).main(_args(results_dir, output_path))
    assert rc == 1
    assert not output_path.exists()


def test_reducer_tenant_returns_wrong_type(tmp_path: Path) -> None:
    results_dir = _stage_results(tmp_path, [_make_result("u1", 1, {"x": 1})])
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(lambda _r: "not a decision").main(  # type: ignore[arg-type, return-value]
        _args(results_dir, output_path)
    )
    assert rc == 1


def test_reducer_tenant_returns_dict(tmp_path: Path) -> None:
    """Even a valid-shaped dict is rejected — must be a ReducerDecision instance."""
    results_dir = _stage_results(tmp_path, [_make_result("u1", 1, {"x": 1})])
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(lambda _r: {"verdict": "agree", "merged": {}}).main(  # type: ignore[arg-type, return-value]
        _args(results_dir, output_path)
    )
    assert rc == 1


# ---- ReducerDecision model rules --------------------------------------------


def test_decision_requires_merged_on_agree() -> None:
    with pytest.raises(ValueError, match="merged is required"):
        ReducerDecision(schema_version="0.1", verdict="agree")


def test_decision_forbids_merged_on_disagree() -> None:
    with pytest.raises(ValueError, match="merged must be omitted"):
        ReducerDecision(schema_version="0.1", verdict="disagree", merged={"x": 1})


# ---- output validates against published JSON Schema -------------------------


def test_decision_matches_published_schema(tmp_path: Path) -> None:
    results_dir = _stage_results(tmp_path, [_make_result("u1", 1, {"x": 1})])
    output_path = tmp_path / "decision.json"

    rc = ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={"x": 1})
    ).main(_args(results_dir, output_path))
    assert rc == 0

    schema = load_schema("reducer_decision_v0_1.json")
    written = json.loads(output_path.read_text())
    jsonschema.validate(written, schema)


# ---- atomic write -----------------------------------------------------------


def test_reducer_atomic_rename_no_partial(tmp_path: Path) -> None:
    results_dir = _stage_results(tmp_path, [_make_result("u1", 1, {"x": 1})])
    output_path = tmp_path / "decision.json"

    ReducerHarness(
        lambda _r: ReducerDecision(schema_version="0.1", verdict="agree", merged={"x": 1})
    ).main(_args(results_dir, output_path))

    assert list(tmp_path.glob("*.tmp")) == []
