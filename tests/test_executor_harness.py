"""Tests for ExecutorHarness — the SDK helper for writing tenant executors.

Covers the executor-invocation contract (--input, --output, --models, --timeout)
per sdk_license_boundary_position.md §6.4 and the exit-code conventions
(0 = success; 1 = tenant code failure; 2 = harness/IO failure).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auspexai_tenant.executor import ExecutorHarness
from auspexai_tenant.workunits import ExecutorOutput, WorkUnit

FIXTURES = Path(__file__).parent / "fixtures"


# ---- helpers ----------------------------------------------------------------


def _stage_workunit(tmp_path: Path, fixture: str = "valid_workunit.json") -> Path:
    """Copy a work-unit fixture into a temp dir, return its path."""
    src = FIXTURES / fixture
    dst = tmp_path / "work_unit.json"
    dst.write_text(src.read_text())
    return dst


def _stage_models_dir(tmp_path: Path) -> Path:
    """Create an empty models directory; mimics the worker's bind mount."""
    d = tmp_path / "models"
    d.mkdir()
    return d


def _args(
    input_path: Path,
    output_path: Path,
    models_dir: Path,
    timeout: int = 600,
) -> list[str]:
    return [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--models",
        str(models_dir),
        "--timeout",
        str(timeout),
    ]


# ---- happy path -------------------------------------------------------------


def test_harness_happy_path(tmp_path: Path) -> None:
    """A run_one returning a dict produces a valid executor-output JSON."""
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    def run_one(unit: WorkUnit, models: Path) -> dict:
        assert unit.unit_id == "test-unit-0001"
        assert unit.payload["test_payload_value"] == 42
        assert models == models_dir
        return {"score": 0.42, "label": "drifted"}

    rc = ExecutorHarness(run_one).main(_args(input_path, output_path, models_dir))
    assert rc == 0

    # Output file exists and validates as ExecutorOutput
    written = json.loads(output_path.read_text())
    output = ExecutorOutput.model_validate(written)
    assert output.unit_id == "test-unit-0001"
    assert output.exit_code == 0
    assert output.payload == {"score": 0.42, "label": "drifted"}
    assert output.schema_version == "0.1"


def test_harness_creates_output_parent_dir(tmp_path: Path) -> None:
    """Harness creates the output's parent directory if missing."""
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "subdir/result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, models_dir)
    )
    assert rc == 0
    assert output_path.is_file()


# ---- harness/IO failures (exit 2) -------------------------------------------


def test_harness_missing_input(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(tmp_path / "nonexistent.json", output_path, models_dir)
    )
    assert rc == 2
    assert not output_path.exists()


def test_harness_malformed_json_input(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.json"
    input_path.write_text("{not valid json")
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, models_dir)
    )
    assert rc == 2
    assert not output_path.exists()


def test_harness_input_fails_workunit_schema(tmp_path: Path) -> None:
    """Valid JSON but doesn't match WorkUnit schema → exit 2."""
    input_path = tmp_path / "wrong_shape.json"
    input_path.write_text(json.dumps({"not_a_work_unit": True}))
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, models_dir)
    )
    assert rc == 2


def test_harness_missing_models_dir(tmp_path: Path) -> None:
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, tmp_path / "nonexistent_models")
    )
    assert rc == 2


def test_harness_models_dir_is_file_not_dir(tmp_path: Path) -> None:
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    not_a_dir = tmp_path / "models_is_file"
    not_a_dir.write_text("oops")

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, not_a_dir)
    )
    assert rc == 2


# ---- tenant-code failures (exit 1) ------------------------------------------


def test_harness_tenant_raises(tmp_path: Path) -> None:
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    def boom(_u: WorkUnit, _m: Path) -> dict:
        raise RuntimeError("tenant code exploded")

    rc = ExecutorHarness(boom).main(_args(input_path, output_path, models_dir))
    assert rc == 1
    assert not output_path.exists()


def test_harness_tenant_returns_non_dict(tmp_path: Path) -> None:
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: "not a dict").main(  # type: ignore[arg-type, return-value]
        _args(input_path, output_path, models_dir)
    )
    assert rc == 1
    assert not output_path.exists()


def test_harness_tenant_returns_none(tmp_path: Path) -> None:
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: None).main(  # type: ignore[arg-type, return-value]
        _args(input_path, output_path, models_dir)
    )
    assert rc == 1


# ---- atomic write -----------------------------------------------------------


def test_harness_atomic_rename_leaves_no_partial(tmp_path: Path) -> None:
    """After a successful run, no .tmp file lingers in the output directory."""
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, models_dir)
    )
    assert rc == 0

    tmp_artifacts = list(tmp_path.glob("*.tmp"))
    assert tmp_artifacts == [], f"unexpected .tmp leftovers: {tmp_artifacts}"


# ---- executor output schema cross-check -------------------------------------


def test_harness_output_matches_published_schema(tmp_path: Path) -> None:
    """The written output should validate against the published executor_output
    JSON Schema, not just the Pydantic model."""
    import jsonschema

    from auspexai_tenant.schemas import load_schema

    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"a": 1, "nested": {"b": 2}}).main(
        _args(input_path, output_path, models_dir)
    )
    assert rc == 0

    schema = load_schema("executor_output_v0_1.json")
    written = json.loads(output_path.read_text())
    jsonschema.validate(written, schema)


# ---- argparse / CLI behavior ------------------------------------------------


def test_harness_missing_required_arg(tmp_path: Path) -> None:
    """argparse exits the process on missing required args; verify behavior."""
    input_path = _stage_workunit(tmp_path)

    with pytest.raises(SystemExit):
        ExecutorHarness(lambda _u, _m: {"ok": True}).main(
            ["--input", str(input_path)]  # missing --output and --models
        )


def test_harness_accepts_and_ignores_timeout(tmp_path: Path) -> None:
    """--timeout is parsed but does not affect the harness's own behavior;
    enforcement is the worker daemon's job."""
    input_path = _stage_workunit(tmp_path)
    output_path = tmp_path / "result.json"
    models_dir = _stage_models_dir(tmp_path)

    rc = ExecutorHarness(lambda _u, _m: {"ok": True}).main(
        _args(input_path, output_path, models_dir, timeout=1)
    )
    assert rc == 0
