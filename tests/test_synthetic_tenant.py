"""§5.3 forcing function tests: the synthetic test tenant.

These tests exercise the entire SDK contract surface against a deliberately
non-LLM-shaped tenant (the integer doubler at examples/synthetic_tenant/).
If these tests start failing, the SDK has grown a Sentinel-shaped assumption
somewhere — the §5.3 rule-of-three-inverse alarm.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from auspexai_tenant.manifest import Manifest
from auspexai_tenant.workunits import ExecutorOutput, WorkUnit, tar_reader, tar_writer

SYNTH_DIR = Path(__file__).parent.parent / "examples" / "synthetic_tenant"


# ---- Manifest validates ------------------------------------------------------


def test_synthetic_manifest_validates() -> None:
    raw = json.loads((SYNTH_DIR / "manifest.json").read_text())
    m = Manifest.model_validate(raw)
    assert m.tenant_id == "synth-doubler"
    assert m.experiment_id == "synth-doubler-v1"
    # Non-sensitive content — no approver attestations required
    assert m.sensitive_content_flags == []
    assert m.approver_attestations is None


def test_synthetic_manifest_is_not_llm_shaped() -> None:
    """§5.3 guard: the synthetic tenant should not look like Sentinel.
    If anyone accidentally adds LLM-shaped fields to the synthetic manifest,
    this test catches it before the SDK grows around those assumptions.
    """
    raw = json.loads((SYNTH_DIR / "manifest.json").read_text())
    research_goal = raw["research_goal_paragraph"].lower()
    prompt_chars = raw["prompt_set_characteristics"].lower()
    # The synthetic tenant deliberately states it has no prompts and no model
    assert "no prompts" in prompt_chars
    # Tenant doesn't have a real model name — the placeholder is intentional
    assert raw["models"][0]["local_weights_required"] is False
    # Research goal explicitly notes the §5.3 rationale
    assert "5.3" in research_goal or "non-llm" in research_goal


# ---- Executor runs as a subprocess ------------------------------------------


def _make_unit(value: int, unit_id: str = "u0001") -> dict:
    return {
        "schema_version": "0.1",
        "unit_id": unit_id,
        "tenant_id": "synth-doubler",
        "experiment_id": "synth-doubler-v1",
        "manifest_sha256": "0" * 64,
        "created_at": "2026-05-17T20:00:00Z",
        "payload": {"value": value},
    }


def test_synthetic_executor_runs_end_to_end(tmp_path: Path) -> None:
    """Invoke executor.py as a subprocess against a crafted work unit and
    verify the output structure — exactly what the worker daemon will do."""
    input_path = tmp_path / "unit.json"
    output_path = tmp_path / "result.json"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    input_path.write_text(json.dumps(_make_unit(value=21)))

    result = subprocess.run(
        [
            sys.executable,
            str(SYNTH_DIR / "executor.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--models",
            str(models_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    output = ExecutorOutput.model_validate(json.loads(output_path.read_text()))
    assert output.unit_id == "u0001"
    assert output.exit_code == 0
    assert output.payload == {"doubled": 42, "input": 21}


def test_synthetic_executor_handles_bad_payload(tmp_path: Path) -> None:
    """The synthetic executor raises on non-int payload — verify it surfaces
    as a tenant-code failure (exit 1) rather than a harness failure (exit 2)."""
    input_path = tmp_path / "unit.json"
    output_path = tmp_path / "result.json"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bad_unit = _make_unit(value=21)
    bad_unit["payload"] = {"value": "not an int"}
    input_path.write_text(json.dumps(bad_unit))

    result = subprocess.run(
        [
            sys.executable,
            str(SYNTH_DIR / "executor.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--models",
            str(models_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, "expected tenant-code failure (exit 1)"
    assert "TypeError" in result.stderr


# ---- Full SDK contract: write tarball, exercise executor on each unit -----


def test_synthetic_tenant_full_contract_round_trip(tmp_path: Path) -> None:
    """Exercise the full SDK surface: build work units → tar_writer →
    tar_reader → executor subprocess on each unit → verify outputs.

    If this test passes, the SDK has roundtripped a complete tenant's data
    through every published contract."""
    # Build a set of WorkUnits
    now = datetime.now(UTC)
    units = [
        WorkUnit(
            schema_version="0.1",
            unit_id=f"u{i:04d}",
            tenant_id="synth-doubler",
            experiment_id="synth-doubler-v1",
            manifest_sha256="0" * 64,
            created_at=now,
            payload={"value": i},
        )
        for i in range(1, 6)
    ]

    # Pack to tarball
    tarball_path = tmp_path / "work_units.tar.gz"
    count = tar_writer(units, tarball_path)
    assert count == 5

    # Compute the tarball_sha256 the tenant would put in the manifest
    sha = hashlib.sha256(tarball_path.read_bytes()).hexdigest()
    assert len(sha) == 64

    # Unpack and run executor on each
    unpacked = tar_reader(tarball_path)
    assert len(unpacked) == 5

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for unit in unpacked:
        input_path = tmp_path / f"input_{unit.unit_id}.json"
        output_path = tmp_path / f"output_{unit.unit_id}.json"
        input_path.write_text(unit.model_dump_json())

        result = subprocess.run(
            [
                sys.executable,
                str(SYNTH_DIR / "executor.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--models",
                str(models_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"unit {unit.unit_id}: {result.stderr}"
        output = ExecutorOutput.model_validate(json.loads(output_path.read_text()))
        expected_value = unit.payload["value"]
        assert output.payload == {"doubled": expected_value * 2, "input": expected_value}
