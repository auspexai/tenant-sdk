"""§5.3 / §9 #30-33 forcing-function tests: the archetype-C synthetic tenant.

Complements ``test_synthetic_tenant.py`` (the integer doubler = archetype A,
"trivial stateless"). ``synth-geometry`` is archetype C — static weights-only
geometric analysis: no runtime input, the model is the subject, the output is a
**floating-point** metric, and consensus rides the #33 determinism contract.

If these tests fail, the SDK has either grown a JSON/integer-shaped assumption or
the determinism contract has regressed — the archetype-C diversity alarm. See
``Documentation/AuspexAI/v0.1.0/{genericity_pass_ratification,archetype_c_tenant_design}.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from auspexai_tenant.manifest import Manifest

GEOM_DIR = Path(__file__).parent.parent / "examples" / "synth_geometry"
METRIC_KEY = "mean_abs_cosine_separation"


def _write_model(models_dir: Path) -> None:
    """A small, fixed weight matrix — deterministic, no numpy needed."""
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        [0.10, -0.40, 0.32, 0.05],
        [-0.22, 0.18, 0.91, -0.13],
        [0.55, 0.20, -0.07, 0.44],
        [-0.61, 0.33, 0.12, 0.28],
        [0.04, -0.88, 0.21, 0.66],
        [0.39, 0.41, -0.50, -0.09],
    ]
    (models_dir / "W.json").write_text(json.dumps(rows, separators=(",", ":")))


def _unit(seed: int, *, unit_id: str = "g0001", n_samples: int = 500) -> dict:
    return {
        "schema_version": "0.1",
        "unit_id": unit_id,
        "tenant_id": "synth-geometry",
        "experiment_id": "synth-geometry-v1",
        "manifest_sha256": "0" * 64,
        "created_at": "2026-06-01T20:00:00Z",
        "payload": {"seed": seed, "n_samples": n_samples, "probe": METRIC_KEY},
    }


def _run_executor(unit: dict, models_dir: Path, tmp_path: Path, tag: str) -> dict:
    input_path = tmp_path / f"unit-{tag}.json"
    output_path = tmp_path / f"result-{tag}.json"
    input_path.write_text(json.dumps(unit))
    proc = subprocess.run(
        [
            sys.executable,
            str(GEOM_DIR / "executor.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--models",
            str(models_dir),
            "--timeout",
            "60",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"executor failed: {proc.stderr or proc.stdout}"
    return json.loads(output_path.read_text())


# ---- Manifest validates + is archetype-C-shaped -----------------------------


def test_geometry_manifest_validates() -> None:
    raw = json.loads((GEOM_DIR / "manifest.json").read_text())
    m = Manifest.model_validate(raw)
    assert m.tenant_id == "synth-geometry"
    assert m.experiment_id == "synth-geometry-v1"
    assert m.sensitive_content_flags == []


def test_geometry_manifest_is_archetype_c_shaped() -> None:
    """Archetype-C guard: weights-only, no prompts, declares local weights.

    This is the inverse of the doubler's guard — where the doubler asserts
    ``local_weights_required is False`` (no model), archetype C asserts it is
    True (the model is the subject). If this flips, the example stopped
    exercising the capability-declaration path (#30a).
    """
    raw = json.loads((GEOM_DIR / "manifest.json").read_text())
    assert raw["models"][0]["local_weights_required"] is True
    assert "no prompts" in raw["prompt_set_characteristics"].lower()
    assert raw["reducer"]["kind"] == "builtin_hash_agreement"


# ---- Executor runs + produces a float metric --------------------------------


def test_geometry_executor_runs_end_to_end(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    env = _run_executor(_unit(seed=1234), models_dir, tmp_path, "a")

    assert env["exit_code"] == 0
    payload = env["payload"]
    assert isinstance(payload[METRIC_KEY], float)
    assert 0.0 <= payload[METRIC_KEY] <= 1.0  # |cosine| is in [0, 1]
    assert payload["seed"] == 1234


def test_geometry_output_is_canonically_quantized(tmp_path: Path) -> None:
    """The metric must be rounded to the contract's 6 places (#33)."""
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    metric = _run_executor(_unit(seed=7), models_dir, tmp_path, "q")["payload"][METRIC_KEY]
    assert round(metric, 6) == metric


# ---- The keystone: replicas agree under the determinism contract ------------


def test_geometry_replicas_are_byte_identical(tmp_path: Path) -> None:
    """Two replicas of the same seeded unit produce byte-identical
    {exit_code, payload} — i.e. they pass the coordinator's exact-hash
    ``hash_agreement`` with no coordinator change. This IS the #33 claim."""
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    unit = _unit(seed=99, n_samples=800)

    a = _run_executor(unit, models_dir, tmp_path, "r0")
    b = _run_executor(unit, models_dir, tmp_path, "r1")

    def _hashed(env: dict) -> str:
        # Mirror the coordinator's semantic_hash input (NOT completed_at).
        return json.dumps(
            {"exit_code": env["exit_code"], "payload": env["payload"]},
            sort_keys=True,
            separators=(",", ":"),
        )

    assert _hashed(a) == _hashed(b)


def test_geometry_different_seeds_differ(tmp_path: Path) -> None:
    """Sanity: different seeds explore different estimates (the metric is not
    a constant), so the experiment-level reduce has something to aggregate."""
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    m1 = _run_executor(_unit(seed=1), models_dir, tmp_path, "s1")["payload"][METRIC_KEY]
    m2 = _run_executor(_unit(seed=2), models_dir, tmp_path, "s2")["payload"][METRIC_KEY]
    assert m1 != m2


# ---- Experiment-level cross-unit reduce (#34 Phase-1) -----------------------


def test_geometry_experiment_reduce() -> None:
    sys.path.insert(0, str(GEOM_DIR))
    try:
        from reduce_experiment import reduce_metrics
    finally:
        sys.path.pop(0)

    fp = reduce_metrics([0.141, 0.138, 0.143, 0.140, 0.142, 0.139], bins=4)
    assert fp["unit_count"] == 6
    assert fp["min"] <= fp["grand_mean"] <= fp["max"]
    assert sum(fp["histogram"]) == 6
