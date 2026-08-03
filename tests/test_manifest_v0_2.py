"""Manifest v0_2 bundle (M1-M4): inference_determinism, requires_real_execution,
served-weights digest (Model.expected_gguf_sha256), declared output schema.

All optional; schema_version bumps to 0.2 exactly when a member is declared, so a
0.1 manifest stays valid forever (re-verify-forever, no forced migration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import ExperimentConfig, manifest_dict_from_config
from auspexai_tenant.manifest import Manifest

FIXTURES = Path(__file__).parent / "fixtures"


def _v01() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ── model: back-compat + the new members ────────────────────────────────────


def test_v01_still_valid():
    Manifest.model_validate(_v01())  # 0.1 manifests keep validating, untouched


def test_v02_accepts_all_members():
    m = _v01()
    m["schema_version"] = "0.2"
    m["models"][0]["expected_gguf_sha256"] = "ab" * 32
    m["requires_real_execution"] = True
    m["inference_determinism"] = {
        "temperature": 0.0,
        "seed": 42,
        "serving_version_pin": "ollama/0.17.7",
    }
    m["output_schema"] = {"measurement_level": "ratio", "shape": [], "dtype": "float32"}
    Manifest.model_validate(m)


def test_v02_members_optional():
    m = _v01()
    m["schema_version"] = "0.2"  # a 0.2 manifest declaring NO members is valid
    Manifest.model_validate(m)


def test_unknown_schema_version_rejected():
    m = _v01()
    m["schema_version"] = "9.9"
    with pytest.raises(ValidationError):
        Manifest.model_validate(m)


def test_bad_gguf_digest_rejected():
    m = _v01()
    m["models"][0]["expected_gguf_sha256"] = "not-hex"
    with pytest.raises(ValidationError):
        Manifest.model_validate(m)


def test_bad_measurement_level_rejected():
    m = _v01()
    m["output_schema"] = {"measurement_level": "made_up"}
    with pytest.raises(ValidationError):
        Manifest.model_validate(m)


def test_determinism_extra_forbidden():
    m = _v01()
    m["inference_determinism"] = {"temperature": 0.0, "bogus": 1}
    with pytest.raises(ValidationError):
        Manifest.model_validate(m)


# ── experiment.toml → manifest mapping + schema_version bump ─────────────────


def _cfg(experiment: dict, **tables) -> ExperimentConfig:
    raw = {
        "experiment": experiment,
        "executor": {"command": ["python", "x.py"]},
        "reducer": {"kind": "builtin_hash_agreement"},
        **tables,
    }
    return ExperimentConfig(experiment=experiment, raw=raw)


_BASE_EXP = {
    "tenant_id": "lab",
    "contact": "a@b.org",
    "model_id": "m",
    "research_goal": "x" * 60,
    "prompt_characteristics": "neutral probes",
}


def test_build_no_members_still_stamps_provenance():
    # D17 (ratified 2026-07-03): EVERY build stamps config_provenance, so a
    # built manifest floors at 0.6 even with no other members declared.
    m = manifest_dict_from_config(_cfg(_BASE_EXP), package_sha256="ab" * 32, label="lab-x")
    assert m["schema_version"] == "0.7"
    assert "resolved_config_sha256" in m["config_provenance"]
    assert "inference_determinism" not in m
    Manifest.model_validate(m)


def test_build_maps_v0_2_members_and_bumps():
    cfg = _cfg(
        {**_BASE_EXP, "expected_gguf_sha256": "cd" * 32, "requires_real_execution": True},
        determinism={"temperature": 0.0, "seed": 7, "serving_version_pin": "ollama/0.17.7"},
        output={"measurement_level": "interval", "shape": [3], "dtype": "float32"},
    )
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-y")
    assert m["schema_version"] == "0.7"
    assert m["models"][0]["expected_gguf_sha256"] == "cd" * 32
    assert m["requires_real_execution"] is True
    assert m["inference_determinism"]["seed"] == 7
    assert m["output_schema"]["measurement_level"] == "interval"
    Manifest.model_validate(m)
