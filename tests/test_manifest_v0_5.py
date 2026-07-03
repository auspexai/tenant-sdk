"""Manifest v0_5: the seeded-sampling generation policy
(inference_determinism_scoping_memo.md, RATIFIED 2026-07-02).

v0.5 adds NO new member — it extends inference_determinism with the ratified
whitelist knobs (top_p/top_k/min_p) and makes the pinned-seed floor structural:
temperature > 0 requires seed; a knob requires temperature > 0; knobs require
schema_version "0.5" (the published v0.2-v0.4 artifacts are immutable)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import ExperimentConfig, manifest_dict_from_config
from auspexai_tenant.manifest import InferenceDeterminism, Manifest

FIXTURES = Path(__file__).parent / "fixtures"


def _v01() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ── InferenceDeterminism: modes + the pinned-seed floor ──────────────────────


def test_greedy_default_unchanged():
    det = InferenceDeterminism()
    assert det.temperature == 0.0
    assert not det.is_sampling
    assert det.sampling_knobs == {}


def test_sampling_requires_seed():
    with pytest.raises(ValidationError, match="pinned 'seed'"):
        InferenceDeterminism(temperature=0.7)


def test_sampling_with_seed_valid():
    det = InferenceDeterminism(temperature=0.7, seed=42, top_p=0.9, top_k=40, min_p=0.05)
    assert det.is_sampling
    assert det.sampling_knobs == {"top_p": 0.9, "top_k": 40, "min_p": 0.05}


def test_knobs_require_sampling():
    with pytest.raises(ValidationError, match="require temperature > 0"):
        InferenceDeterminism(temperature=0.0, top_p=0.9)


def test_negative_temperature_rejected():
    with pytest.raises(ValidationError):
        InferenceDeterminism(temperature=-0.1)


@pytest.mark.parametrize(
    "knob,bad",
    [("top_p", 0.0), ("top_p", 1.5), ("top_k", 0), ("min_p", 1.0), ("min_p", -0.1)],
)
def test_knob_bounds(knob, bad):
    with pytest.raises(ValidationError):
        InferenceDeterminism(temperature=0.7, seed=1, **{knob: bad})


# ── Manifest: the v0.5 version gate + back-compat ────────────────────────────


def _sampling_manifest(version: str = "0.5", **det) -> dict:
    m = _v01()
    m["schema_version"] = version
    m["inference_determinism"] = {"temperature": 0.7, "seed": 42, **det}
    # §3c coherence: a sampling manifest needs a non-agreement collection mode.
    m["reducer"] = {"kind": "custom", "command": ["python", "fold.py"]}
    return m


def test_v05_sampling_manifest_valid():
    Manifest.model_validate(_sampling_manifest(top_p=0.9))


def test_v05_no_knobs_valid():
    # A v0.5 manifest declaring no sampling knob is structurally a v0.4 manifest.
    m = _v01()
    m["schema_version"] = "0.5"
    Manifest.model_validate(m)


def test_knobs_below_v05_rejected():
    # The published v0.2-v0.4 schema artifacts are immutable — knobs are v0.5-only.
    for version in ("0.2", "0.3", "0.4"):
        with pytest.raises(ValidationError, match=r"0\.5"):
            Manifest.model_validate(_sampling_manifest(version, top_p=0.9))


def test_sampling_without_knobs_stays_valid_at_v02():
    # temperature/seed alone are the ORIGINAL v0.2 M1 shape — no v0.5 required.
    Manifest.model_validate(_sampling_manifest("0.2"))


@pytest.mark.parametrize("kind", ["builtin_hash_agreement", "builtin_within_cell_tolerance"])
def test_sampling_incoherent_with_agreement_reducer(kind):
    # The §3c coherence gate, mirrored at build (also enforced at coordinator submit).
    m = _sampling_manifest()
    m["reducer"] = {"kind": kind}
    with pytest.raises(ValidationError, match="incoherent with the agreement"):
        Manifest.model_validate(m)


def test_v05_json_schema_mirrors_model():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).parent.parent / "schemas" / "manifest_v0_5.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    good = _sampling_manifest(top_p=0.9)
    jsonschema.validate(good, schema)
    unseeded = _v01()
    unseeded["schema_version"] = "0.5"
    unseeded["inference_determinism"] = {"temperature": 0.7}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(unseeded, schema)


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


def test_build_maps_sampling_knobs_and_bumps_to_0_5():
    cfg = _cfg(
        _BASE_EXP,
        determinism={"temperature": 0.7, "seed": 7, "top_p": 0.9, "top_k": 40},
        reducer={"kind": "custom", "command": ["python", "fold.py"]},
    )
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-s")
    assert m["schema_version"] == "0.5"
    assert m["inference_determinism"] == {
        "temperature": 0.7,
        "seed": 7,
        "top_p": 0.9,
        "top_k": 40,
    }
    Manifest.model_validate(m)


def test_build_greedy_determinism_stays_0_2():
    cfg = _cfg(_BASE_EXP, determinism={"temperature": 0.0, "seed": 7})
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-g")
    assert m["schema_version"] == "0.2"
    Manifest.model_validate(m)
