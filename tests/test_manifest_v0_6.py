"""Manifest v0_6: the observe-only reducer + config provenance
(process_only_reducer_and_provenance_v0_6_design.md, RATIFIED 2026-07-03).

C17 `builtin_process_only`: no agreement ever claimed — sampling and
distributional designs become declarable at ANY replication (the §3c
non-agreement mode made first-class). D17 `config_provenance`: SDK-stamped
at every build (ratified Q1 — always stamp), descriptive-never-enforced."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
    manifest_dict_from_config,
)
from auspexai_tenant.manifest import Manifest

FIXTURES = Path(__file__).parent / "fixtures"


def _v01() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ── the observe-only reducer ──────────────────────────────────────────────────


def test_process_only_reducer_valid_at_0_6():
    m = _v01()
    m["schema_version"] = "0.6"
    m["reducer"] = {"kind": "builtin_process_only"}
    Manifest.model_validate(m)


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3", "0.4", "0.5"])
def test_process_only_reducer_rejected_below_0_6(version):
    m = _v01()
    m["schema_version"] = version
    m["reducer"] = {"kind": "builtin_process_only"}
    with pytest.raises(ValidationError, match=r"0\.6"):
        Manifest.model_validate(m)


def test_sampling_with_process_only_coherent_at_any_replication():
    # THE unlock: seeded sampling + observe-only at replication > 1 — the
    # design the agreement reducers rightly forbid (each replica is an
    # independent sample; no agreement will be claimed).
    m = _v01()
    m["schema_version"] = "0.6"
    m["reducer"] = {"kind": "builtin_process_only"}
    m["replication_factor"] = 5
    m["inference_determinism"] = {"temperature": 0.8, "seed": 42, "top_p": 0.9}
    Manifest.model_validate(m)


def test_process_only_reducer_takes_no_extra_fields():
    m = _v01()
    m["schema_version"] = "0.6"
    m["reducer"] = {"kind": "builtin_process_only", "tolerance_features": ["x"]}
    with pytest.raises(ValidationError):
        Manifest.model_validate(m)


# ── config provenance ─────────────────────────────────────────────────────────


def test_config_provenance_valid_at_0_6():
    m = _v01()
    m["schema_version"] = "0.6"
    m["config_provenance"] = {
        "profile": "starter",
        "resolved_config_sha256": "ab" * 32,
        "git_commit": "c" * 40,
        "git_dirty": False,
    }
    Manifest.model_validate(m)


def test_config_provenance_rejected_below_0_6():
    m = _v01()
    m["schema_version"] = "0.5"
    m["config_provenance"] = {"resolved_config_sha256": "ab" * 32}
    with pytest.raises(ValidationError, match=r"0\.6"):
        Manifest.model_validate(m)


# ── build stamping (ratified Q1: every build) ─────────────────────────────────

_BASE_EXP = {
    "tenant_id": "lab",
    "contact": "a@b.org",
    "model_id": "m",
    "research_goal": "x" * 60,
    "prompt_characteristics": "neutral probes",
}


def _cfg(**overrides) -> ExperimentConfig:
    raw = {
        "experiment": _BASE_EXP,
        "executor": {"command": ["python", "x.py"]},
        "reducer": {"kind": "builtin_hash_agreement"},
    }
    raw.update(overrides.pop("raw_extra", {}))
    return ExperimentConfig(experiment=_BASE_EXP, raw=raw, **overrides)


def test_build_always_stamps_provenance_and_floors_at_0_6():
    m = manifest_dict_from_config(_cfg(), package_sha256="ab" * 32, label="lab-p")
    assert m["schema_version"] == "0.6"
    prov = m["config_provenance"]
    assert len(prov["resolved_config_sha256"]) == 64
    assert "git_commit" not in prov  # no source_path → no git fields
    Manifest.model_validate(m)


def test_build_provenance_hash_is_deterministic_and_config_sensitive():
    a = manifest_dict_from_config(_cfg(), package_sha256="ab" * 32, label="l-a")
    b = manifest_dict_from_config(_cfg(), package_sha256="ab" * 32, label="l-b")
    assert (
        a["config_provenance"]["resolved_config_sha256"]
        == b["config_provenance"]["resolved_config_sha256"]
    )
    c = manifest_dict_from_config(
        _cfg(raw_extra={"driver": {"max_rounds": 99}}), package_sha256="ab" * 32, label="l-c"
    )
    # The driver knobs never become manifest fields — but they ARE in the hash.
    assert (
        c["config_provenance"]["resolved_config_sha256"]
        != a["config_provenance"]["resolved_config_sha256"]
    )


def test_build_provenance_records_profile():
    m = manifest_dict_from_config(
        _cfg(active_profile="starter"), package_sha256="ab" * 32, label="lab-s"
    )
    assert m["config_provenance"]["profile"] == "starter"


def test_build_provenance_captures_git_state(tmp_path: Path):
    toml = tmp_path / "experiment.toml"
    toml.write_text(
        "[experiment]\n"
        'tenant_id = "lab"\ncontact = "a@b.org"\nmodel_id = "m"\n'
        f'research_goal = "{"x" * 60}"\nprompt_characteristics = "neutral probes"\n'
        '[executor]\ncommand = ["python", "x.py"]\n'
        '[reducer]\nkind = "builtin_hash_agreement"\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path,
        check=True,
    )
    cfg = load_experiment_config(toml)
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-g")
    prov = m["config_provenance"]
    assert len(prov["git_commit"]) == 40
    assert prov["git_dirty"] is False
    (tmp_path / "scratch.txt").write_text("dirty")
    m2 = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-d")
    assert m2["config_provenance"]["git_dirty"] is True  # the honesty bit


# ── published-artifact mirror ─────────────────────────────────────────────────


def test_v0_6_json_schema_mirrors_model():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).parent.parent / "schemas" / "manifest_v0_6.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    m = _v01()
    m["schema_version"] = "0.6"
    m["reducer"] = {"kind": "builtin_process_only"}
    m["replication_factor"] = 5
    m["inference_determinism"] = {"temperature": 0.8, "seed": 42, "top_p": 0.9}
    m["config_provenance"] = {"resolved_config_sha256": "ab" * 32, "git_dirty": True}
    jsonschema.validate(m, schema)
    bad = dict(m)
    bad["config_provenance"] = {"git_dirty": True}  # missing the required hash
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
