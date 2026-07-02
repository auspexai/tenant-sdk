"""D16.2 — pre-registration (manifest v0.4).

Covers the PreRegistration model, the manifest-level checkability validation
(features reference declared feature_schema entries carrying a `comparison` —
the envelope is REFERENCED, never duplicated), the v0.4 version discipline, the
JSON Schema mirror parity, and the experiment.toml → manifest mapping + bump.
See preregistration_design.md (§3 block, §7 citable gate, Q2/Q4 resolutions).
"""

from __future__ import annotations

import jsonschema
import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import manifest_dict_from_config
from auspexai_tenant.manifest import Manifest, PreRegistration
from auspexai_tenant.schemas import load_schema
from tests.test_feature_schema import _BASE_EXP, VIGILES_FEATURE_SCHEMA, _cfg, _v01

# The Vigiles-shaped worked block (§8 exemplar shape): a DESCRIPTIVE
# stability baseline, the envelope referenced from the feature schema.
PRE_REG = {
    "hypothesis": (
        "gemma-3-1b-it-q4's response to each fixed probe is stable across "
        "rounds (no within-session drift)."
    ),
    "analysis_method": "per probe_id, compare the consensus feature vector round-over-round",
    "features": ["lexical.type_token_ratio", "lexical.top_tokens", "response_sha256"],
    "timescale": "intra_experiment_rounds",
    "decision_rule": (
        "drift IFF a probe's consensus vector moves outside the declared "
        "comparison envelope after convergence"
    ),
    "expected_result": "no probe drifts; null = >=1 probe drifts",
    "stopping_rule": "converge-on-stability (stable_rounds=3); not data-peeking-dependent",
    "comparison_keys": ["probe_id"],
}


def _v04(pre_reg: dict | None = None) -> dict:
    m = _v01()
    m["schema_version"] = "0.4"
    m["feature_schema"] = VIGILES_FEATURE_SCHEMA
    m["pre_registration"] = dict(pre_reg if pre_reg is not None else PRE_REG)
    return m


# ── the model + manifest checkability validation ─────────────────────────────


def test_v04_manifest_with_pre_registration_validates() -> None:
    parsed = Manifest.model_validate(_v04())
    pr = parsed.pre_registration
    assert pr is not None and pr.version == "0.1"
    assert pr.timescale == "intra_experiment_rounds"
    assert "lexical.type_token_ratio" in pr.features
    # The pre-registered envelope IS the referenced feature's comparison.
    assert parsed.feature_schema["lexical.type_token_ratio"].comparison is not None


def test_v04_without_member_is_structurally_prior_version() -> None:
    m = _v01()
    m["schema_version"] = "0.4"
    assert Manifest.model_validate(m).pre_registration is None


def test_pre_registration_requires_v04() -> None:
    m = _v04()
    m["schema_version"] = "0.3"  # member not in the published 0.3 set
    with pytest.raises(ValidationError, match=r"0\.4"):
        Manifest.model_validate(m)


def test_pre_registration_requires_feature_schema() -> None:
    m = _v04()
    del m["feature_schema"]
    with pytest.raises(ValidationError, match="feature_schema"):
        Manifest.model_validate(m)


def test_undeclared_feature_rejected() -> None:
    m = _v04({**PRE_REG, "features": ["not.declared"]})
    with pytest.raises(ValidationError, match="not in feature_schema"):
        Manifest.model_validate(m)


def test_feature_without_comparison_rejected() -> None:
    # eval_count is declared in the Vigiles schema but carries NO comparison —
    # a design cannot pre-register an envelope that was never declared.
    m = _v04({**PRE_REG, "features": ["eval_count"]})
    with pytest.raises(ValidationError, match="no 'comparison'"):
        Manifest.model_validate(m)


def test_undeclared_comparison_key_rejected() -> None:
    m = _v04({**PRE_REG, "comparison_keys": ["seed"]})  # seed is not a declared feature
    with pytest.raises(ValidationError, match="comparison_key"):
        Manifest.model_validate(m)


def test_stopping_rule_required() -> None:
    block = dict(PRE_REG)
    del block["stopping_rule"]  # Q4: precludes "ran until it looked good"
    with pytest.raises(ValidationError, match="stopping_rule"):
        Manifest.model_validate(_v04(block))


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        PreRegistration.model_validate({**PRE_REG, "post_hoc_note": "x"})


# ── JSON Schema mirror parity (the published v0.4 contract) ───────────────────


def test_json_schema_mirror_accepts_good_v04() -> None:
    jsonschema.validate(_v04(), load_schema("manifest_v0_4.json"))


def test_json_schema_mirror_requires_stopping_rule() -> None:
    block = dict(PRE_REG)
    del block["stopping_rule"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_v04(block), load_schema("manifest_v0_4.json"))


def test_json_schema_mirror_rejects_unknown_member() -> None:
    m = _v04()
    m["pre_registration"]["post_hoc_note"] = "x"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, load_schema("manifest_v0_4.json"))


# ── experiment.toml → manifest mapping + the version bump ─────────────────────


def test_build_maps_pre_registration_and_bumps_to_0_4() -> None:
    cfg = _cfg(_BASE_EXP, feature_schema=VIGILES_FEATURE_SCHEMA, pre_registration=dict(PRE_REG))
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-pr")
    assert m["schema_version"] == "0.4"
    assert m["pre_registration"]["timescale"] == "intra_experiment_rounds"
    Manifest.model_validate(m)  # the built 0.4 manifest is valid end-to-end


def test_build_feature_schema_alone_stays_0_3() -> None:
    cfg = _cfg(_BASE_EXP, feature_schema=VIGILES_FEATURE_SCHEMA)
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-fs")
    assert m["schema_version"] == "0.3"
    assert "pre_registration" not in m
