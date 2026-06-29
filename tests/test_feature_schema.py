"""D16.1 — the self-describing feature schema (manifest v0.3).

Covers the FeatureDeclaration model + per-kind §7 bounds, the v0.3 Manifest
member, the JSON Schema mirror parity, the experiment.toml → manifest mapping +
version bump, and the full Vigiles 11-feature exemplar (the §8 worked reference,
proven here so Inc 1 need not touch the live experiment.toml — that moves with
the coordinator's v0.3 acceptance in Inc 2)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import ExperimentConfig, manifest_dict_from_config
from auspexai_tenant.manifest import FeatureDeclaration, Manifest
from auspexai_tenant.schemas import load_schema

FIXTURES = Path(__file__).parent / "fixtures"


def _v01() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ── The Vigiles exemplar: the §8 worked reference for all 11 emitted features ──
VIGILES_FEATURE_SCHEMA: dict[str, dict] = {
    "schema": {
        "meaning": "result schema identifier",
        "kind": "categorical",
        "role": "provenance",
        "change_means": "the executor's result schema version changed",
        "categories": ["vigiles-drift-probe/v0"],
    },
    "probe_id": {
        "meaning": "which fixed probe produced this row",
        "kind": "categorical",
        "role": "key",
        "change_means": "a different probe — the primary drift/comparison join coordinate",
        "categories": ["p-greeting", "p-instruction", "p-refusal"],
    },
    "response_sha256": {
        "meaning": "SHA-256 of the raw model output — the authoritative byte-level drift anchor",
        "kind": "hash",
        "role": "anchor",
        "algorithm": "sha256",
        "change_means": "ANY change = output bytes differed, incl. reordering/whitespace the lexical features cannot see",
        "comparison": {"rule": "exact"},
    },
    "response_chars": {
        "meaning": "length of the raw response in characters",
        "kind": "count",
        "role": "summary",
        "unit": "characters",
        "range": {"min": 0},
        "change_means": "the output got longer or shorter",
    },
    "eval_count": {
        "meaning": "backend-reported completion tokens generated",
        "kind": "count",
        "role": "summary",
        "unit": "tokens",
        "range": {"min": 0},
        "change_means": "the model emitted a different number of completion tokens",
    },
    "lexical.tokens": {
        "meaning": "whitespace-token count of the response",
        "kind": "count",
        "role": "summary",
        "unit": "whitespace_tokens",
        "range": {"min": 0},
        "change_means": "the response contains more or fewer whitespace tokens",
    },
    "lexical.unique_tokens": {
        "meaning": "distinct whitespace tokens in the response",
        "kind": "count",
        "role": "summary",
        "unit": "whitespace_tokens",
        "range": {"min": 0},
        "change_means": "the response's distinct-token count changed",
    },
    "lexical.type_token_ratio": {
        "meaning": "unique tokens / total tokens — lexical diversity",
        "kind": "numeric",
        "role": "summary",
        "unit": "ratio",
        "range": {"min": 0.0, "max": 1.0},
        "valid_when": {"field": "lexical.tokens", "op": ">=", "value": 5},
        "invariant_to": ["token_order", "whitespace", "punctuation"],
        "change_means": "vocabulary richness shifted; does NOT capture reordering or formatting drift",
        "comparison": {"rule": "numeric", "rel": 0.05},
    },
    "lexical.top_tokens": {
        "meaning": "the top-8 tokens by count",
        "kind": "set",
        "role": "summary",
        "element_kind": "categorical",
        "max_cardinality": 8,
        "invariant_to": ["token_order"],
        "change_means": "the most-frequent vocabulary changed; §7 boundary — carries token fragments (certify-gate review)",
        "comparison": {"rule": "set_jaccard", "min": 0.9},
    },
    "model.id": {
        "meaning": "the model id the worker served",
        "kind": "categorical",
        "role": "provenance",
        "categories": ["gemma-3-1b-it-q4"],
        "change_means": "a different model produced this row — stratify or exclude",
    },
    "model.gguf_sha256": {
        "meaning": "SHA-256 of the served GGUF weights",
        "kind": "hash",
        "role": "provenance",
        "algorithm": "sha256",
        "change_means": "the served weights differ — a provenance mismatch",
    },
}


# ── FeatureDeclaration model: the valid kinds ─────────────────────────────────


@pytest.mark.parametrize("decl", VIGILES_FEATURE_SCHEMA.values(), ids=VIGILES_FEATURE_SCHEMA.keys())
def test_each_vigiles_feature_declaration_valid(decl: dict) -> None:
    FeatureDeclaration.model_validate(decl)


# ── FeatureDeclaration model: per-kind §7 bounds enforced ─────────────────────


def _decl(**over) -> dict:
    base = {"meaning": "m", "kind": "count", "role": "summary", "change_means": "c"}
    base.update(over)
    return base


def test_numeric_requires_range() -> None:
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="numeric"))


def test_categorical_requires_closed_categories() -> None:
    # The §7 no-free-text guarantee: a categorical with no closed vocabulary is rejected.
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="categorical"))


def test_hash_requires_algorithm() -> None:
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="hash"))


def test_set_requires_element_kind_and_cardinality() -> None:
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="set", element_kind="categorical"))


def test_no_free_text_kind() -> None:
    # There is deliberately no `text`/free-string kind — the structural §7 guarantee.
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="text"))


def test_cross_kind_bounds_rejected() -> None:
    # A bound that does not belong to the kind is nonsense and rejected.
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="hash", algorithm="sha256", range={"min": 0}))
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(
            _decl(kind="numeric", range={"min": 0, "max": 1}, categories=["x"])
        )


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(bogus=1))


def test_range_max_below_min_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureDeclaration.model_validate(_decl(kind="numeric", range={"min": 1.0, "max": 0.0}))


# ── valid_when (structured) + comparison (the C7 envelope) ────────────────────


def test_valid_when_is_structured_not_freetext() -> None:
    FeatureDeclaration.model_validate(
        _decl(
            kind="numeric",
            range={"min": 0, "max": 1},
            valid_when={"field": "n", "op": ">=", "value": 5},
        )
    )
    with pytest.raises(ValidationError):  # a bad op is rejected
        FeatureDeclaration.model_validate(
            _decl(
                kind="numeric",
                range={"min": 0, "max": 1},
                valid_when={"field": "n", "op": "~", "value": 5},
            )
        )


def test_comparison_rule_fields_validated() -> None:
    ok = _decl(kind="numeric", range={"min": 0, "max": 1})
    FeatureDeclaration.model_validate({**ok, "comparison": {"rule": "numeric", "rel": 0.05}})
    FeatureDeclaration.model_validate({**ok, "comparison": {"rule": "exact"}})
    with pytest.raises(ValidationError):  # numeric needs rel or abs
        FeatureDeclaration.model_validate({**ok, "comparison": {"rule": "numeric"}})
    with pytest.raises(ValidationError):  # set_jaccard needs min in [0,1]
        FeatureDeclaration.model_validate({**ok, "comparison": {"rule": "set_jaccard", "min": 1.5}})
    with pytest.raises(ValidationError):  # exact takes no tolerance fields
        FeatureDeclaration.model_validate({**ok, "comparison": {"rule": "exact", "rel": 0.1}})


# ── Manifest v0.3 integration ─────────────────────────────────────────────────


def test_v03_manifest_with_feature_schema_validates() -> None:
    m = _v01()
    m["schema_version"] = "0.3"
    m["feature_schema"] = VIGILES_FEATURE_SCHEMA
    parsed = Manifest.model_validate(m)
    assert parsed.feature_schema is not None
    assert parsed.feature_schema["response_sha256"].role == "anchor"
    assert parsed.feature_schema["lexical.type_token_ratio"].comparison.rel == 0.05


def test_v03_without_member_is_structurally_prior_version() -> None:
    m = _v01()
    m["schema_version"] = "0.3"  # a 0.3 manifest declaring no feature_schema is valid
    assert Manifest.model_validate(m).feature_schema is None


def test_v01_v02_still_valid() -> None:
    Manifest.model_validate(_v01())  # 0.1 untouched


def test_bad_nested_feature_declaration_fails_manifest() -> None:
    m = _v01()
    m["schema_version"] = "0.3"
    m["feature_schema"] = {
        "x": {"meaning": "m", "kind": "hash", "role": "anchor", "change_means": "c"}
    }
    with pytest.raises(ValidationError):  # hash without algorithm
        Manifest.model_validate(m)


# ── JSON Schema mirror parity ─────────────────────────────────────────────────


def test_json_schema_mirror_accepts_good_v03() -> None:
    m = _v01()
    m["schema_version"] = "0.3"
    m["feature_schema"] = VIGILES_FEATURE_SCHEMA
    jsonschema.validate(m, load_schema("manifest_v0_3.json"))


def test_json_schema_mirror_rejects_bad_kind() -> None:
    m = _v01()
    m["schema_version"] = "0.3"
    m["feature_schema"] = {
        "x": {"meaning": "m", "kind": "text", "role": "summary", "change_means": "c"}
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, load_schema("manifest_v0_3.json"))


# ── experiment.toml → manifest mapping + schema_version bump ──────────────────

_BASE_EXP = {
    "tenant_id": "lab",
    "contact": "a@b.org",
    "model_id": "m",
    "research_goal": "x" * 60,
    "prompt_characteristics": "neutral probes",
}


def _cfg(experiment: dict, **tables) -> ExperimentConfig:
    raw = {
        "experiment": experiment,
        "executor": {"command": ["python", "x.py"]},
        "reducer": {"kind": "builtin_hash_agreement"},
        **tables,
    }
    return ExperimentConfig(experiment=experiment, raw=raw)


def test_build_maps_feature_schema_and_bumps_to_0_3() -> None:
    cfg = _cfg(_BASE_EXP, feature_schema=VIGILES_FEATURE_SCHEMA)
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-z")
    assert m["schema_version"] == "0.3"
    assert m["feature_schema"]["response_sha256"]["role"] == "anchor"
    Manifest.model_validate(m)  # the built 0.3 manifest is valid end-to-end


def test_build_no_feature_schema_does_not_bump_to_0_3() -> None:
    m = manifest_dict_from_config(_cfg(_BASE_EXP), package_sha256="ab" * 32, label="lab-x")
    assert m["schema_version"] == "0.1"
    assert "feature_schema" not in m
