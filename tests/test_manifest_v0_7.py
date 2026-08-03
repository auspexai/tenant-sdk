"""Manifest v0_7: the TOTAL generation chain.

Before v0.7 the worker sent only temperature/seed/num_predict and the serving
provider's own defaults governed the rest of the sampler chain — on Ollama
0.30-0.32 that meant top_k 40, top_p 0.9 and repeat_penalty 1.1 over a 64-token
window were in force on every run ever executed, declarable in no manifest and
recorded in no evidence bundle. A `greedy` footprint therefore named a decoding
mode the backend did not perform: since llama.cpp #9897 a temperature of 0 runs
the full chain and takes the most probable token entering the temperature step,
which is the raw argmax only when the rest of the chain is neutral.

v0.7 closes that: every request carries every chain key, at declared or neutral
values; greedy pins top_k 1; the penalty knobs become declarable; and knobs are
declarable under greedy as well as sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from auspexai_tenant.experiment_config import ExperimentConfig, manifest_dict_from_config
from auspexai_tenant.manifest import (
    DECLARABLE_CHAIN_KNOBS,
    GENERATION_CHAIN_NEUTRAL,
    InferenceDeterminism,
    Manifest,
    default_generation_options,
)

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMAS = Path(__file__).parents[1] / "schemas"


def _v01() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ── chain totality ────────────────────────────────────────────────────────────


def test_every_chain_key_is_always_emitted():
    """The point of the version: no key may be left to the provider."""
    for det in (
        InferenceDeterminism(),
        InferenceDeterminism(temperature=0.8, seed=1),
        InferenceDeterminism(temperature=0.8, seed=1, top_k=40, repeat_penalty=1.1),
    ):
        options = det.effective_generation_options()
        missing = set(GENERATION_CHAIN_NEUTRAL) - set(options)
        assert not missing, f"chain key(s) left to the provider: {sorted(missing)}"


def test_greedy_pins_argmax_not_bare_temperature_zero():
    options = InferenceDeterminism().effective_generation_options()
    assert options["temperature"] == 0.0
    # top_k 1 is what actually selects argmax; temperature 0 alone does not.
    assert options["top_k"] == 1


def test_greedy_neutralises_the_provider_penalty_defaults():
    """Ollama's DefaultOptions carry repeat_penalty 1.1 / repeat_last_n 64. Those
    are the values that were silently in force on every pre-v0.7 run."""
    options = InferenceDeterminism().effective_generation_options()
    assert options["repeat_penalty"] == 1.0
    assert options["repeat_last_n"] == 0


def test_undeclared_knob_resolves_neutral_not_provider_default():
    """A sampling run that declares only a temperature must not inherit
    top_k 40 / top_p 0.9 from the serving stack."""
    options = InferenceDeterminism(temperature=0.8, seed=3).effective_generation_options()
    assert options["top_k"] == GENERATION_CHAIN_NEUTRAL["top_k"] == 0
    assert options["top_p"] == GENERATION_CHAIN_NEUTRAL["top_p"] == 1.0


def test_declared_knobs_override_neutral():
    det = InferenceDeterminism(
        temperature=0.8, seed=3, top_k=40, top_p=0.9, min_p=0.05, repeat_penalty=1.1
    )
    options = det.effective_generation_options()
    assert options["top_k"] == 40
    assert options["top_p"] == 0.9
    assert options["min_p"] == 0.05
    assert options["repeat_penalty"] == 1.1
    assert options["temperature"] == 0.8
    assert options["seed"] == 3


def test_declared_knobs_override_the_greedy_argmax_pin():
    """(a): a greedy run may deliberately study a non-neutral chain member."""
    det = InferenceDeterminism(temperature=0.0, top_k=40, repeat_penalty=1.1)
    options = det.effective_generation_options()
    assert options["top_k"] == 40
    assert options["repeat_penalty"] == 1.1
    assert options["temperature"] == 0.0


def test_default_generation_options_matches_an_empty_block():
    assert default_generation_options() == InferenceDeterminism().effective_generation_options()


def test_declarable_knobs_are_a_subset_of_the_chain():
    assert set(DECLARABLE_CHAIN_KNOBS) <= set(GENERATION_CHAIN_NEUTRAL)


def test_knob_properties_partition_correctly():
    det = InferenceDeterminism(temperature=0.8, seed=1, top_p=0.9, repeat_penalty=1.1)
    assert det.sampling_knobs == {"top_p": 0.9}
    assert det.penalty_knobs == {"repeat_penalty": 1.1}
    assert det.declared_knobs == {"top_p": 0.9, "repeat_penalty": 1.1}


# ── the pinned-seed floor is untouched ────────────────────────────────────────


def test_sampling_still_requires_a_pinned_seed():
    with pytest.raises(ValidationError, match="pinned 'seed'"):
        InferenceDeterminism(temperature=0.7, top_k=1)


# ── version gates on the DECLARABLE surface ───────────────────────────────────


def test_penalty_knobs_valid_at_0_7():
    m = _v01()
    m["schema_version"] = "0.7"
    m["inference_determinism"] = {"temperature": 0.0, "repeat_penalty": 1.1}
    Manifest.model_validate(m)


@pytest.mark.parametrize("version", ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"])
def test_penalty_knobs_rejected_below_0_7(version):
    # Greedy, so the assertion is about the VERSION gate rather than the
    # sampling/agreement coherence rule the fixture's replication would trip.
    m = _v01()
    m["schema_version"] = version
    m["inference_determinism"] = {"temperature": 0.0, "repeat_penalty": 1.1}
    with pytest.raises(ValidationError, match=r"0\.7"):
        Manifest.model_validate(m)


def test_knobs_at_greedy_valid_at_0_7():
    m = _v01()
    m["schema_version"] = "0.7"
    m["inference_determinism"] = {"temperature": 0.0, "top_k": 1}
    Manifest.model_validate(m)


@pytest.mark.parametrize("version", ["0.5", "0.6"])
def test_knobs_at_greedy_rejected_below_0_7(version):
    """An older contract keeps its older meaning rather than silently acquiring
    the new declarable surface."""
    m = _v01()
    m["schema_version"] = version
    m["inference_determinism"] = {"temperature": 0.0, "top_k": 1}
    with pytest.raises(ValidationError, match=r"0\.7"):
        Manifest.model_validate(m)


def test_knobs_under_sampling_still_valid_at_0_5():
    m = _v01()
    m["schema_version"] = "0.5"
    m["replication_factor"] = 1  # sampling + an agreement reducer is coherent only here
    m["inference_determinism"] = {"temperature": 0.8, "seed": 1, "top_k": 40}
    Manifest.model_validate(m)


# ── the published schema artifact ─────────────────────────────────────────────


def test_v0_7_schema_accepts_the_new_members():
    schema = json.loads((SCHEMAS / "manifest_v0_7.json").read_text())
    m = _v01()
    m["schema_version"] = "0.7"
    m["inference_determinism"] = {
        "temperature": 0.8,
        "seed": 1,
        "top_k": 40,
        "seed_policy": "per_round",
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
    }
    jsonschema.validate(m, schema)


def test_v0_7_schema_carries_seed_policy_which_v0_6_omitted():
    """The v0.6 model shipped seed_policy but the immutable v0.6 artifact never
    carried it, and that artifact is additionalProperties:false."""
    v0_6 = json.loads((SCHEMAS / "manifest_v0_6.json").read_text())
    v0_7 = json.loads((SCHEMAS / "manifest_v0_7.json").read_text())
    assert "seed_policy" not in v0_6["properties"]["inference_determinism"]["properties"]
    assert "seed_policy" in v0_7["properties"]["inference_determinism"]["properties"]


def test_v0_7_schema_admits_a_knob_under_greedy():
    schema = json.loads((SCHEMAS / "manifest_v0_7.json").read_text())
    m = _v01()
    m["schema_version"] = "0.7"
    m["inference_determinism"] = {"temperature": 0.0, "top_k": 1}
    jsonschema.validate(m, schema)


# ── the builder ───────────────────────────────────────────────────────────────

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


def test_build_maps_penalty_knobs():
    cfg = _cfg(
        _BASE_EXP,
        determinism={"temperature": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 0},
    )
    m = manifest_dict_from_config(cfg, package_sha256="ab" * 32, label="lab-p")
    assert m["schema_version"] == "0.7"
    assert m["inference_determinism"]["repeat_penalty"] == 1.0
    assert m["inference_determinism"]["repeat_last_n"] == 0
    Manifest.model_validate(m)
