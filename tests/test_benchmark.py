"""The Drift Benchmark (D16.4) — envelope-units math, aggregation honesty,
the byte overlay, valid_when exclusion, and the bundle adapter.

The semantics under test mirror the coordinator reducer exactly where they
overlap (numeric tolerance = max(abs, rel*|reference|); jaccard over full
hashable elements) — the benchmark is another READER of the same declared
envelope, never a second definition of it."""

from __future__ import annotations

from typing import ClassVar

from auspexai_tenant.benchmark import (
    DriftBenchmark,
    drift_benchmark,
    drift_benchmark_bundles,
    envelope_units,
    format_report,
)

SCHEMA = {
    "probe_id": {
        "meaning": "which probe",
        "kind": "categorical",
        "role": "key",
        "change_means": "different probe",
        "categories": ["p-a", "p-b"],
    },
    "response_sha256": {
        "meaning": "byte anchor",
        "kind": "hash",
        "role": "anchor",
        "algorithm": "sha256",
        "change_means": "bytes differed",
        "comparison": {"rule": "exact"},
    },
    "lexical.type_token_ratio": {
        "meaning": "diversity",
        "kind": "numeric",
        "role": "summary",
        "range": {"min": 0.0, "max": 1.0},
        "change_means": "vocab shift",
        "comparison": {"rule": "numeric", "rel": 0.02},
        "valid_when": {"field": "lexical.tokens", "op": ">=", "value": 5},
    },
    "lexical.top_tokens": {
        "meaning": "top tokens",
        "kind": "set",
        "role": "summary",
        "element_kind": "categorical",
        "max_cardinality": 8,
        "change_means": "vocab changed",
        "comparison": {"rule": "set_jaccard", "min": 0.9},
    },
}


def _obs(probe, sha, ttr, tokens, top):
    return {
        "probe_id": probe,
        "response_sha256": sha,
        "lexical": {"type_token_ratio": ttr, "tokens": tokens, "top_tokens": top},
    }


# ── envelope_units math ───────────────────────────────────────────────────────


def test_numeric_rel_eu_matches_reducer_semantics():
    # tol = rel * |reference|; EU = |v - ref| / tol
    cmp_ = {"rule": "numeric", "rel": 0.02}
    assert envelope_units(cmp_, 0.9615, 0.9474) == (0.9615 - 0.9474) / (0.02 * 0.9474)


def test_numeric_abs_and_rel_take_the_larger_tolerance():
    cmp_ = {"rule": "numeric", "rel": 0.01, "abs": 0.5}
    # ref=10 → rel tol 0.1 < abs 0.5 → tol 0.5; delta 1.0 → 2 EU
    assert envelope_units(cmp_, 11.0, 10.0) == 2.0


def test_numeric_zero_tolerance_is_undefined_not_infinite():
    assert envelope_units({"rule": "numeric", "rel": 0.02}, 1.0, 0.0) is None


def test_set_jaccard_eu():
    cmp_ = {"rule": "set_jaccard", "min": 0.9}
    a = [["x", 1], ["y", 2]]
    b = [["x", 1], ["z", 2]]  # jaccard 1/3 over full (token,count) elements
    eu = envelope_units(cmp_, a, b)
    assert abs(eu - ((1 - 1 / 3) / 0.1)) < 1e-9


def test_set_elements_compare_as_full_pairs_like_the_reducer():
    cmp_ = {"rule": "set_jaccard", "min": 0.9}
    # Same tokens, different counts → different ELEMENTS (the reducer convention).
    assert envelope_units(cmp_, [["x", 1]], [["x", 2]]) > 0


def test_binary_rules_never_yield_eu():
    assert envelope_units({"rule": "exact"}, "a", "b") is None
    assert envelope_units({"rule": "categorical_exact"}, "a", "b") is None


# ── the benchmark ─────────────────────────────────────────────────────────────


def test_within_noise_scores_under_one():
    ref = [_obs("p-a", "h1", 0.900, 20, [["x", 1], ["y", 1]])]
    obs = [_obs("p-a", "h1", 0.905, 20, [["x", 1], ["y", 1]])]
    r = drift_benchmark(obs, ref, SCHEMA)
    assert r.peak_eu is not None and r.peak_eu < 1.0
    assert r.breadth == 0.0
    assert r.byte_divergence_rate == 0.0


def test_drift_beyond_envelope_flags_probe_and_breadth():
    ref = [_obs("p-a", "h1", 0.900, 20, [["x", 1], ["y", 1]])]
    obs = [_obs("p-a", "h2", 0.950, 20, [["q", 1], ["r", 1]])]  # jaccard 0 → 10 EU
    r = drift_benchmark(obs, ref, SCHEMA)
    assert r.peak_eu >= 1.0
    assert r.breadth == 1.0
    assert r.byte_divergence_rate == 1.0
    assert r.probes[0].beyond_envelope


def test_byte_divergence_never_inflates_the_scalar():
    # Identical features, different bytes (the constrained-probe case, §5):
    # the scalar stays ~0, the overlay reads 100%.
    ref = [_obs("p-a", "h1", 0.900, 20, [["x", 1]])]
    obs = [_obs("p-a", "DIFFERENT", 0.900, 20, [["x", 1]])]
    r = drift_benchmark(obs, ref, SCHEMA)
    assert r.peak_eu == 0.0
    assert r.byte_divergence_rate == 1.0


def test_valid_when_excludes_degenerate_observations():
    # tokens < 5 on one side → TTR pairs excluded (and counted), set still scored.
    ref = [_obs("p-a", "h1", 1.0, 3, [["x", 1]])]
    obs = [_obs("p-a", "h1", 0.5, 20, [["x", 1]])]
    r = drift_benchmark(obs, ref, SCHEMA)
    ttr = next(f for f in r.probes[0].features if f.feature == "lexical.type_token_ratio")
    assert ttr.eu is None and ttr.pairs == 0 and ttr.invalid_excluded == 1


def test_median_over_cross_pairs_is_robust_to_one_outlier_round():
    # 3 observation rounds, one wildly off (the 0.17.7 leak shape): the median
    # pair keeps the score honest while eu_max preserves the spread.
    ref = [_obs("p-a", "h1", 0.900, 20, [["x", 1]])]
    obs = [
        _obs("p-a", "h1", 0.902, 20, [["x", 1]]),
        _obs("p-a", "h1", 0.903, 20, [["x", 1]]),
        _obs("p-a", "h1", 0.990, 20, [["x", 1]]),
    ]
    r = drift_benchmark(obs, ref, SCHEMA)
    ttr = next(f for f in r.probes[0].features if f.feature == "lexical.type_token_ratio")
    assert ttr.eu < 1.0 < ttr.eu_max


def test_unmatched_keys_are_reported_not_silently_dropped():
    ref = [_obs("p-a", "h1", 0.9, 20, [["x", 1]])]
    obs = [_obs("p-b", "h1", 0.9, 20, [["x", 1]])]
    r = drift_benchmark(obs, ref, SCHEMA)
    assert not r.probes
    assert any("only in observations" in n for n in r.notes)
    assert any("only in reference" in n for n in r.notes)


def test_no_scalar_features_is_named_not_a_zero_score():
    schema = {k: v for k, v in SCHEMA.items() if k in ("probe_id", "response_sha256")}
    ref = [_obs("p-a", "h1", 0.9, 20, [])]
    obs = [_obs("p-a", "h2", 0.9, 20, [])]
    r = drift_benchmark(obs, ref, schema)
    assert r.peak_eu is None
    assert any("benchmark scalar is undefined" in n for n in r.notes)


# ── bundle adapter + rendering ────────────────────────────────────────────────


def _bundle(payloads, schema=SCHEMA):
    return {
        "manifest": {"feature_schema": schema},
        "consensus_results": [{"unit_id": f"u{i}", "payload": p} for i, p in enumerate(payloads)],
    }


def test_bundle_adapter_uses_reference_envelope_and_notes_schema_drift():
    loose = dict(SCHEMA)
    loose["lexical.type_token_ratio"] = {
        **SCHEMA["lexical.type_token_ratio"],
        "comparison": {"rule": "numeric", "rel": 0.5},
    }
    obs_b = _bundle([_obs("p-a", "h1", 0.95, 20, [["x", 1]])], schema=loose)
    ref_b = _bundle([_obs("p-a", "h1", 0.90, 20, [["x", 1]])])
    r = drift_benchmark_bundles(obs_b, ref_b)
    ttr = next(f for f in r.probes[0].features if f.feature == "lexical.type_token_ratio")
    # Scored under the REFERENCE envelope (0.02), not the observation's loose 0.5.
    assert ttr.eu > 1.0
    assert any("REFERENCE bundle's declared envelope" in n for n in r.notes)


def test_format_report_renders():
    ref = [_obs("p-a", "h1", 0.9, 20, [["x", 1]])]
    obs = [_obs("p-a", "h1", 0.9, 20, [["x", 1]])]
    text = format_report(drift_benchmark(obs, ref, SCHEMA))
    assert "drift benchmark: peak" in text and "envelope units" in text


def test_report_round_trips_to_dict():
    ref = [_obs("p-a", "h1", 0.9, 20, [["x", 1]])]
    obs = [_obs("p-a", "h1", 0.9, 20, [["x", 1]])]
    d = drift_benchmark(obs, ref, SCHEMA).to_dict()
    assert isinstance(d["probes"][0]["features"], list)
    assert isinstance(DriftBenchmark(**{}) if False else d["peak_eu"], float)


def test_empty_bundle_side_is_named_not_silent():
    # The Qwen-contrast lesson (2026-07-03): an all-diverged experiment exports
    # ZERO consensus results — the benchmark must say why there is nothing to
    # score, not just print n/a.
    empty = {"manifest": {"feature_schema": SCHEMA}, "consensus_results": []}
    ref_b = _bundle([_obs("p-a", "h1", 0.9, 20, [["x", 1]])])
    r = drift_benchmark_bundles(empty, ref_b)
    assert r.peak_eu is None
    assert any("no consensus results" in n for n in r.notes)


def test_diverged_units_surface_from_the_signed_predicate(monkeypatch):
    # Firewall #1: diverged results are valid evidentiary data — the bundle
    # carries only their hashes, so the benchmark reports their existence per
    # probe (signed evidence of within-run disagreement), never an EU score.

    class _Att:
        diverged_units: ClassVar[list[dict]] = [
            {"unit_id": "u0", "result_hashes": ["a", "b"]},
            {"unit_id": "u1", "result_hashes": ["c", "d"]},
        ]

    monkeypatch.setattr("auspexai_tenant.evidence._attestation_from_bundle", lambda blob: _Att())
    obs_b = {
        "manifest": {"feature_schema": SCHEMA},
        "consensus_results": [],
        "attestation": {"cose_b64": "x"},
        "work_units": [
            {"unit_id": "u0", "payload": {"probe_id": "p-a"}},
            {"unit_id": "u1", "payload": {"probe_id": "p-b"}},
        ],
    }
    ref_b = _bundle([_obs("p-a", "h1", 0.9, 20, [["x", 1]])])
    r = drift_benchmark_bundles(obs_b, ref_b)
    assert r.diverged_units_total == 2
    assert r.diverged_by_key == {"p-a": 1, "p-b": 1}
    text = format_report(r)
    assert "within-run divergence" in text
    assert any("could not corroborate" in n for n in r.notes)


def test_plot_report_writes_png(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")
    from auspexai_tenant.benchmark_plot import plot_report

    ref = [_obs("p-a", "h1", 0.900, 20, [["x", 1], ["y", 1]])]
    obs = [_obs("p-a", "h2", 0.950, 20, [["q", 1], ["r", 1]])]
    out = tmp_path / "ladder.png"
    plot_report(drift_benchmark(obs, ref, SCHEMA), str(out), title="t")
    assert out.stat().st_size > 5000  # a real PNG, not an empty file


def test_benchmark_declaration_parses_and_rejects_typos(tmp_path):
    # Tenant-generic declaration, platform-fixed semantics: [benchmark].reference
    # is the SDK-level surface every tenant uses; typo'd keys fail loudly (the
    # declarative-enforcement-gap lesson: every declared field must be read).
    import pytest as _pytest

    from auspexai_tenant.experiment_config import load_experiment_config

    good = tmp_path / "experiment.toml"
    good.write_text('[experiment]\nlabel = "x"\n[benchmark]\nreference = "exp-ref"\n')
    assert load_experiment_config(good).benchmark_reference == "exp-ref"

    none = tmp_path / "none.toml"
    none.write_text('[experiment]\nlabel = "x"\n')
    assert load_experiment_config(none).benchmark_reference is None

    bad = tmp_path / "bad.toml"
    bad.write_text('[experiment]\nlabel = "x"\n[benchmark]\nrefrence = "exp-ref"\n')
    with _pytest.raises(ValueError, match="unknown key"):
        load_experiment_config(bad).benchmark_reference  # noqa: B018

    # "" = the explicit opt-out (a baseline profile switching OFF an inherited
    # top-level declaration).
    off = tmp_path / "off.toml"
    off.write_text(
        '[experiment]\nlabel = "x"\n[benchmark]\nreference = "exp-ref"\n'
        '[profiles.calibration.benchmark]\nreference = ""\n'
    )
    assert load_experiment_config(off).benchmark_reference == "exp-ref"
    assert load_experiment_config(off, profile="calibration").benchmark_reference is None
