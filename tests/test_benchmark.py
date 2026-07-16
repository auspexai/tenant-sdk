"""The Drift Benchmark (D16.4) — envelope-units math, aggregation honesty,
the byte overlay, valid_when exclusion, and the bundle adapter.

The semantics under test mirror the coordinator reducer exactly where they
overlap (numeric tolerance = max(abs, rel*|reference|); jaccard over full
hashable elements) — the benchmark is another READER of the same declared
envelope, never a second definition of it."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

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


def test_auto_benchmark_scores_declared_run_at_completion(tmp_path, monkeypatch, capsys):
    # `launch`/`run` end with the benchmark established — no flag, no extra
    # command: the recorded declaration drives an export→verify→score→persist.
    import json as _json

    import auspexai_tenant.cli as cli_mod

    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", str(tmp_path / "runs"))
    run_dir = tmp_path / "runs" / "lab-x"
    run_dir.mkdir(parents=True)
    (run_dir / "benchmark_reference.json").write_text(
        _json.dumps(
            {
                "schema": "auspexai-benchmark-declaration/v0",
                "experiment_id": "exp-obs",
                "label": "lab-x",
                "reference_experiment_id": "exp-base",
            }
        )
    )
    bundles = {
        "exp-obs": _bundle([_obs("p-a", "h2", 0.95, 20, [["x", 1]])]),
        "exp-base": _bundle([_obs("p-a", "h1", 0.90, 20, [["x", 1]])]),
    }

    class _Client:
        def export(self, exp_id):
            return bundles[exp_id]

    class _OkVerify:
        ok = True

    monkeypatch.setattr("auspexai_tenant.evidence.verify_bundle", lambda b: _OkVerify())
    cli_mod._auto_benchmark(_Client(), "lab-x")
    saved = _json.loads((run_dir / "benchmark_vs_exp-base.json").read_text())
    assert saved["observation"]["experiment_id"] == "exp-obs"
    assert saved["report"]["peak_eu"] is not None
    out = capsys.readouterr().out
    assert "benchmark: peak" in out
    # Idempotent: a resumed run doesn't re-score.
    cli_mod._auto_benchmark(_Client(), "lab-x")


def test_auto_benchmark_is_silent_without_declaration(tmp_path, monkeypatch, capsys):
    import auspexai_tenant.cli as cli_mod

    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs" / "lab-y").mkdir(parents=True)
    cli_mod._auto_benchmark(object(), "lab-y")
    assert "benchmark" not in capsys.readouterr().out


def test_runs_base_falls_back_to_stable_path_not_cwd(tmp_path, monkeypatch):
    # The 2026-07-03 live lesson: a cwd-relative base is unreadable by any
    # other process. With no config, no env, and no existing ./runs, the base
    # is the stable per-user path both the CLI and the dashboard resolve.
    from auspexai_tenant.runs import runs_base, stable_runs_base

    monkeypatch.delenv("AUSPEXAI_RUNS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # no ./runs here
    assert runs_base(None) == stable_runs_base()
    # An existing ./runs keeps the repo-local workflow working.
    (tmp_path / "runs").mkdir()
    assert runs_base(None) == Path("runs")


def test_additional_observation_rows_score_by_default_diverged_opt_in():
    # D19 (ratified): observe-only extras are first-class evidence — they join
    # the scoring set; diverged/outlier payloads are forensics (opt-in).
    ref_b = _bundle([_obs("p-a", "h1", 0.90, 20, [["x", 1]])])
    obs_b = _bundle([_obs("p-a", "h1", 0.905, 20, [["x", 1]])])
    obs_b["additional_results"] = [
        {
            "unit_id": "u9",
            "integrity_basis": "observation",
            "payload": _obs("p-a", "h2", 0.91, 20, [["x", 1]]),
        },
        {
            "unit_id": "u9",
            "integrity_basis": "diverged",
            "payload": _obs("p-a", "h3", 0.30, 20, [["z", 1]]),
        },
    ]
    from auspexai_tenant.benchmark import observations_from_bundle

    default_obs, _ = observations_from_bundle(obs_b)
    assert len(default_obs) == 2  # consensus + observation, NOT diverged
    forensic_obs, _ = observations_from_bundle(obs_b, include_diverged=True)
    assert len(forensic_obs) == 3
    # And the scalar stays honest: the wild diverged payload moves the score
    # only under the explicit flag.
    calm = drift_benchmark_bundles(obs_b, ref_b)
    wild = drift_benchmark_bundles(obs_b, ref_b, include_diverged=True)
    assert (calm.peak_eu or 0) < (wild.peak_eu or 0)


# ── self-baseline scorer (a run scores against its OWN early rounds) ──────────


def _round_bundle(rows, schema=SCHEMA, *, basis=None, prefix="o10gs"):
    """rows: (round, probe, sha, ttr, tokens, top). basis=None → consensus_results;
    basis='observation' → additional_results (the process_only drift-study path).
    unit_ids mimic the driver's <prefix>-<probe_id>-r<round> stamp."""
    results = [
        {
            "unit_id": f"{prefix}-{probe}-r{rnd}",
            "payload": _obs(probe, sha, ttr, tokens, top),
            **({"integrity_basis": basis} if basis else {}),
        }
        for (rnd, probe, sha, ttr, tokens, top) in rows
    ]
    key = "additional_results" if basis else "consensus_results"
    return {"manifest": {"feature_schema": schema}, key: results}


def test_round_of_unit_parses_the_trailing_round():
    from auspexai_tenant.benchmark import round_of_unit

    assert round_of_unit("o10dgs-p-greeting-r7") == 7  # hyphenated probe + digit-bearing prefix
    assert round_of_unit("cal-p-format-json-r12") == 12
    assert round_of_unit("u0") is None  # a foreign/hand-authored unit id
    assert round_of_unit(None) is None


def test_self_baseline_splits_at_the_k_boundary():
    from auspexai_tenant.benchmark import observations_with_round_from_bundle, split_self_baseline

    b = _round_bundle([(r, "p-a", "h", 0.9, 20, [["x", 1]]) for r in range(5)])
    rows, _ = observations_with_round_from_bundle(b)
    baseline, monitoring, stats = split_self_baseline(rows, 3)
    assert stats == {
        "baseline_rounds": 3,
        "baseline_rows": 3,  # rounds 0,1,2
        "monitoring_rows": 2,  # rounds 3,4
        "unplaced_rows": 0,
    }
    assert len(baseline) == 3 and len(monitoring) == 2


def test_self_baseline_deterministic_run_is_zero_self_drift():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # Identical output every round (a greedy model): monitoring == baseline →
    # 0 EU, 0% breadth. This is the prototype's gpt-oss greedy = 0.00x result.
    b = _round_bundle([(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(6)])
    r = drift_benchmark_self(b, 3)
    assert r.peak_eu == 0.0
    assert r.breadth == 0.0
    assert any("self-baseline: reference = this run's first 3" in n for n in r.notes)


def test_self_baseline_flags_drift_from_its_own_baseline():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # Baseline TTR 0.90 (rounds 0..2); a monitoring round moves it to 0.50 —
    # beyond the model's own 0.02-rel envelope. Drift from ITS OWN normal.
    rows = [(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(3)]
    rows += [(r, "p-a", "h2", 0.50, 20, [["x", 1]]) for r in (3, 4)]
    r = drift_benchmark_self(_round_bundle(rows), 3)
    assert r.peak_eu is not None and r.peak_eu > 1.0
    assert r.breadth == 1.0  # the one probe is beyond its own envelope
    assert r.probes[0].beyond_envelope


def test_self_baseline_converged_within_window_is_self_stable():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # Only baseline rounds exist (the run converged at K, no monitoring round):
    # self-stable by construction — reported 0.0, not None, with a clear note.
    b = _round_bundle([(r, "p-a", "h", 0.9, 20, [["x", 1]]) for r in range(3)])
    r = drift_benchmark_self(b, 5)  # K exceeds the rounds present
    assert r.peak_eu == 0.0 and r.breadth == 0.0
    assert any("no monitoring rows" in n and "self-stable" in n for n in r.notes)


def test_self_baseline_no_baseline_rows_is_named_not_zero():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # Every row is past the boundary (round >= K): the split can't self-reference.
    # Named as unscored (peak None), never mislabeled as 0 drift.
    b = _round_bundle([(r, "p-a", "h", 0.9, 20, [["x", 1]]) for r in (5, 6, 7)])
    r = drift_benchmark_self(b, 3)
    assert r.peak_eu is None
    assert any("no baseline rows" in n for n in r.notes)


def test_self_baseline_requires_positive_k():
    from auspexai_tenant.benchmark import drift_benchmark_self

    b = _round_bundle([(0, "p-a", "h", 0.9, 20, [["x", 1]])])
    try:
        drift_benchmark_self(b, 0)
        raise AssertionError("expected ValueError for baseline_rounds=0")
    except ValueError as e:
        assert "baseline_rounds > 0" in str(e)


def test_self_baseline_scores_observation_basis_rows():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # The drift studies run process_only, so their rows land in
    # additional_results (basis 'observation'), not consensus_results — the
    # self-baseline split must reach them too.
    rows = [(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(3)]
    rows += [(3, "p-a", "h2", 0.50, 20, [["x", 1]])]
    r = drift_benchmark_self(_round_bundle(rows, basis="observation"), 3)
    assert r.peak_eu is not None and r.peak_eu > 1.0


# ── §3.2 self-calibrated envelope (drift measured vs the model's OWN wobble) ──


def test_self_calibrated_normalizers_is_widest_baseline_wobble_floored_at_one():
    from auspexai_tenant.benchmark import self_calibrated_normalizers

    # A noisy baseline (TTR swings 0.90..0.70) → normalizer above the C7 floor.
    noisy = [_obs("p-a", "h", 0.90, 20, [["x", 1]]), _obs("p-a", "h", 0.70, 20, [["x", 1]])]
    assert self_calibrated_normalizers(noisy, SCHEMA)[("p-a", "lexical.type_token_ratio")] > 1.0
    # A perfectly stable baseline floors at exactly 1.0 (never tightens below C7).
    stable = [_obs("p-a", "h", 0.90, 20, [["x", 1]]), _obs("p-a", "h", 0.90, 20, [["x", 1]])]
    assert self_calibrated_normalizers(stable, SCHEMA)[("p-a", "lexical.type_token_ratio")] == 1.0


def test_self_calibrated_envelope_loosens_for_a_noisy_baseline():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # Baseline TTR swings 0.90..0.70 (this probe is naturally noisy); a monitoring
    # round at 0.85 is WITHIN that own wobble. The fixed C7 envelope flags it; the
    # self-calibrated envelope reads it as < 1 EU (within the model's own noise).
    rows = [(0, "p-a", "h", 0.90, 20, [["x", 1]]), (1, "p-a", "h", 0.70, 20, [["x", 1]])]
    rows += [(2, "p-a", "h2", 0.85, 20, [["x", 1]])]
    b = _round_bundle(rows)
    fixed = drift_benchmark_self(b, 2)
    calib = drift_benchmark_self(b, 2, calibrate_envelope=True)
    assert fixed.peak_eu > 1.0  # fixed C7 envelope flags it
    assert calib.peak_eu < 1.0  # within the model's OWN baseline wobble
    assert calib.probes[0].beyond_envelope is False
    assert any("self-calibrated envelope" in n for n in calib.notes)


def test_self_calibrated_envelope_floors_at_declared_for_a_stable_baseline():
    from auspexai_tenant.benchmark import drift_benchmark_self

    # A perfectly stable baseline → normalizer floored at 1.0, so calibrated ==
    # fixed: self-calibration only loosens for a noisy model, never tightens below
    # the calibrated-safe C7 floor. A genuine monitoring drift still reads beyond.
    rows = [(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(3)]
    rows += [(3, "p-a", "h2", 0.70, 20, [["x", 1]])]
    b = _round_bundle(rows)
    fixed = drift_benchmark_self(b, 3)
    calib = drift_benchmark_self(b, 3, calibrate_envelope=True)
    assert fixed.peak_eu is not None and fixed.peak_eu > 1.0
    assert calib.peak_eu == fixed.peak_eu  # floored → unchanged


def test_benchmark_config_self_mode_and_baseline_rounds(tmp_path):
    from auspexai_tenant.experiment_config import load_experiment_config

    p = tmp_path / "experiment.toml"
    p.write_text(
        '[experiment]\nlabel = "x"\n[driver]\nbaseline_rounds = 3\nmax_rounds = 100\n'
        '[benchmark]\nmode = "self_baseline"\n'
    )
    cfg = load_experiment_config(p)
    assert cfg.benchmark_mode == "self_baseline"
    assert cfg.benchmark_reference is None  # no external reference in self mode
    assert cfg.benchmark_baseline_rounds == 3


def test_benchmark_calibrate_envelope_config(tmp_path):
    from auspexai_tenant.experiment_config import load_experiment_config

    on = tmp_path / "on.toml"
    on.write_text(
        '[experiment]\nlabel = "x"\n[driver]\nbaseline_rounds = 3\n'
        '[benchmark]\nmode = "self_baseline"\ncalibrate_envelope = true\n'
    )
    assert load_experiment_config(on).benchmark_calibrate_envelope is True
    off = tmp_path / "off.toml"
    off.write_text('[experiment]\nlabel = "x"\n[benchmark]\nmode = "self_baseline"\n')
    assert load_experiment_config(off).benchmark_calibrate_envelope is False


def test_benchmark_reference_self_alias_and_k_clamps_to_max_rounds(tmp_path):
    from auspexai_tenant.experiment_config import load_experiment_config

    # reference = "self" is the self_baseline alias; K clamps to max_rounds.
    p = tmp_path / "experiment.toml"
    p.write_text(
        '[experiment]\nlabel = "x"\n[driver]\nbaseline_rounds = 9\nmax_rounds = 4\n'
        '[benchmark]\nreference = "self"\n'
    )
    cfg = load_experiment_config(p)
    assert cfg.benchmark_mode == "self_baseline"
    assert cfg.benchmark_reference is None
    assert cfg.benchmark_baseline_rounds == 4  # min(9, 4)


def test_benchmark_bad_mode_is_rejected(tmp_path):
    import pytest as _pytest

    from auspexai_tenant.experiment_config import load_experiment_config

    p = tmp_path / "experiment.toml"
    p.write_text('[experiment]\nlabel = "x"\n[benchmark]\nmode = "whoops"\n')
    with _pytest.raises(ValueError, match="mode must be"):
        load_experiment_config(p).benchmark_mode  # noqa: B018


def test_record_and_auto_benchmark_self_baseline(tmp_path, monkeypatch, capsys):
    # End-to-end: launch records a self-baseline declaration (K from [driver]),
    # and completion auto-scores the run against its OWN baseline → benchmark_self.json.
    import json as _json

    import auspexai_tenant.cli as cli_mod
    from auspexai_tenant.experiment_config import load_experiment_config

    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", str(tmp_path / "runs"))
    p = tmp_path / "experiment.toml"
    p.write_text(
        '[experiment]\nlabel = "lab-z"\n[driver]\nbaseline_rounds = 3\n'
        '[benchmark]\nmode = "self_baseline"\n'
    )
    cfg = load_experiment_config(p)
    cli_mod._record_benchmark_declaration(cfg, "exp-z", "lab-z")
    run_dir = tmp_path / "runs" / "lab-z"
    decl = _json.loads((run_dir / "benchmark_reference.json").read_text())
    assert decl["mode"] == "self_baseline" and decl["baseline_rounds"] == 3
    assert decl["calibrate_envelope"] is False  # not declared → off

    # A monitoring round (r3) drifts beyond the baseline (r0..2) → scored self-drift.
    rows = [(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(3)]
    rows += [(3, "p-a", "h2", 0.50, 20, [["x", 1]])]
    bundle = _round_bundle(rows)

    class _Client:
        def export(self, exp_id):
            return bundle

    class _Ok:
        ok = True

    monkeypatch.setattr("auspexai_tenant.evidence.verify_bundle", lambda b: _Ok())
    cli_mod._auto_benchmark(_Client(), "lab-z")
    saved = _json.loads((run_dir / "benchmark_self.json").read_text())
    assert saved["reference"] == {
        "mode": "self_baseline",
        "baseline_rounds": 3,
        "calibrate_envelope": False,
    }
    assert saved["report"]["peak_eu"] > 1.0
    assert "self-drift peak" in capsys.readouterr().out
    cli_mod._auto_benchmark(_Client(), "lab-z")  # idempotent — no re-score/crash


def test_self_baseline_entry_carries_k_from_either_record_shape():
    # build_entry_self must record the real K whether the record's `reference` is the
    # CLI's canonical FLAT shape ({mode, baseline_rounds}) or the dashboard's NESTED
    # shape ({self_baseline: {baseline_rounds}}); reading only flat serialized null K.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from auspexai_tenant.benchmark_entry import build_entry_self, verify_entry

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, data: bytes) -> bytes:
            return self._k.sign(data)

    report = {
        "peak_eu": 1.1,
        "breadth": 0.08,
        "byte_divergence_rate": 0.05,
        "diverged_units_total": None,
        "key_feature": "probe_id",
        "probes": [],
    }
    bundle = {
        "manifest_hash": "m" * 64,
        "manifest": {
            "models": [{"id": "qwen3-1.7b-q4"}],
            "inference_determinism": {"temperature": 0.8, "top_p": 0.9},
        },
        "attestation": {"merkle_root": "r" * 64, "algorithm": "result-set-v1"},
    }

    def _entry(reference):
        e = build_entry_self(
            record={
                "computed_at": "2026-07-16T20:00:00+00:00",
                "observation": {"experiment_id": "exp-a", "label": "run-a"},
                "reference": reference,
                "report": report,
            },
            observation_bundle=bundle,
            tenant_id="vigiles-lab",
            key=_Key(),
        )
        return verify_entry(e)

    nested = _entry({"self_baseline": {"baseline_rounds": 5, "calibrate_envelope": True}})
    assert nested["self_baseline"]["baseline_rounds"] == 5  # was null before the fix
    assert nested["self_baseline"]["calibrate_envelope"] is True
    flat = _entry({"mode": "self_baseline", "baseline_rounds": 7, "calibrate_envelope": False})
    assert flat["self_baseline"]["baseline_rounds"] == 7
    # model + generation come from the manifest regardless of the reference shape.
    assert nested["entry_kind"] == "self_baseline"
    assert nested["self_baseline"]["model"] == "qwen3-1.7b-q4"
    assert "sampling" in nested["self_baseline"]["generation"]  # temp=0.8 → sampling(…)


def test_registry_entry_signs_and_verifies_and_tamper_fails():
    # G5: the entry is the researcher's SIGNED claim — pubkey inside the signed
    # body (key identity is part of the claim), any field tamper breaks it.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from auspexai_tenant.benchmark_entry import build_entry, verify_entry

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, data: bytes) -> bytes:
            return self._k.sign(data)

    record = {
        "computed_at": "2026-07-03T20:00:00+00:00",
        "observation": {"experiment_id": "exp-obs", "label": "run-a"},
        "reference": {"experiment_id": "exp-ref", "label": "calibration"},
        "report": {
            "peak_eu": 10.0,
            "breadth": 1.0,
            "byte_divergence_rate": 0.94,
            "diverged_units_total": None,
            "key_feature": "probe_id",
            "probes": [{"key": "p-a", "peak_eu": 10.0, "beyond_envelope": True}],
        },
    }
    bundle = {
        "manifest_hash": "m" * 64,
        "attestation": {
            "merkle_root": "r" * 64,
            "algorithm": "result-set-v1",
            "rekor_log_index": 123,
            "rekor_entry_uuid": "uuid-1",
        },
    }
    entry = build_entry(
        record=record,
        observation_bundle=bundle,
        reference_bundle=bundle,
        tenant_id="vigiles-lab",
        key=_Key(),
    )
    payload = verify_entry(entry)
    assert payload is not None
    assert payload["observation"]["attestation"]["rekor_log_index"] == 123
    assert payload["report"]["peak_eu"] == 10.0
    # Tampering with the signed bytes breaks it.
    import base64 as _b64

    body = _b64.b64decode(entry["payload_b64"]).replace(b"10.0", b"0.1")
    assert verify_entry({**entry, "payload_b64": _b64.b64encode(body).decode()}) is None
    # Envelope-pubkey swap breaks it (the key identity is INSIDE the payload).
    assert verify_entry({**entry, "publisher_pubkey_hex": "ab" * 32}) is None


def test_config_delta_is_derived_not_written():
    from auspexai_tenant.benchmark_entry import derive_config_delta

    ref = {
        "models": [{"id": "gemma-3-1b-it-q4"}],
        "reducer": {"kind": "builtin_within_cell_tolerance"},
        "feature_schema": {"f": 1},
    }
    same = derive_config_delta(dict(ref), dict(ref))
    assert same["changed"] == {} and "model" in same["unchanged"]

    qwen = {
        "models": [{"id": "qwen2.5-0.5b-instruct-q4"}],
        "reducer": {"kind": "builtin_process_only"},
        "feature_schema": {"f": 1},
    }
    d = derive_config_delta(qwen, ref)
    assert d["changed"]["model"] == {"from": "gemma-3-1b-it-q4", "to": "qwen2.5-0.5b-instruct-q4"}
    assert d["changed"]["reducer"]["to"] == "builtin_process_only"

    sampling = {**ref, "inference_determinism": {"temperature": 0.8, "top_p": 0.9, "seed": 0}}
    d2 = derive_config_delta(sampling, ref)
    assert d2["changed"]["generation"] == {
        "from": "greedy",
        "to": "sampling(temp=0.8,top_p=0.9,seeded)",
    }


def test_grounded_verify_requires_coordinator_custody_binding():
    # Researcher-push admission rule: publisher key must be the coordinator-
    # attested COLLECTOR of both experiments — machine-checkable, no curator.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from auspexai_tenant.benchmark_entry import build_entry, verify_entry_grounded

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, d):
            return self._k.sign(d)

    coord = _Key()
    publisher = _Key()

    def bundle(mh):
        rec = f"root|{publisher.pubkey_hex}|2026-07-03T22:00:00+00:00|{mh}".encode()
        return {
            "manifest_hash": mh,
            "manifest": {"models": [{"id": "m"}], "feature_schema": {}},
            "attestation": {"merkle_root": "r" * 64, "rekor_log_index": 1},
            "transfer": {
                "result_set_root": "root",
                "collected_by_pubkey": publisher.pubkey_hex,
                "collected_at": "2026-07-03T22:00:00+00:00",
                "manifest_hash": mh,
                "coordinator_pubkey_hex": coord.pubkey_hex,
                "coordinator_signature": coord.sign(rec).hex(),
            },
        }

    record = {
        "computed_at": "t",
        "observation": {"experiment_id": "exp-a", "label": "a"},
        "reference": {"experiment_id": "exp-b", "label": "b"},
        "report": {
            "peak_eu": 1.0,
            "breadth": 0.0,
            "byte_divergence_rate": 0.0,
            "diverged_units_total": None,
            "key_feature": "probe_id",
            "probes": [],
        },
    }
    entry = build_entry(
        record=record,
        observation_bundle=bundle("a" * 64),
        reference_bundle=bundle("b" * 64),
        tenant_id="t",
        key=publisher,
    )
    # Grounded iff the coordinator key is pinned.
    assert verify_entry_grounded(entry, authorized_signers=(coord.pubkey_hex,)) is not None
    assert verify_entry_grounded(entry, authorized_signers=("ab" * 32,)) is None
    # A DIFFERENT publisher signing the same custody records must fail.
    thief = _Key()
    stolen = build_entry(
        record=record,
        observation_bundle=bundle("a" * 64),
        reference_bundle=bundle("b" * 64),
        tenant_id="t",
        key=thief,
    )
    # custody binds `publisher`, thief signed the entry → collected_by mismatch.
    assert verify_entry_grounded(stolen, authorized_signers=(coord.pubkey_hex,)) is None


def test_self_baseline_entry_is_single_anchor_and_grounds_on_observation_only():
    # §3.4: a self-baseline entry has ONE experiment (its own baseline is the
    # reference). It carries the run's OWN model/generation + K + calibrate in a
    # self_baseline block, and grounds on the single observation custody.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from auspexai_tenant.benchmark_entry import (
        build_entry_self,
        verify_entry,
        verify_entry_grounded,
    )

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, d):
            return self._k.sign(d)

    coord, publisher = _Key(), _Key()

    def bundle(mh, temp=None):
        rec = f"root|{publisher.pubkey_hex}|2026-07-06T11:00:00+00:00|{mh}".encode()
        manifest = {"models": [{"id": "gpt-oss-20b-mxfp4"}], "feature_schema": {}}
        if temp:
            manifest["inference_determinism"] = {"temperature": temp, "top_p": 0.9}
        return {
            "manifest_hash": mh,
            "manifest": manifest,
            "attestation": {"merkle_root": "r" * 64, "rekor_log_index": 1},
            "transfer": {
                "result_set_root": "root",
                "collected_by_pubkey": publisher.pubkey_hex,
                "collected_at": "2026-07-06T11:00:00+00:00",
                "manifest_hash": mh,
                "coordinator_pubkey_hex": coord.pubkey_hex,
                "coordinator_signature": coord.sign(rec).hex(),
            },
        }

    record = {
        "computed_at": "t",
        "observation": {"experiment_id": "exp-gptoss", "label": "gptoss"},
        "reference": {"mode": "self_baseline", "baseline_rounds": 5, "calibrate_envelope": True},
        "report": {
            "peak_eu": 0.0,
            "breadth": 0.0,
            "byte_divergence_rate": 0.0,
            "diverged_units_total": None,
            "key_feature": "probe_id",
            "probes": [],
        },
    }
    entry = build_entry_self(
        record=record,
        observation_bundle=bundle("a" * 64, temp=0.8),
        tenant_id="vigiles-lab",
        key=publisher,
    )
    payload = verify_entry(entry)
    assert payload["entry_kind"] == "self_baseline"
    assert "reference" not in payload  # single anchor — no reference experiment
    sb = payload["self_baseline"]
    assert sb["model"] == "gpt-oss-20b-mxfp4"
    assert sb["generation"].startswith("sampling(temp=0.8")
    assert sb["baseline_rounds"] == 5 and sb["calibrate_envelope"] is True
    # Grounds on the SINGLE observation custody (the both-sides loop must not
    # demand a nonexistent reference side).
    assert verify_entry_grounded(entry, authorized_signers=(coord.pubkey_hex,)) is not None
    assert verify_entry_grounded(entry, authorized_signers=("ab" * 32,)) is None
    # A thief re-signing the same custody fails (collected_by binds the publisher).
    thief = _Key()
    stolen = build_entry_self(
        record=record,
        observation_bundle=bundle("a" * 64, temp=0.8),
        tenant_id="vigiles-lab",
        key=thief,
    )
    assert verify_entry_grounded(stolen, authorized_signers=(coord.pubkey_hex,)) is None


def test_publish_self_baseline_writes_a_single_anchor_entry(tmp_path, monkeypatch):
    # §3.4: `benchmark publish` on a self-baseline run builds + writes a single-
    # anchor entry (self_baseline block, own model, self-drift score), no submit.
    import json as _json

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    import auspexai_tenant.cli as cli_mod
    from auspexai_tenant.benchmark_entry import verify_entry
    from auspexai_tenant.runs import RunLayout, runs_base

    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", str(tmp_path / "runs"))
    layout = RunLayout("lab-z", base=runs_base(str(tmp_path / "runs")))
    layout.dir.mkdir(parents=True)
    (layout.dir / "benchmark_reference.json").write_text(
        _json.dumps({"mode": "self_baseline", "baseline_rounds": 3, "calibrate_envelope": False})
    )
    assert cli_mod._is_self_baseline_run(layout) is True
    assert cli_mod._self_baseline_k_and_calibrate(layout) == (3, False)

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, d):
            return self._k.sign(d)

    publisher = _Key()
    rows = [(r, "p-a", "h", 0.90, 20, [["x", 1]]) for r in range(3)]
    rows += [(3, "p-a", "h2", 0.50, 20, [["x", 1]])]
    bundle = _round_bundle(rows)
    bundle["manifest_hash"] = "a" * 64
    bundle["manifest"]["tenant_id"] = "vigiles-lab"
    bundle["manifest"]["models"] = [{"id": "gpt-oss-20b-mxfp4"}]
    bundle["attestation"] = {"merkle_root": "r" * 64, "rekor_log_index": 1}
    bundle["transfer"] = {"result_set_root": "root", "collected_by_pubkey": publisher.pubkey_hex}

    class _Client:
        def export(self, e):
            return bundle

    class _Ok:
        ok = True

    class _Exp:  # authorization request fails fast → authorization None path
        def __init__(self, *a, **k):
            pass

        def _post(self, *a, **k):
            raise ConnectionError("no coordinator in the test")

    monkeypatch.setattr("auspexai_tenant.evidence.verify_bundle", lambda b: _Ok())
    monkeypatch.setattr("auspexai_tenant.experiment.Experiment", _Exp)
    cli_mod._publish_self_baseline(
        _Client(), publisher, "http://x", "exp-z", "lab-z", layout, no_submit=True
    )
    entry = _json.loads((layout.dir / "benchmark_entry_self.json").read_text())
    payload = verify_entry(entry)
    assert payload["entry_kind"] == "self_baseline"
    assert "reference" not in payload
    assert payload["self_baseline"]["model"] == "gpt-oss-20b-mxfp4"
    assert payload["self_baseline"]["baseline_rounds"] == 3
    assert payload["report"]["peak_eu"] > 1.0  # the r3 monitoring drift


def test_error_code_parses_the_real_nested_envelope():
    # The 2026-07-04 resume-replay 409: the coordinator nests under "detail",
    # so the typed-conflict parse never matched and resume idempotency was
    # dead code. Both shapes must parse.
    import httpx

    from auspexai_tenant.experiment import _error_code

    nested = httpx.Response(
        409, json={"detail": {"error": {"code": "unit_id_already_submitted", "message": "x"}}}
    )
    bare = httpx.Response(409, json={"error": {"code": "max_units_exceeded"}})
    assert _error_code(nested) == "unit_id_already_submitted"
    assert _error_code(bare) == "max_units_exceeded"
    assert _error_code(httpx.Response(409, text="<html>")) is None


def test_grounded_verify_validates_the_authorization_block():
    # G6: when the entry carries a publication authorization, the grounded rule
    # verifies the coordinator's signature, the publisher binding, the
    # experiment match, and standing >= R1 — a forged or transplanted block
    # fails admission.
    import json as _json

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from auspexai_tenant.benchmark_entry import build_entry, verify_entry_grounded

    class _Key:
        def __init__(self):
            self._k = Ed25519PrivateKey.generate()
            self.pubkey_hex = self._k.public_key().public_bytes_raw().hex()

        def sign(self, d):
            return self._k.sign(d)

    coord, publisher = _Key(), _Key()

    def bundle(mh):
        rec = f"root|{publisher.pubkey_hex}|2026-07-06T20:00:00+00:00|{mh}".encode()
        return {
            "manifest_hash": mh,
            "manifest": {"models": [{"id": "m"}], "feature_schema": {}},
            "attestation": {"merkle_root": "r" * 64, "rekor_log_index": 1},
            "transfer": {
                "result_set_root": "root",
                "collected_by_pubkey": publisher.pubkey_hex,
                "collected_at": "2026-07-06T20:00:00+00:00",
                "manifest_hash": mh,
                "coordinator_pubkey_hex": coord.pubkey_hex,
                "coordinator_signature": coord.sign(rec).hex(),
            },
        }

    def auth_block(standing=2, experiment_id="exp-a", pub=None):
        body = {
            "schema": "auspexai-publication-authorization/v0",
            "action": "benchmark-publication",
            "experiment_id": experiment_id,
            "tenant_id": "t",
            "publisher_pubkey": (pub or publisher).pubkey_hex,
            "standing_at_issue": standing,
            "summary_sha256": "0" * 64,
            "issued_at": "2026-07-06T20:00:00+00:00",
        }
        # sign over the body EXCLUDING coordinator_signature/pubkey — mirror the
        # coordinator: canonical of the block sans coordinator_pubkey_hex.
        signed = dict(body)
        signed["coordinator_signature"] = coord.sign(
            _json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()
        ).hex()
        signed["coordinator_pubkey_hex"] = coord.pubkey_hex
        return signed

    record = {
        "computed_at": "t",
        "observation": {"experiment_id": "exp-a", "label": "a"},
        "reference": {"experiment_id": "exp-b", "label": "b"},
        "report": {
            "peak_eu": 1.0,
            "breadth": 0.0,
            "byte_divergence_rate": 0.0,
            "diverged_units_total": None,
            "key_feature": "probe_id",
            "probes": [],
        },
    }
    kw = dict(
        record=record,
        observation_bundle=bundle("a" * 64),
        reference_bundle=bundle("b" * 64),
        tenant_id="t",
        key=publisher,
    )
    ok = build_entry(**kw, authorization=auth_block())
    assert verify_entry_grounded(ok, authorized_signers=(coord.pubkey_hex,)) is not None
    # Wrong experiment in the block → transplant refused.
    wrong = build_entry(**kw, authorization=auth_block(experiment_id="exp-OTHER"))
    assert verify_entry_grounded(wrong, authorized_signers=(coord.pubkey_hex,)) is None
    # A block naming a different publisher → refused.
    thief = _Key()
    swapped = build_entry(**kw, authorization=auth_block(pub=thief))
    assert verify_entry_grounded(swapped, authorized_signers=(coord.pubkey_hex,)) is None
    # No block at all → still admissible (pre-flag-day grace).
    plain = build_entry(**kw)
    assert verify_entry_grounded(plain, authorized_signers=(coord.pubkey_hex,)) is not None


def test_capture_raw_declaration_and_manifest(tmp_path):
    # D20: [capture] raw opt-in → manifest capture block + sensitive flag.
    from auspexai_tenant.experiment_config import load_experiment_config, manifest_dict_from_config

    base = (
        "[experiment]\n"
        'tenant_id = "lab"\ncontact = "a@b.org"\nmodel_id = "m"\n'
        f'research_goal = "{"x" * 60}"\nprompt_characteristics = "neutral probes"\n'
        '[executor]\ncommand = ["python", "x.py"]\n'
        '[reducer]\nkind = "builtin_hash_agreement"\n'
    )
    off = tmp_path / "off.toml"
    off.write_text(base)
    assert load_experiment_config(off).capture_raw is False

    on = tmp_path / "on.toml"
    on.write_text(base + "[capture]\nraw = true\n")
    cfg = load_experiment_config(on)
    assert cfg.capture_raw is True
    m = manifest_dict_from_config(cfg, package_sha256="a" * 64, label="capture-x")
    assert m["capture"] == {"raw": True}
    assert "raw_content_capture" in m["sensitive_content_flags"]

    # The gap this closes: a capture manifest must PASS Manifest.model_validate —
    # `experiment build`/`submit` both validate through it, and the writer's
    # `capture` + `raw_content_capture` members were never taught to the model, so
    # every capture build 500'd (extra_forbidden / literal_error). Crucially,
    # raw_content_capture must NOT trip the §5.12 approver-attestation requirement
    # (D20: declared, not separately reviewed) — no approver_attestations here.
    from auspexai_tenant.manifest import Manifest

    assert "approver_attestations" not in m
    validated = Manifest.model_validate(m)
    assert validated.capture is not None and validated.capture.raw is True

    # But a HARMFUL-content flag still requires an approver attestation.
    harmful = {**m, "sensitive_content_flags": [*m["sensitive_content_flags"], "dual_use"]}
    with pytest.raises(ValueError, match="approver_attestations is required"):
        Manifest.model_validate(harmful)

    # D20 coherence: capture WITHOUT the review flag is a review-bypass → refused.
    bad = {
        **m,
        "sensitive_content_flags": [
            f for f in m["sensitive_content_flags"] if f != "raw_content_capture"
        ],
    }
    with pytest.raises(ValueError, match="raw_content_capture"):
        Manifest.model_validate(bad)
