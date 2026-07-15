"""The Drift Benchmark — one comparable measurement of behavioral drift
(D16.4; drift_benchmark_design.md, 2026-07-03).

The problem it solves: a Vigiles-class run collects several per-probe features
(a byte anchor, numeric summaries, a top-tokens set) that are deliberately
ORTHOGONAL — none is individually "the drift number", so findings are hard to
communicate and impossible to compare across models/configurations.

The idea: every feature that participates in consensus already declares a
CALIBRATED comparison envelope (D16.1 `comparison`; calibrated by the C7
Phase-0 campaign). Dividing a feature's observed delta by ITS OWN envelope
yields a dimensionless quantity — **envelope units (EU)** — "how many times the
calibrated noise floor did this feature move." EU is comparable across
orthogonal features because each is normalized by its own calibrated noise:

    numeric (rel r / abs a):  EU = |v - ref| / max(a, r * |ref|)
    set_jaccard (min m):      EU = (1 - jaccard) / (1 - m)
    exact / categorical:      binary -> NEVER in the scalar (the overlay; see below)

EU < 1 means within calibrated noise; EU >= 1 means the behavior moved beyond
what the calibration says same-behavior can produce. The per-probe score is the
MAX over features (the C7 conjunctive predicate, inverted); the benchmark's
headline **peak** is the max over probes, with **breadth** (the fraction of
probes at or beyond 1.0) as the companion so one drifted probe reads
differently from a panel-wide shift.

Honesty rules (the same discipline as the consensus machinery — this module
REUSES the declared envelope, it never invents one):
- Only features with a declared, non-binary `comparison` participate in the
  scalar. Binary rules (exact / categorical_exact) ride as the BYTE-DIVERGENCE
  OVERLAY — Phase-0 proved byte-difference is not behavioral difference, so a
  hash mismatch must never inflate the score, and feature-blind divergence
  (2-char outputs) must never hide behind a low one.
- `valid_when` predicates are honored: an observation invalid for a feature is
  excluded from that feature's pairs (and reported).
- Deltas are computed on every cross pair (obs_a x obs_b) per probe, mirroring
  the Phase-0 pairwise method; the reported EU per feature is the MEDIAN pair
  (robust), with the max kept as spread.
- The semantics mirror the coordinator reducer byte-for-byte where they
  overlap: numeric tolerance = max(abs, rel*|reference|) against the REFERENCE
  side's value; set elements are made hashable exactly like `_as_frozenset`
  ([token, count] pairs compare as full elements).
- DESCRIPTIVE ONLY: the benchmark is a communication surface. It never feeds
  consensus, receipts, or trust (firewall #1 hygiene).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field, replace
from typing import Any

BINARY_RULES = frozenset({"exact", "categorical_exact"})
SCALAR_RULES = frozenset({"numeric", "set_jaccard"})


# ── payload access (dotted feature paths, the load_verified convention) ──────


def _get_path(payload: dict, path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _hashable(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _as_frozenset(value: Any) -> frozenset:
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(_hashable(el) for el in value)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _valid_for(decl: dict, payload: dict) -> bool:
    """Evaluate the feature's declared `valid_when` against this observation."""
    vw = decl.get("valid_when")
    if not vw:
        return True
    actual = _get_path(payload, vw["field"])
    if actual is None:
        return False
    op, expected = vw["op"], vw["value"]
    try:
        return {
            ">=": actual >= expected,
            "<=": actual <= expected,
            ">": actual > expected,
            "<": actual < expected,
            "==": actual == expected,
            "!=": actual != expected,
        }[op]
    except (KeyError, TypeError):
        return False


# ── envelope units ────────────────────────────────────────────────────────────


def envelope_units(comparison: dict, value: Any, reference: Any) -> float | None:
    """The feature's delta in envelope units, or None when undefined (binary
    rule, missing value, or a zero-tolerance denominator)."""
    rule = comparison.get("rule")
    if rule == "numeric":
        if not _is_number(value) or not _is_number(reference):
            return None
        tol = 0.0
        if comparison.get("abs") is not None:
            tol = max(tol, abs(float(comparison["abs"])))
        if comparison.get("rel") is not None:
            tol = max(tol, abs(float(comparison["rel"])) * abs(float(reference)))
        if tol == 0.0:
            return None  # a zero denominator would fabricate infinite drift
        return abs(float(value) - float(reference)) / tol
    if rule == "set_jaccard":
        m = float(comparison.get("min", 0.0))
        if m >= 1.0:
            return None  # jaccard 1.0 required == binary; overlay territory
        j = _jaccard(_as_frozenset(value), _as_frozenset(reference))
        return (1.0 - j) / (1.0 - m)
    return None  # binary rules never yield EU


# ── report shapes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    rule: str
    eu: float | None  # median over cross pairs; None = no valid pairs
    eu_max: float | None
    pairs: int
    invalid_excluded: int  # observations excluded by valid_when


@dataclass(frozen=True)
class ProbeDrift:
    key: str
    features: tuple[FeatureDrift, ...]
    peak_eu: float | None  # max over features' median EU
    beyond_envelope: bool  # peak_eu >= 1.0
    byte_divergence_rate: float | None  # mismatch rate over binary-rule anchors
    observations: int
    reference_observations: int


@dataclass(frozen=True)
class DriftBenchmark:
    """The comparable measurement. `peak_eu` is the headline scalar; `breadth`
    is its companion; `byte_divergence_rate` is the overlay."""

    probes: tuple[ProbeDrift, ...]
    peak_eu: float | None
    breadth: float | None  # fraction of scored probes with peak >= 1.0
    byte_divergence_rate: float | None
    key_feature: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    # The WITHIN-RUN divergence overlay (bundle path only): units whose replicas
    # disagreed under an agreement reducer, read from the SIGNED predicate's
    # diverged_units. Their payloads are not exportable (hashes only), so they
    # cannot be EU-scored — but their existence is custody-verified evidence
    # that the run itself could not corroborate. An apparatus/stability signal,
    # never folded into the drift-vs-reference scalar. {probe_key: diverged_count}
    diverged_units_total: int | None = None
    diverged_by_key: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return {
            "peak_eu": self.peak_eu,
            "breadth": self.breadth,
            "byte_divergence_rate": self.byte_divergence_rate,
            "key_feature": self.key_feature,
            "notes": list(self.notes),
            "diverged_units_total": self.diverged_units_total,
            "diverged_by_key": self.diverged_by_key,
            "probes": [
                {
                    "key": p.key,
                    "peak_eu": p.peak_eu,
                    "beyond_envelope": p.beyond_envelope,
                    "byte_divergence_rate": p.byte_divergence_rate,
                    "observations": p.observations,
                    "reference_observations": p.reference_observations,
                    "features": [
                        {
                            "feature": f.feature,
                            "rule": f.rule,
                            "eu": f.eu,
                            "eu_max": f.eu_max,
                            "pairs": f.pairs,
                            "invalid_excluded": f.invalid_excluded,
                        }
                        for f in p.features
                    ],
                }
                for p in self.probes
            ],
        }


# ── the benchmark ─────────────────────────────────────────────────────────────


def drift_benchmark(
    observations: list[dict],
    reference: list[dict],
    feature_schema: dict[str, dict],
    *,
    key_feature: str = "probe_id",
) -> DriftBenchmark:
    """Score `observations` against `reference` (two sets of result payloads —
    e.g. two runs, two models, or two configurations) over the declared,
    calibrated feature envelope. Payloads are the executor result dicts (the
    same objects `load_verified` flattens under `output.*`)."""
    notes: list[str] = []
    scalar_features = {
        path: decl
        for path, decl in feature_schema.items()
        if isinstance(decl.get("comparison"), dict)
        and decl["comparison"].get("rule") in SCALAR_RULES
    }
    binary_features = {
        path: decl
        for path, decl in feature_schema.items()
        if isinstance(decl.get("comparison"), dict)
        and decl["comparison"].get("rule") in BINARY_RULES
    }
    if not scalar_features:
        notes.append(
            "no feature declares a scalar (numeric/set_jaccard) comparison — "
            "the benchmark scalar is undefined; only the byte overlay is reported"
        )

    def by_key(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in rows:
            k = _get_path(r, key_feature)
            if k is not None:
                out.setdefault(str(k), []).append(r)
        return out

    obs_by, ref_by = by_key(observations), by_key(reference)
    shared = sorted(set(obs_by) & set(ref_by))
    only_obs, only_ref = set(obs_by) - set(ref_by), set(ref_by) - set(obs_by)
    if only_obs:
        notes.append(f"keys only in observations (unscored): {sorted(only_obs)}")
    if only_ref:
        notes.append(f"keys only in reference (unscored): {sorted(only_ref)}")

    probes: list[ProbeDrift] = []
    for key in shared:
        a_rows, b_rows = obs_by[key], ref_by[key]
        feats: list[FeatureDrift] = []
        for path, decl in sorted(scalar_features.items()):
            cmp_ = decl["comparison"]
            a_valid = [r for r in a_rows if _valid_for(decl, r)]
            b_valid = [r for r in b_rows if _valid_for(decl, r)]
            excluded = (len(a_rows) - len(a_valid)) + (len(b_rows) - len(b_valid))
            eus = [
                eu
                for a in a_valid
                for b in b_valid
                if (eu := envelope_units(cmp_, _get_path(a, path), _get_path(b, path))) is not None
            ]
            feats.append(
                FeatureDrift(
                    feature=path,
                    rule=cmp_["rule"],
                    eu=statistics.median(eus) if eus else None,
                    eu_max=max(eus) if eus else None,
                    pairs=len(eus),
                    invalid_excluded=excluded,
                )
            )
        scored = [f.eu for f in feats if f.eu is not None]
        peak = max(scored) if scored else None

        # The overlay: mismatch rate across cross pairs on binary-rule features.
        mismatches = total = 0
        for path in sorted(binary_features):
            for a in a_rows:
                for b in b_rows:
                    va, vb = _get_path(a, path), _get_path(b, path)
                    if va is None and vb is None:
                        continue
                    total += 1
                    if _hashable(va) != _hashable(vb):
                        mismatches += 1
        probes.append(
            ProbeDrift(
                key=key,
                features=tuple(feats),
                peak_eu=peak,
                beyond_envelope=bool(peak is not None and peak >= 1.0),
                byte_divergence_rate=(mismatches / total) if total else None,
                observations=len(a_rows),
                reference_observations=len(b_rows),
            )
        )

    scored_probes = [p for p in probes if p.peak_eu is not None]
    overlay_rates = [p.byte_divergence_rate for p in probes if p.byte_divergence_rate is not None]
    return DriftBenchmark(
        probes=tuple(probes),
        peak_eu=max((p.peak_eu for p in scored_probes), default=None),
        breadth=(
            sum(1 for p in scored_probes if p.beyond_envelope) / len(scored_probes)
            if scored_probes
            else None
        ),
        byte_divergence_rate=(sum(overlay_rates) / len(overlay_rates) if overlay_rates else None),
        key_feature=key_feature,
        notes=tuple(notes),
    )


# ── bundle adapters (the offline, custody-verified entry point) ───────────────


def observations_from_bundle(
    bundle: dict, *, include_diverged: bool = False
) -> tuple[list[dict], dict[str, dict]]:
    """(result payloads, feature_schema) from an exported evidence bundle.
    Callers who need custody assurance run `verify_bundle` first (the CLI does).

    D19 (ratified 2026-07-03): `additional_results` rows with basis
    `observation` (observe-only extra replicas — first-class evidence) join by
    DEFAULT; `diverged`/`outlier` rows joined only on `include_diverged=True`
    (they failed the run's own integrity predicate — forensics, not the
    headline scalar)."""
    payloads = [
        r.get("payload") or {} for r in bundle.get("consensus_results") or [] if r.get("payload")
    ]
    for r in bundle.get("additional_results") or []:
        basis = r.get("integrity_basis")
        if not r.get("payload"):
            continue
        if basis == "observation" or (include_diverged and basis in ("diverged", "outlier")):
            payloads.append(r["payload"])
    schema = (bundle.get("manifest") or {}).get("feature_schema") or {}
    return payloads, schema


def diverged_by_key_from_bundle(bundle: dict, key_feature: str = "probe_id") -> dict[str, int]:
    """{probe_key: diverged-unit count} from the SIGNED predicate's
    diverged_units, joined to probes via the bundle's work-unit INPUT payloads.
    Firewall #1: a diverged result is valid, evidentiary data — its payload is
    not exportable (hashes only), but its existence is signed evidence that the
    run could not corroborate that unit. Empty when the attestation is absent
    or nothing diverged."""
    att = bundle.get("attestation")
    if not isinstance(att, dict):
        return {}
    try:
        from auspexai_tenant.evidence import _attestation_from_bundle

        diverged = _attestation_from_bundle(att).diverged_units or []
    except Exception:
        return {}
    unit_keys = {
        w.get("unit_id"): _get_path(w.get("payload") or {}, key_feature)
        for w in bundle.get("work_units") or []
    }
    out: dict[str, int] = {}
    for d in diverged:
        key = unit_keys.get(d.get("unit_id"))
        out[str(key) if key is not None else "(unknown)"] = (
            out.get(str(key) if key is not None else "(unknown)", 0) + 1
        )
    return out


def drift_benchmark_bundles(
    bundle: dict,
    reference_bundle: dict,
    *,
    key_feature: str = "probe_id",
    include_diverged: bool = False,
) -> DriftBenchmark:
    """Benchmark one exported bundle against a reference bundle. The feature
    schema (and therefore the ENVELOPE) is taken from the REFERENCE bundle —
    the reference defines the calibrated behavior being drifted FROM."""
    obs, obs_schema = observations_from_bundle(bundle, include_diverged=include_diverged)
    ref, ref_schema = observations_from_bundle(reference_bundle)
    schema = ref_schema or obs_schema
    report = drift_benchmark(obs, ref, schema, key_feature=key_feature)
    diverged = diverged_by_key_from_bundle(bundle, key_feature=key_feature)
    if diverged:
        report = DriftBenchmark(
            probes=report.probes,
            peak_eu=report.peak_eu,
            breadth=report.breadth,
            byte_divergence_rate=report.byte_divergence_rate,
            key_feature=report.key_feature,
            notes=(
                *report.notes,
                "diverged units are signed evidence the run could not corroborate "
                "those units (within-run disagreement) — payloads are not exported, "
                "so they cannot be EU-scored; the count is reported, never folded "
                "into the drift scalar",
            ),
            diverged_units_total=sum(diverged.values()),
            diverged_by_key=diverged,
        )
    empty_notes = tuple(
        f"the {label} bundle carries no consensus results — every unit diverged "
        "(0-receipt) or its payloads aged off; there is nothing custody-verified "
        "to score. (An observe-only run — reducer builtin_process_only — makes "
        "every replica consensus-grade and exportable.)"
        for label, rows in (("observation", obs), ("reference", ref))
        if not rows
    )
    if empty_notes:
        report = DriftBenchmark(
            probes=report.probes,
            peak_eu=report.peak_eu,
            breadth=report.breadth,
            byte_divergence_rate=report.byte_divergence_rate,
            key_feature=report.key_feature,
            notes=(*report.notes, *empty_notes),
            diverged_units_total=report.diverged_units_total,
            diverged_by_key=report.diverged_by_key,
        )
    if ref_schema and obs_schema and ref_schema != obs_schema:
        report = replace(
            report,
            notes=(
                *report.notes,
                "feature schemas differ between the bundles; the REFERENCE bundle's "
                "declared envelope was used",
            ),
        )
    return report


# ── self-baseline (the run scores against ITS OWN early rounds) ───────────────
# self_baseline_drift_design.md §3.1: drift is a model's change from its OWN
# established normal, not its distance to a fixed anchor model. The driver spends
# the first K rounds as a CALIBRATION phase (drift_driver.py baseline_rounds);
# here the same run's baseline rows (round < K) become the reference and its
# monitoring rows (round >= K) the observations — reference-agnostic
# drift_benchmark does the rest. K is read from the benchmark declaration (which
# records [driver].baseline_rounds at launch); it is hash-attested in the signed
# manifest via config_provenance.resolved_config_sha256. The round is parsed from
# the unit_id the driver stamps as "<prefix>-<probe_id>-r<round>".

_ROUND_RE = re.compile(r"-r(\d+)$")


def round_of_unit(unit_id: str | None) -> int | None:
    """The round index the driver encoded in a unit_id (`…-r<round>`), or None
    when the id doesn't carry one (a foreign/hand-authored unit)."""
    m = _ROUND_RE.search(unit_id or "")
    return int(m.group(1)) if m else None


def observations_with_round_from_bundle(
    bundle: dict, *, include_diverged: bool = False
) -> tuple[list[tuple[int | None, dict]], dict[str, dict]]:
    """Like `observations_from_bundle`, but pairs each payload with the ROUND
    parsed from its unit_id — what the self-baseline split needs. Same row
    selection (consensus + observation-basis additional_results; diverged only on
    opt-in), same schema source."""
    rows: list[tuple[int | None, dict]] = []
    for r in bundle.get("consensus_results") or []:
        if r.get("payload"):
            rows.append((round_of_unit(r.get("unit_id")), r["payload"]))
    for r in bundle.get("additional_results") or []:
        basis = r.get("integrity_basis")
        if not r.get("payload"):
            continue
        if basis == "observation" or (include_diverged and basis in ("diverged", "outlier")):
            rows.append((round_of_unit(r.get("unit_id")), r["payload"]))
    schema = (bundle.get("manifest") or {}).get("feature_schema") or {}
    return rows, schema


def split_self_baseline(
    rows: list[tuple[int | None, dict]], baseline_rounds: int
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Split (round, payload) rows at the K boundary the driver froze at:
    baseline = round < K, monitoring = round >= K. Rows whose round can't be
    parsed are dropped from both sides and counted (they can't be placed on the
    timeline). Returns (baseline_payloads, monitoring_payloads, stats)."""
    baseline: list[dict] = []
    monitoring: list[dict] = []
    unplaced = 0
    for rnd, payload in rows:
        if rnd is None:
            unplaced += 1
        elif rnd < baseline_rounds:
            baseline.append(payload)
        else:
            monitoring.append(payload)
    return (
        baseline,
        monitoring,
        {
            "baseline_rounds": baseline_rounds,
            "baseline_rows": len(baseline),
            "monitoring_rows": len(monitoring),
            "unplaced_rows": unplaced,
        },
    )


def drift_benchmark_self(
    bundle: dict,
    baseline_rounds: int,
    *,
    key_feature: str = "probe_id",
    include_diverged: bool = False,
) -> DriftBenchmark:
    """Self-baseline Drift Benchmark: score a run's MONITORING rounds (round >= K)
    against its OWN BASELINE rounds (round < K) — drift from this model's own
    normal (the Sentinel method), NOT distance to a fixed anchor model. The
    reference-agnostic `drift_benchmark` does the math unchanged; only the
    reference differs (the run's own early rows). The envelope is the run's own
    declared feature_schema.

    baseline_rounds must be > 0. A run that converged inside the baseline window
    (no monitoring rows) is, by construction, self-stable: reported as peak 0.0 EU
    / 0% breadth with an explanatory note (a deterministic model that never left
    its baseline — cf. the prototype's gpt-oss greedy = 0.00x self-drift)."""
    if baseline_rounds <= 0:
        raise ValueError("drift_benchmark_self requires baseline_rounds > 0 (a self-baseline run)")
    rows, schema = observations_with_round_from_bundle(bundle, include_diverged=include_diverged)
    baseline, monitoring, stats = split_self_baseline(rows, baseline_rounds)

    split_note = (
        f"self-baseline: reference = this run's first {baseline_rounds} round(s) "
        f"({stats['baseline_rows']} baseline rows); scored {stats['monitoring_rows']} "
        f"monitoring row(s) against them"
        + (
            f"; {stats['unplaced_rows']} row(s) had no parseable round"
            if stats["unplaced_rows"]
            else ""
        )
    )

    if not baseline:
        # No baseline rows at all — the split can't self-reference. Return an
        # unscored report that says why (rather than mislabeling it 0 drift).
        return DriftBenchmark(
            probes=(),
            peak_eu=None,
            breadth=None,
            byte_divergence_rate=None,
            key_feature=key_feature,
            notes=(
                split_note,
                "no baseline rows recovered (round < K) — nothing to self-reference; "
                "check that the run issued its baseline rounds and the unit_ids carry -r<round>",
            ),
        )
    if not monitoring:
        # Converged within the baseline window: the panel never produced a
        # post-baseline round, so there is no self-drift by construction.
        return DriftBenchmark(
            probes=(),
            peak_eu=0.0,
            breadth=0.0,
            byte_divergence_rate=0.0,
            key_feature=key_feature,
            notes=(
                split_note,
                "no monitoring rows (round >= K): the run converged within its baseline "
                "window — self-stable by construction (zero drift from its own baseline)",
            ),
        )

    report = drift_benchmark(monitoring, baseline, schema, key_feature=key_feature)
    report = replace(report, notes=(split_note, *report.notes))

    diverged = diverged_by_key_from_bundle(bundle, key_feature=key_feature)
    if diverged:
        report = replace(
            report,
            notes=(
                *report.notes,
                "diverged units are signed evidence the run could not corroborate those "
                "units (within-run disagreement); the count is reported, never folded into "
                "the drift scalar",
            ),
            diverged_units_total=sum(diverged.values()),
            diverged_by_key=diverged,
        )
    return report


def format_report(report: DriftBenchmark) -> str:
    """A terminal-readable rendering (the CLI's output)."""
    lines: list[str] = []
    peak = f"{report.peak_eu:.2f}" if report.peak_eu is not None else "n/a"
    breadth = (
        f"{report.breadth:.0%} of probes beyond envelope" if report.breadth is not None else "n/a"
    )
    overlay = (
        f"{report.byte_divergence_rate:.0%}" if report.byte_divergence_rate is not None else "n/a"
    )
    lines.append(f"drift benchmark: peak {peak} EU  ·  {breadth}  ·  byte-divergence {overlay}")
    if report.diverged_units_total:
        per_key = " · ".join(f"{k}: {n}" for k, n in sorted((report.diverged_by_key or {}).items()))
        lines.append(
            f"  ⚠ within-run divergence (signed evidence, not EU-scorable): "
            f"{report.diverged_units_total} unit(s) — {per_key}"
        )
    lines.append("  (EU = envelope units: 1.0 = the calibrated noise floor; <1 within noise)")
    for p in report.probes:
        ppeak = f"{p.peak_eu:.2f}" if p.peak_eu is not None else "n/a"
        flag = " ← beyond envelope" if p.beyond_envelope else ""
        byte = (
            f"  byte-div {p.byte_divergence_rate:.0%}" if p.byte_divergence_rate is not None else ""
        )
        lines.append(f"  {p.key}: peak {ppeak} EU{byte}{flag}")
        for f in p.features:
            eu = f"{f.eu:.2f}" if f.eu is not None else "n/a"
            mx = f" (max {f.eu_max:.2f})" if f.eu_max is not None else ""
            lines.append(
                f"    {f.feature} [{f.rule}]: {eu} EU{mx}  pairs={f.pairs}"
                + (f"  invalid_excluded={f.invalid_excluded}" if f.invalid_excluded else "")
            )
    for n in report.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def _load_bundle_file(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
