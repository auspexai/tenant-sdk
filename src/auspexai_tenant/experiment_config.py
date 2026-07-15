"""`experiment.toml` — the researcher's experiment knobs in one checked-in file.

Stand-up becomes `experiment submit pkg/ && experiment run latest` with no
env-var one-liners for a terminal/clipboard to mangle
(`researcher_experiment_lifecycle_and_ergonomics_design.md` §3). Two tables:

  - `[experiment]` feeds the manifest BUILD (the tenant's build.py reads it via
    `load_experiment_config`): label, model_id, integrity_policy, replication,
    caps. Per the ratified approach-B split, the build stamps a UNIQUE label
    (`make_unique_label`) into the manifest, so `experiment submit` stays a pure
    courier that signs + ships exactly what was built.
  - `[driver]` feeds `experiment run`: the driver entrypoint, the journal path,
    and arbitrary opaque pass-through knobs (cadence, duration, …) the tenant's
    driver factory consumes. Opaque pass-through ⇒ non-vigiles tenants reuse the
    file unchanged.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "experiment.toml"


@dataclass(frozen=True)
class ExperimentConfig:
    """Parsed `experiment.toml`. Missing tables read as empty dicts so callers
    fall back to flags/defaults rather than crashing on a partial file."""

    experiment: dict[str, Any] = field(default_factory=dict)
    driver: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)  # the whole parsed file
    source_path: Path | None = None
    active_profile: str | None = None  # the [profiles.<name>] applied, if any

    def section(self, name: str) -> dict[str, Any]:
        """An arbitrary top-level table (executor / reducer / work_unit_source /
        …), or {} when absent. Used by the generic `experiment build`."""
        v = self.raw.get(name)
        return dict(v) if isinstance(v, dict) else {}

    @property
    def available_profiles(self) -> list[str]:
        """Names of the [profiles.<name>] override-sets declared in the file."""
        p = self.raw.get("profiles")
        return sorted(p) if isinstance(p, dict) else []

    @property
    def label(self) -> str | None:
        return self.experiment.get("label")

    @property
    def model_id(self) -> str | None:
        return self.experiment.get("model_id")

    @property
    def driver_entrypoint(self) -> str | None:
        return self.driver.get("entrypoint")

    @property
    def driver_path(self) -> str | None:
        """`[driver].path` — a directory (relative to `experiment.toml`) prepended
        to `sys.path` before importing the entrypoint, so a driver module that
        lives in a subdir (e.g. `driver/`) resolves regardless of the invocation
        cwd. Lets `run`/`launch` work from the repo root with no `cd`."""
        return self.driver.get("path")

    @property
    def capture_raw(self) -> bool:
        """`[capture] raw` — D20 opt-in to raw model-output collection (R3-only,
        live). One line, profile-inheritable. Absent = today's behavior (raw
        never leaves the executor). Declared pre-run → feeds admission via
        sensitive_content_flags; enables the executor to emit `raw_response`."""
        c = self.raw.get("capture")
        return bool(c.get("raw")) if isinstance(c, dict) else False

    @property
    def benchmark_mode(self) -> str:
        """`[benchmark].mode` — how this run's Drift Benchmark chooses its
        reference. "fixed_reference" (default): score against another experiment's
        signed baseline (`reference`). "self_baseline": score this run's
        monitoring rounds against its OWN first-K baseline rounds — drift from the
        model's own normal, not distance to an anchor model
        (self_baseline_drift_design.md §3.1). `reference = "self"` is a
        convenience alias for `mode = "self_baseline"`."""
        b = self.raw.get("benchmark")
        if not isinstance(b, dict):
            return "fixed_reference"
        m = b.get("mode")
        if m is None:
            return (
                "self_baseline"
                if str(b.get("reference") or "").strip() == "self"
                else "fixed_reference"
            )
        if m not in ("fixed_reference", "self_baseline"):
            raise ValueError(
                f"[benchmark].mode must be 'fixed_reference' or 'self_baseline', got {m!r}"
            )
        return m

    @property
    def benchmark_reference(self) -> str | None:
        """`[benchmark].reference` — for fixed_reference mode, the experiment this
        run's Drift Benchmark scores against (its signed manifest defines the
        envelope). TENANT-GENERIC declaration, platform-fixed semantics (the
        feature_schema pattern): tenants declare WHAT to score against; the
        envelope-units standard is never per-tenant. Declared up front = the
        pre-registration posture (the comparison is chosen before the data
        exists); the launch flow records it beside the run so the dashboard
        materializes the score automatically. None in self_baseline mode (the
        run's own baseline is the reference — see `benchmark_baseline_rounds`)."""
        b = self.raw.get("benchmark")
        if not isinstance(b, dict):
            return None
        unknown = set(b) - {"reference", "mode"}
        if unknown:
            raise ValueError(
                f"[benchmark] has unknown key(s) {sorted(unknown)} — "
                "only 'reference' and 'mode' are defined"
            )
        if self.benchmark_mode == "self_baseline":
            return None  # the reference is this run's own baseline rounds
        ref = b.get("reference")
        if ref is None:
            return None
        if not isinstance(ref, str):
            raise ValueError("[benchmark].reference must be a string")
        r = ref.strip()
        # "" is the explicit opt-out — how a profile (e.g. the calibration
        # baseline itself) switches OFF a top-level declaration it inherits.
        # "self" is the self_baseline alias (handled by benchmark_mode) → no ext ref.
        return None if r == "self" else (r or None)

    @property
    def benchmark_baseline_rounds(self) -> int:
        """The self-baseline window K the scorer resolves the reference from — the
        SAME `[driver].baseline_rounds` the driver calibrates over, clamped to
        max_rounds like the driver does. 0 when unset. Recorded in the benchmark
        declaration at launch (plaintext, pre-data); the value is hash-attested in
        the signed manifest via config_provenance.resolved_config_sha256."""
        v = self.driver.get("baseline_rounds")
        try:
            k = max(0, int(v)) if v is not None else 0
        except (TypeError, ValueError):
            return 0
        mx = self.driver.get("max_rounds")
        try:
            if mx is not None:
                k = min(k, max(0, int(mx)))
        except (TypeError, ValueError):
            pass
        return k

    @property
    def key_path(self) -> str | None:
        """`[experiment].key_path` — the tenant signing key. Set once here and
        the SDK signs with it, so `--key` need not be passed on every command."""
        return self.experiment.get("key_path")

    @property
    def runs_dir(self) -> str | None:
        """`[runs].dir` — the base directory for per-run local artifacts (the
        evidence bundle, journal, results). Unset ⇒ the SDK default
        (`$AUSPEXAI_RUNS_DIR`, else `./runs`). See `auspexai_tenant.runs`."""
        r = self.raw.get("runs")
        return r.get("dir") if isinstance(r, dict) else None

    def journal_path(self, label: str) -> Path:
        """The run-journal path: an explicit `[driver].journal` wins; otherwise
        the shared per-run layout `<runs_base>/<label>/run.journal`, so the
        journal lands beside its evidence bundle instead of in the CWD root."""
        j = self.driver.get("journal")
        if j in (None, "", "auto"):
            from auspexai_tenant.runs import RunLayout, runs_base

            return RunLayout(label, base=runs_base(self.runs_dir)).journal_path()
        return Path(j)


def _find_config_upward(start: Path) -> Path | None:
    """Walk up from `start` to the filesystem root, returning the first
    `experiment.toml`. Lets `run`/`launch` work from a subdir (e.g. `driver/`)
    with no `--config` — the config is found at the repo root just like git
    finds `.git`."""
    for d in (start, *start.parents):
        candidate = d / DEFAULT_CONFIG_NAME
        if candidate.exists():
            return candidate
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto a copy of `base`: dict values merge
    key-by-key; every other value (scalars, lists) replaces wholesale — so a
    profile's `sensitive_content_flags = []` CLEARS the list rather than
    appending, and a scalar override wins outright."""
    out = dict(base)
    for k, v in override.items():
        cur = out.get(k)
        if isinstance(v, dict) and isinstance(cur, dict):
            out[k] = _deep_merge(cur, v)
        else:
            out[k] = v
    return out


def _resolve_profile(
    data: dict[str, Any], profile: str | None
) -> tuple[dict[str, Any], str | None]:
    """Apply a `[profiles.<name>]` override-set to the parsed toml `data`.

    The active profile is `profile`, else `[default_profile]`, else none (the base
    config is returned unchanged — fully backward-compatible). A profile's
    TABLE-valued keys are section overrides, deep-merged onto the matching
    top-level table (`[profiles.starter.driver]` → `[driver]`); its scalar-valued
    keys (`description`, and forward-compat markers) are profile metadata, ignored
    by the merge.

    Phase-3 note: this is the override MECHANISM only. Which sections a *certified*
    starter profile may touch (the locked executor / reducer / probe-panel) and
    the tier-gated selection are enforced later (`vigiles_onramp_phase3_design.md`,
    Phase-3 increments 3+) — not here.
    """
    selected = profile if profile is not None else data.get("default_profile")
    if not selected:
        return data, None
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or selected not in profiles:
        available = sorted(profiles) if isinstance(profiles, dict) else []
        hint = (
            f" (available: {', '.join(available)})" if available else " (no [profiles.*] defined)"
        )
        raise ValueError(f"experiment.toml has no profile '{selected}'{hint}")
    block = profiles[selected]
    if not isinstance(block, dict):
        raise ValueError(f"experiment.toml [profiles.{selected}] is not a table")
    merged = dict(data)
    for k, v in block.items():
        if isinstance(v, dict):  # a section override → deep-merge onto the base table
            cur = merged.get(k)
            merged[k] = _deep_merge(cur if isinstance(cur, dict) else {}, v)
        # scalar profile keys (description / certified / …) are metadata, not merged
    return merged, str(selected)


def load_experiment_config(
    path: str | Path | None = None, *, profile: str | None = None
) -> ExperimentConfig:
    """Load `experiment.toml`. `path` may be the file, a directory containing
    it, or None (→ walk up from cwd to find it). A missing file yields an empty
    config (every knob then has to be supplied by flag), never an error — the
    file is a convenience, not a requirement.

    `profile` selects a `[profiles.<name>]` override-set (see `_resolve_profile`).
    The resolved config is passed to the driver factory by the caller (`run`/
    `launch` → `factory(cfg)`), so the active profile reaches the driver as an
    argument — never re-loaded from disk or carried by a side-channel. An EXPLICIT
    `profile=` against a missing file is a hard error (the caller asked for a
    profile that can't be honored)."""
    if path is not None:
        p = Path(path)
        if p.is_dir():
            p = p / DEFAULT_CONFIG_NAME
    else:
        found = _find_config_upward(Path.cwd())
        if found is None:
            if profile:
                raise ValueError(f"profile '{profile}' requested but no experiment.toml was found")
            return ExperimentConfig(source_path=None)
        p = found
    if not p.exists():
        if profile:
            raise ValueError(f"profile '{profile}' requested but no experiment.toml at {p}")
        return ExperimentConfig(source_path=None)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    data, active_profile = _resolve_profile(data, profile)
    return ExperimentConfig(
        experiment=dict(data.get("experiment") or {}),
        driver=dict(data.get("driver") or {}),
        raw=dict(data),
        source_path=p,
        active_profile=active_profile,
    )


def make_unique_label(base: str, *, now: datetime | None = None) -> str:
    """Append a UTC timestamp suffix so a re-built experiment never collides
    with a prior label (coordinator labels are unique forever, incl. aborted).
    The suffix is digits + hyphens, so `<base>-<suffix>` still matches the
    manifest `experiment_id` pattern `^[a-z][a-z0-9-]{2,127}$` for any valid
    base. Pass `now` for deterministic tests."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{stamp}"


def manifest_dict_from_config(
    cfg: ExperimentConfig, *, package_sha256: str, label: str
) -> dict[str, Any]:
    """Assemble a v0.1 manifest dict from experiment.toml — the generic build
    that `experiment build` writes (replacing per-tenant build.py). The caller
    computes `package_sha256` over the package dir and resolves the (suffixed)
    `label`; this maps the config onto the manifest schema. A missing required
    field raises ValueError naming the table + key. The result is NOT validated
    here — the caller runs `Manifest.model_validate` so schema errors surface in
    one place."""
    e = cfg.experiment

    def req(table: dict[str, Any], key: str, where: str) -> Any:
        v = table.get(key)
        if v in (None, ""):
            raise ValueError(f"experiment.toml [{where}] is missing required '{key}'")
        return v

    executor = cfg.section("executor")
    reducer = cfg.section("reducer")
    wus = cfg.section("work_unit_source")

    model: dict[str, Any] = {
        "id": req(e, "model_id", "experiment"),
        "version": str(e.get("model_version", "1.0")),
        "local_weights_required": bool(e.get("local_weights_required", True)),
    }
    if e.get("expected_gguf_sha256"):  # v0_2 / M3 (#13b): pin the served weights
        model["expected_gguf_sha256"] = e["expected_gguf_sha256"]
    # M3 lazy auto-acquire (#39/#44): where a worker pulls a missing model from.
    # Without these a `local_weights_required` model only routes to workers that
    # already hold it; with them an auto_acquire worker fetches it in-line.
    if e.get("hf_repo"):
        model["hf_repo"] = e["hf_repo"]
    if e.get("hf_filename"):
        model["hf_filename"] = e["hf_filename"]

    manifest: dict[str, Any] = {
        "tenant_id": req(e, "tenant_id", "experiment"),
        "tenant_maintainer_contact": req(e, "contact", "experiment"),
        "experiment_id": label,
        "research_goal_paragraph": req(e, "research_goal", "experiment"),
        "models": [model],
        "prompt_set_characteristics": req(e, "prompt_characteristics", "experiment"),
        "sensitive_content_flags": list(e.get("sensitive_content_flags") or []),
        # §9 #48: the declared research class (optional; omitted ⇒ human review).
        **({"research_class": e["research_class"]} if e.get("research_class") else {}),
        "expected_duration_hours": float(e.get("duration_hours", 1)),
        "replication_factor": int(e.get("replication", 1)),
        "work_unit_source": {
            "kind": wus.get("kind", "static"),  # driver-fed default
            "tarball_sha256": wus.get("tarball_sha256", "0" * 64),
        },
        "executor": {
            "command": list(req(executor, "command", "executor")),
            "package_sha256": package_sha256,
        },
        "reducer": {
            "kind": req(reducer, "kind", "reducer"),
            # C7: a within_cell_tolerance reducer may name the predicate feature
            # subset; passed through when declared (omit ⇒ all comparison'd features).
            **(
                {"tolerance_features": list(reducer["tolerance_features"])}
                if reducer.get("tolerance_features")
                else {}
            ),
            # A custom reducer (the non-agreement fold a seeded-sampling experiment
            # declares — §3c) carries its own argv + optional image digest.
            **({"command": list(reducer["command"])} if reducer.get("command") else {}),
            **({"image_sha256": reducer["image_sha256"]} if reducer.get("image_sha256") else {}),
        },
    }
    # Approver attestations (required by the schema when sensitive flags are set)
    # ride an [approver] / [[approver_attestations]] table when present.
    attestations = cfg.raw.get("approver_attestations")
    if attestations:
        manifest["approver_attestations"] = attestations

    # ── v0_2 members (all optional; emitted only when declared) ──────────────
    if e.get("requires_real_execution"):  # M2
        manifest["requires_real_execution"] = True
    det = cfg.section("determinism")  # M1 (+ the v0_5 sampling whitelist)
    if det:
        manifest["inference_determinism"] = {
            k: det[k]
            for k in (
                "temperature",
                "seed",
                "serving_version_pin",
                "hardware_class",
                "top_p",
                "top_k",
                "min_p",
            )
            if det.get(k) is not None
        }
    out = cfg.section("output")  # M4
    if out.get("measurement_level"):
        manifest["output_schema"] = {
            k: out[k] for k in ("measurement_level", "shape", "dtype") if out.get(k) is not None
        }

    # ── v0_3 / D16.1: the self-describing feature schema ─────────────────────
    # The toml [feature_schema."<path>"] tables map straight onto the manifest
    # member (the toml structure mirrors FeatureDeclaration); Manifest.model_validate
    # (run by the caller) does the per-kind §7 validation.
    fs = cfg.raw.get("feature_schema")
    if fs:
        manifest["feature_schema"] = fs

    # ── v0_3 / D16.2: the pre-registered design ──────────────────────────────
    # The toml [pre_registration] table maps straight onto the manifest member
    # (mirrors PreRegistration); Manifest.model_validate checks the block
    # references declared features carrying a comparison. The tenant signature +
    # the coordinator's submit-time Rekor anchor make `design ≺ data` provable.
    pr = cfg.raw.get("pre_registration")
    if pr:
        manifest["pre_registration"] = pr

    # ── v0_6 / D17: config provenance — ALWAYS stamped at build (ratified Q1).
    # HOW this experiment was configured: the active profile, a fingerprint of
    # the RESOLVED (post-profile-merge) config (the driver knobs are inside the
    # hash even though they never become manifest fields), and the tenant repo's
    # commit + dirty bit (best-effort; absent outside a git tree). Descriptive,
    # never enforced — its readers are the D16.2 verify chain/bundle + the
    # dashboard. Every toml-built manifest is therefore v0.6+.
    # D20: raw-content capture opt-in (ratified 2026-07-06) — declared pre-run so
    # the coordinator permits the reserved `raw_response` key at ingest, and the
    # admission assessment sees it via sensitive_content_flags.
    if cfg.capture_raw:
        manifest["capture"] = {"raw": True}
        flags = manifest.get("sensitive_content_flags") or []
        if "raw_content_capture" not in flags:
            flags = [*flags, "raw_content_capture"]
        manifest["sensitive_content_flags"] = flags

    manifest["config_provenance"] = _config_provenance(cfg)

    # schema_version bumps to the lowest version that covers the declared
    # members. Provenance stamping (above) floors every BUILT manifest at 0.6;
    # hand-written manifests at older versions stay valid forever (the model).
    manifest["schema_version"] = "0.6"
    return manifest


def _config_provenance(cfg: ExperimentConfig) -> dict[str, Any]:
    """The D17 provenance block. Pure over `cfg.raw` (already profile-resolved
    by `load_experiment_config`); git fields are best-effort and never fail the
    build (no git / not a repo / git errors ⇒ fields absent)."""
    import hashlib
    import subprocess

    canonical = json.dumps(cfg.raw, sort_keys=True, separators=(",", ":"), default=str)
    out: dict[str, Any] = {
        "resolved_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    if cfg.active_profile:
        out["profile"] = cfg.active_profile
    if cfg.source_path is not None:
        repo_dir = str(cfg.source_path.parent)
        try:
            commit = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            if commit:
                out["git_commit"] = commit
                status = subprocess.run(
                    ["git", "-C", repo_dir, "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout
                out["git_dirty"] = bool(status.strip())
        except (OSError, subprocess.SubprocessError):
            pass  # not a repo / no git — provenance stays config-hash-only
    return out
