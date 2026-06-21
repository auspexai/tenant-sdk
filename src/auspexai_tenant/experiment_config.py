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

import os
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "experiment.toml"

# A `run`/`launch` that selected a [profiles.<name>] override-set exports it here
# so a driver factory that re-loads the config itself (e.g. Vigiles'
# drift_driver.build) picks up the SAME active profile with no extra wiring.
#
# INTERIM BRIDGE — not the end-state. The active profile is currently re-resolved
# independently at each consumer (CLI→manifest, driver→toml re-load), kept coherent
# by this env var. Future consumers (the coordinator's tier-gating, the sandboxed
# executor) can't read a client-side env var. The robust end-state is to resolve
# ONCE and bake the resolved profile into the submitted manifest, so every consumer
# reads that single artifact and this bridge is retired. Decision tracked in
# vigiles_onramp_phase3_design.md §6 item 6 (folded into Phase-3 increment 3).
PROFILE_ENV_VAR = "AUSPEXAI_PROFILE"


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
    def key_path(self) -> str | None:
        """`[experiment].key_path` — the tenant signing key. Set once here and
        the SDK signs with it, so `--key` need not be passed on every command."""
        return self.experiment.get("key_path")

    def journal_path(self, label: str) -> Path:
        """The run-journal path: an explicit `[driver].journal`, or
        `<label>.journal` when it is unset / "auto"."""
        j = self.driver.get("journal")
        if j in (None, "", "auto"):
            return Path(f"{label}.journal")
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

    `profile` selects a `[profiles.<name>]` override-set (see `_resolve_profile`);
    when None it falls back to the `AUSPEXAI_PROFILE` env var, so the active
    profile a `run`/`launch` selected propagates to a driver factory that re-loads
    the config itself (e.g. Vigiles' `drift_driver.build`) with no extra wiring.
    An EXPLICIT `profile=` against a missing file is a hard error (the caller asked
    for a profile that can't be honored); the env-var fallback degrades to empty."""
    explicit = profile is not None
    effective_profile = profile if explicit else (os.environ.get(PROFILE_ENV_VAR) or None)
    if path is not None:
        p = Path(path)
        if p.is_dir():
            p = p / DEFAULT_CONFIG_NAME
    else:
        found = _find_config_upward(Path.cwd())
        if found is None:
            if explicit and effective_profile:
                raise ValueError(
                    f"profile '{effective_profile}' requested but no experiment.toml was found"
                )
            return ExperimentConfig(source_path=None)
        p = found
    if not p.exists():
        if explicit and effective_profile:
            raise ValueError(
                f"profile '{effective_profile}' requested but no experiment.toml at {p}"
            )
        return ExperimentConfig(source_path=None)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    data, active_profile = _resolve_profile(data, effective_profile)
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
        "reducer": {"kind": req(reducer, "kind", "reducer")},
    }
    # Approver attestations (required by the schema when sensitive flags are set)
    # ride an [approver] / [[approver_attestations]] table when present.
    attestations = cfg.raw.get("approver_attestations")
    if attestations:
        manifest["approver_attestations"] = attestations

    # ── v0_2 members (all optional; emitted only when declared) ──────────────
    if e.get("requires_real_execution"):  # M2
        manifest["requires_real_execution"] = True
    det = cfg.section("determinism")  # M1
    if det:
        manifest["inference_determinism"] = {
            k: det[k]
            for k in ("temperature", "seed", "serving_version_pin", "hardware_class")
            if det.get(k) is not None
        }
    out = cfg.section("output")  # M4
    if out.get("measurement_level"):
        manifest["output_schema"] = {
            k: out[k] for k in ("measurement_level", "shape", "dtype") if out.get(k) is not None
        }

    # schema_version bumps to 0.2 exactly when a v0_2 member is present, so an
    # existing config (none declared) stays a valid 0.1 manifest unchanged.
    uses_v0_2 = (
        "expected_gguf_sha256" in model
        or "requires_real_execution" in manifest
        or "inference_determinism" in manifest
        or "output_schema" in manifest
    )
    manifest["schema_version"] = "0.2" if uses_v0_2 else "0.1"
    return manifest
