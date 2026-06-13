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

    def section(self, name: str) -> dict[str, Any]:
        """An arbitrary top-level table (executor / reducer / work_unit_source /
        …), or {} when absent. Used by the generic `experiment build`."""
        v = self.raw.get(name)
        return dict(v) if isinstance(v, dict) else {}

    @property
    def label(self) -> str | None:
        return self.experiment.get("label")

    @property
    def model_id(self) -> str | None:
        return self.experiment.get("model_id")

    @property
    def driver_entrypoint(self) -> str | None:
        return self.driver.get("entrypoint")

    def journal_path(self, label: str) -> Path:
        """The run-journal path: an explicit `[driver].journal`, or
        `<label>.journal` when it is unset / "auto"."""
        j = self.driver.get("journal")
        if j in (None, "", "auto"):
            return Path(f"{label}.journal")
        return Path(j)


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load `experiment.toml`. `path` may be the file, a directory containing
    it, or None (→ cwd). A missing file yields an empty config (every knob then
    has to be supplied by flag), never an error — the file is a convenience, not
    a requirement."""
    p = Path(path) if path is not None else Path.cwd()
    if p.is_dir():
        p = p / DEFAULT_CONFIG_NAME
    if not p.exists():
        return ExperimentConfig(source_path=None)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return ExperimentConfig(
        experiment=dict(data.get("experiment") or {}),
        driver=dict(data.get("driver") or {}),
        raw=dict(data),
        source_path=p,
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
    manifest: dict[str, Any] = {
        "schema_version": "0.1",
        "tenant_id": req(e, "tenant_id", "experiment"),
        "tenant_maintainer_contact": req(e, "contact", "experiment"),
        "experiment_id": label,
        "research_goal_paragraph": req(e, "research_goal", "experiment"),
        "models": [
            {
                "id": req(e, "model_id", "experiment"),
                "version": str(e.get("model_version", "1.0")),
                "local_weights_required": bool(e.get("local_weights_required", True)),
            }
        ],
        "prompt_set_characteristics": req(e, "prompt_characteristics", "experiment"),
        "sensitive_content_flags": list(e.get("sensitive_content_flags") or []),
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
    return manifest
