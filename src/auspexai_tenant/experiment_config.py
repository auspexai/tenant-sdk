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
    source_path: Path | None = None

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
