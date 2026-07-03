"""Local run-directory layout — the single source of truth for where a run's
local artifacts (evidence bundle, run journal, results) are written.

Both the CLI (`experiment export`, the driver's journal default) and the
researcher dashboard's download button resolve paths through here, so a download
is organized identically however it is triggered
(researcher_onboarding_and_rd_surfaces_design.md §8).

Layout: ``<base>/<label>/{bundle.json, results.json, run.journal}`` — one
directory per run, keyed by the tenant **label** (the researcher-authored,
forever-unique name = ``manifest.experiment_id``; the coordinator experiment id
maps to it via ``_resolve_experiment``). Base precedence: an explicit configured
dir (``experiment.toml`` ``[runs].dir``) > ``$AUSPEXAI_RUNS_DIR`` > ``./runs``
**when it already exists** (repo-local workflow compat) > the stable per-user
default ``~/.local/share/auspexai-tenant/runs``.

The stable default exists because a cwd-relative base is unreadable by any
OTHER process (2026-07-03, live: `launch` wrote a benchmark beside the run in
``<cwd>/runs`` and the dashboard — scanning its own base — reported "no
benchmark" for a scored run). Services must never depend on a CLI's cwd; a
tenant that wants a repo-local base declares ``[runs].dir`` explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RUNS_DIRNAME = "runs"
RUNS_DIR_ENV = "AUSPEXAI_RUNS_DIR"


def runs_base(configured: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the runs base directory.

    Precedence: an explicit ``configured`` value (e.g. ``experiment.toml``
    ``[runs].dir``) > ``$AUSPEXAI_RUNS_DIR`` > ``./runs``. The result is the
    *base*; per-run paths nest a ``<label>/`` directory beneath it.
    """
    if configured:
        return Path(configured).expanduser()
    env = os.environ.get(RUNS_DIR_ENV)
    if env:
        return Path(env).expanduser()
    cwd_runs = Path(DEFAULT_RUNS_DIRNAME)
    if cwd_runs.is_dir():
        return cwd_runs  # repo-local compat: an existing ./runs keeps working
    return stable_runs_base()


def stable_runs_base() -> Path:
    """The cwd-independent per-user base — what a fresh context resolves to,
    and what OTHER processes (the dashboard) can always find."""
    return Path.home() / ".local" / "share" / "auspexai-tenant" / "runs"


class RunLayout:
    """Per-run artifact paths under ``<base>/<label>/``.

    Construct with the tenant label; pass ``base`` to override the resolved
    default. The label must be a single safe path component (it becomes a
    directory name) — a label with a separator or traversal token is rejected
    rather than silently nesting or escaping the base.
    """

    def __init__(self, label: str, *, base: str | os.PathLike[str] | None = None) -> None:
        if not label:
            raise ValueError("RunLayout requires a non-empty label")
        if "/" in label or "\\" in label or label in (".", ".."):
            raise ValueError(f"unsafe run label for a directory name: {label!r}")
        self.label = label
        self.base = Path(base) if base is not None else runs_base()
        self.dir = self.base / label

    def bundle_path(self) -> Path:
        return self.dir / "bundle.json"

    def results_path(self) -> Path:
        return self.dir / "results.json"

    def journal_path(self) -> Path:
        return self.dir / "run.journal"

    def ensure(self) -> RunLayout:
        """Create the run directory (parents included) if absent; return self."""
        self.dir.mkdir(parents=True, exist_ok=True)
        return self
