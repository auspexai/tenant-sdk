"""Unit tests for the shared run-directory layout (`auspexai_tenant.runs`).

The layout is the single source of truth the CLI and the dashboard both use, so
a download is organized identically however it is triggered. Covers base
resolution (precedence), per-run paths, dir creation, and label safety.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_tenant.runs import RunLayout, runs_base


def test_runs_base_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No config, no env: an EXISTING ./runs keeps the repo-local workflow; a
    # fresh context resolves the stable per-user base (cwd-relative bases are
    # unreadable by other processes — the 2026-07-03 dashboard lesson).
    from auspexai_tenant.runs import stable_runs_base

    monkeypatch.delenv("AUSPEXAI_RUNS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert runs_base() == stable_runs_base()
    (tmp_path / "runs").mkdir()
    assert runs_base() == Path("runs")


def test_runs_base_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", "/data/auspexai")
    assert runs_base() == Path("/data/auspexai")


def test_runs_base_configured_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit configured dir (e.g. experiment.toml [runs].dir) beats the env.
    monkeypatch.setenv("AUSPEXAI_RUNS_DIR", "/from/env")
    assert runs_base("/from/toml") == Path("/from/toml")


def test_runs_base_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUSPEXAI_RUNS_DIR", raising=False)
    assert runs_base("~/runs") == Path.home() / "runs"


def test_layout_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUSPEXAI_RUNS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()  # repo-local workflow
    lay = RunLayout("drift-a1")
    assert lay.dir == Path("runs/drift-a1")
    assert lay.bundle_path() == Path("runs/drift-a1/bundle.json")
    assert lay.results_path() == Path("runs/drift-a1/results.json")
    assert lay.journal_path() == Path("runs/drift-a1/run.journal")


def test_layout_explicit_base() -> None:
    lay = RunLayout("vig-2", base="/tmp/x")
    assert lay.bundle_path() == Path("/tmp/x/vig-2/bundle.json")


def test_layout_ensure_creates_dir(tmp_path: Path) -> None:
    lay = RunLayout("vig-3", base=tmp_path).ensure()
    assert lay.dir.is_dir()
    # idempotent — a second ensure() on an existing dir is fine.
    assert lay.ensure().dir.is_dir()


def test_layout_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RunLayout("")


@pytest.mark.parametrize("bad", ["a/b", "..", ".", "x\\y"])
def test_layout_rejects_unsafe_label(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        RunLayout(bad)
