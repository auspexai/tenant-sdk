"""experiment launch — pkg/config resolution (v0.6.7).

`launch` resolves PKG_DIR from the experiment.toml it walks up to find, so a bare
`auspexai-tenant experiment launch` works from anywhere in the repo (the earlier
default of the literal `pkg` failed from any other directory). These cover the
two resolution guards.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from auspexai_tenant.cli import _wait_for_approval, main


class _FakeClient:
    """Returns the next queued status on each get_experiment call."""

    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)

    def get_experiment(self, experiment_id: str) -> dict[str, str]:
        return {"status": self._statuses.pop(0)}


def test_wait_for_approval_returns_immediately_when_approved() -> None:
    assert _wait_for_approval(_FakeClient(["approved"]), "exp-x", poll_interval=0) == "approved"


def test_wait_for_approval_polls_through_submitted() -> None:
    # submitted twice (the 409 state), then approved → it waits, then proceeds
    c = _FakeClient(["submitted", "submitted", "approved"])
    assert _wait_for_approval(c, "exp-x", poll_interval=0) == "approved"


def test_wait_for_approval_exits_on_terminal_status() -> None:
    with pytest.raises(SystemExit) as ei:
        _wait_for_approval(_FakeClient(["aborted"]), "exp-x", poll_interval=0)
    assert ei.value.code == 1


def test_launch_no_experiment_toml_is_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    r = CliRunner().invoke(main, ["experiment", "launch"])
    assert r.exit_code == 1
    assert "no experiment.toml found" in r.output


def test_launch_missing_pkg_dir_is_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "experiment.toml").write_text('[experiment]\nlabel = "x"\n')  # no pkg/ beside it
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(main, ["experiment", "launch"])
    assert r.exit_code == 1
    assert "package dir not found" in r.output


def test_launch_resolves_pkg_next_to_toml_from_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fix: from a subdir, `launch` finds pkg/ next to experiment.toml — it gets
    PAST resolution (no pkg/config error) and on to the build step."""
    (tmp_path / "experiment.toml").write_text(
        '[experiment]\nlabel = "x"\n[driver]\nentrypoint = "d:b"\n'
    )
    (tmp_path / "pkg").mkdir()
    sub = tmp_path / "driver"
    sub.mkdir()
    monkeypatch.chdir(sub)
    r = CliRunner().invoke(main, ["experiment", "launch"])
    # Resolution succeeded — neither guard fired (it failed later, on build/submit).
    assert "no experiment.toml found" not in r.output
    assert "package dir not found" not in r.output
