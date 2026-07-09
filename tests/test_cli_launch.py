"""experiment launch — pkg/config resolution (v0.6.7).

`launch` resolves PKG_DIR from the experiment.toml it walks up to find, so a bare
`auspexai-tenant experiment launch` works from anywhere in the repo (the earlier
default of the literal `pkg` failed from any other directory). These cover the
two resolution guards.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

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


def test_wait_for_approval_uses_doorbell_wake() -> None:
    """D22-A: with a --doorbell wake, `_wait_for_approval` reacts via `wake.wait()`
    (the SSE approval event) instead of the poll floor. poll_interval=999 would
    hang the test if it slept, so passing proves the wake path is taken."""

    class _Wake:
        def __init__(self) -> None:
            self.waits = 0

        def wait(self) -> None:
            self.waits += 1

    wake = _Wake()
    c = _FakeClient(["submitted", "submitted", "approved"])
    assert _wait_for_approval(c, "exp-x", poll_interval=999, wake=wake) == "approved"
    assert wake.waits == 2  # doorbell-driven, not the 999s poll floor


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


# ── D14 tail: Ctrl-C in the submit→approval window (v0.6.25) ─────────────────
#
# A Ctrl-C while `launch` waits for maintainer approval previously exited with
# the experiment still submitted server-side — it then auto-approved as a
# driverless orphan (exp-oK4PrkRP). The launch handler now mirrors the
# drive-loop D14 semantics: abort by default; --resumable leaves it submitted.


class _AbortRecorder:
    aborted: ClassVar[list[str]] = []

    def __init__(self, coordinator: str, key, experiment_id: str) -> None:
        self._id = experiment_id

    def abort(self) -> dict:
        _AbortRecorder.aborted.append(self._id)
        return {"status": "aborted"}


def _launch_repo(tmp_path: Path) -> None:
    (tmp_path / "experiment.toml").write_text(
        '[experiment]\nlabel = "x"\n[driver]\nentrypoint = "d:b"\n'
    )
    (tmp_path / "pkg").mkdir()


def _patch_launch_through_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the build/submit/client legs so `launch` reaches the approval wait."""
    from click.testing import CliRunner as _  # noqa: F401  (keep import local pattern)

    import auspexai_tenant.cli as cli_mod

    monkeypatch.setattr(cli_mod.experiment_build, "callback", lambda **kw: None)
    monkeypatch.setattr(cli_mod.experiment_submit, "callback", lambda **kw: None)
    monkeypatch.setattr(cli_mod, "_make_client", lambda coordinator, key_path: object())
    monkeypatch.setattr(cli_mod, "_resolve_experiment", lambda client, target: ("exp-1", "x"))
    monkeypatch.setattr(cli_mod, "_load_key", lambda key_path: object())


def test_launch_ctrl_c_during_approval_wait_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auspexai_tenant.cli as cli_mod
    import auspexai_tenant.experiment as experiment_mod

    _launch_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_launch_through_submit(monkeypatch)

    def _interrupted_wait(client, experiment_id, poll_interval, *, resumable=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "_wait_for_approval", _interrupted_wait)
    _AbortRecorder.aborted = []
    monkeypatch.setattr(experiment_mod, "Experiment", _AbortRecorder)

    r = CliRunner().invoke(main, ["experiment", "launch"], standalone_mode=False)
    assert isinstance(r.exception, SystemExit) and r.exception.code == 130
    assert _AbortRecorder.aborted == ["exp-1"]  # the orphan is aborted, not left


def test_launch_ctrl_c_resumable_leaves_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auspexai_tenant.cli as cli_mod
    import auspexai_tenant.experiment as experiment_mod

    _launch_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_launch_through_submit(monkeypatch)

    def _interrupted_wait(client, experiment_id, poll_interval, *, resumable=False):
        assert resumable is True  # the flag reaches the wait's announced hint
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "_wait_for_approval", _interrupted_wait)
    _AbortRecorder.aborted = []
    monkeypatch.setattr(experiment_mod, "Experiment", _AbortRecorder)

    r = CliRunner().invoke(main, ["experiment", "launch", "--resumable"], standalone_mode=False)
    assert isinstance(r.exception, SystemExit) and r.exception.code == 130
    assert _AbortRecorder.aborted == []  # left submitted server-side by request


def test_launch_resolves_its_own_stamped_label_not_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrency fix: launch drives the experiment IT submitted — resolved by the
    unique label the build stamped into the manifest — NOT `latest`, so concurrent
    --detach launches don't all collapse onto the last-submitted experiment."""
    import json

    import auspexai_tenant.cli as cli_mod

    _launch_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _fake_build(**kw):  # write a manifest carrying THIS launch's stamped label
        kw["out_path"].write_text(json.dumps({"experiment_id": "vig-mine-20260709-000001"}))

    monkeypatch.setattr(cli_mod.experiment_build, "callback", _fake_build)
    monkeypatch.setattr(cli_mod.experiment_submit, "callback", lambda **kw: None)
    monkeypatch.setattr(cli_mod, "_make_client", lambda coordinator, key_path: object())
    monkeypatch.setattr(cli_mod, "_load_key", lambda key_path: object())
    monkeypatch.setattr(cli_mod, "_record_benchmark_declaration", lambda *a, **k: None)

    seen = {}

    def _capture_resolve(client, target):
        seen["target"] = target
        return ("exp-mine", "vig-mine-20260709-000001")

    monkeypatch.setattr(cli_mod, "_resolve_experiment", _capture_resolve)
    # --no-drive returns right after the resolve, which is all we're asserting on.
    r = CliRunner().invoke(main, ["experiment", "launch", "--no-drive"], standalone_mode=False)
    assert r.exit_code == 0, r.output
    assert seen["target"] == "vig-mine-20260709-000001"  # its OWN label, not "latest"
