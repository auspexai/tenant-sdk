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
# driverless orphan (exp-oK4PrkRP). At the approval-wait stage no work units
# exist yet, so the handler aborts by default (--resumable leaves it submitted).
# (The drive-loop handler differs — it FINALIZES to keep completed work.)


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


class _LifecycleRecorder:
    """Records finalize/abort so a drive-loop interrupt test can assert the run's
    completed work was KEPT (finalized), not thrown away (aborted)."""

    calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, coordinator: str, key, experiment_id: str) -> None:
        self._id = experiment_id

    def finalize(self) -> dict:
        _LifecycleRecorder.calls.append(("finalize", self._id))
        return {"status": "approved"}

    def abort(self) -> dict:
        _LifecycleRecorder.calls.append(("abort", self._id))
        return {"status": "aborted"}

    def driver_heartbeat(self, status, *, reason=None, round=None, run_id=None) -> None:
        pass  # best-effort telemetry — not part of the finalize/abort assertion


def test_run_ctrl_c_finalizes_completed_work_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C during the drive loop FINALIZES the run (keeps the units already
    completed) rather than aborting it — completed, consensus-reached work is never
    discarded on an interrupt. The coordinator then wraps up the finalized run."""
    from types import SimpleNamespace

    import auspexai_tenant.cli as cli_mod
    import auspexai_tenant.driver as driver_mod
    import auspexai_tenant.experiment as experiment_mod
    import auspexai_tenant.experiment_config as ec_mod
    from auspexai_tenant import Counter
    from auspexai_tenant.driver import DriverSpec

    spec = DriverSpec(
        condition=lambda agg: False,
        next_batch=lambda agg, rnd: None,
        reduce=Counter(bucket=lambda r: "x"),
    )
    cfg = SimpleNamespace(
        driver_path=None,
        source_path=None,
        available_profiles=[],
        active_profile=None,
        capture_raw=False,
    )
    monkeypatch.setattr(ec_mod, "load_experiment_config", lambda config_path, profile=None: cfg)
    monkeypatch.setattr(cli_mod, "_load_attr", lambda spec_str: lambda cfg: spec)
    monkeypatch.setattr(cli_mod, "_load_key", lambda key_path: object())
    monkeypatch.setattr(cli_mod, "_make_client", lambda coordinator, key_path: object())
    monkeypatch.setattr(cli_mod, "_resolve_experiment", lambda client, target: ("exp-1", "x"))
    monkeypatch.setattr(cli_mod, "_wait_for_approval", lambda *a, **k: "approved")

    def _interrupted_run(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(driver_mod, "run_until", _interrupted_run)
    _LifecycleRecorder.calls = []
    monkeypatch.setattr(experiment_mod, "Experiment", _LifecycleRecorder)

    r = CliRunner().invoke(
        main,
        ["experiment", "run", "exp-1", "--driver", "x:y", "--journal", str(tmp_path / "j.jsonl")],
        standalone_mode=False,
    )
    assert isinstance(r.exception, SystemExit) and r.exception.code == 130
    assert _LifecycleRecorder.calls == [("finalize", "exp-1")]  # kept, not aborted


def test_run_ctrl_c_resumable_leaves_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--resumable opts out of finalize: the run is left running server-side (neither
    finalized nor aborted) so the researcher can resume and add more units."""
    from types import SimpleNamespace

    import auspexai_tenant.cli as cli_mod
    import auspexai_tenant.driver as driver_mod
    import auspexai_tenant.experiment as experiment_mod
    import auspexai_tenant.experiment_config as ec_mod
    from auspexai_tenant import Counter
    from auspexai_tenant.driver import DriverSpec

    spec = DriverSpec(
        condition=lambda agg: False,
        next_batch=lambda agg, rnd: None,
        reduce=Counter(bucket=lambda r: "x"),
    )
    cfg = SimpleNamespace(
        driver_path=None,
        source_path=None,
        available_profiles=[],
        active_profile=None,
        capture_raw=False,
    )
    monkeypatch.setattr(ec_mod, "load_experiment_config", lambda config_path, profile=None: cfg)
    monkeypatch.setattr(cli_mod, "_load_attr", lambda spec_str: lambda cfg: spec)
    monkeypatch.setattr(cli_mod, "_load_key", lambda key_path: object())
    monkeypatch.setattr(cli_mod, "_make_client", lambda coordinator, key_path: object())
    monkeypatch.setattr(cli_mod, "_resolve_experiment", lambda client, target: ("exp-1", "x"))
    monkeypatch.setattr(cli_mod, "_wait_for_approval", lambda *a, **k: "approved")

    def _interrupted_run(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(driver_mod, "run_until", _interrupted_run)
    _LifecycleRecorder.calls = []
    monkeypatch.setattr(experiment_mod, "Experiment", _LifecycleRecorder)

    r = CliRunner().invoke(
        main,
        [
            "experiment",
            "run",
            "exp-1",
            "--driver",
            "x:y",
            "--resumable",
            "--journal",
            str(tmp_path / "j.jsonl"),
        ],
        standalone_mode=False,
    )
    assert isinstance(r.exception, SystemExit) and r.exception.code == 130
    assert _LifecycleRecorder.calls == []  # neither finalized nor aborted — left running


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


def test_launch_drives_the_experiment_it_submitted_not_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (concurrent --detach collapse): after approval, launch must DRIVE
    the exact experiment id it just submitted — never `latest`. The prior bug
    resolved the stamped label for the approval WAIT (line 1524) but then invoked
    `experiment run --target latest` for the DRIVE, so N concurrent launches all
    drove the last-submitted experiment and the other N-1 drivers submitted 0 units.
    The --no-drive test above never exercised the drive call, so it missed this.
    Assert the target handed to `experiment_run` is the resolved exp- id."""
    import json

    import auspexai_tenant.cli as cli_mod

    _launch_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _fake_build(**kw):
        kw["out_path"].write_text(json.dumps({"experiment_id": "vig-mine-20260709-000001"}))

    monkeypatch.setattr(cli_mod.experiment_build, "callback", _fake_build)
    monkeypatch.setattr(cli_mod.experiment_submit, "callback", lambda **kw: None)
    monkeypatch.setattr(cli_mod, "_make_client", lambda coordinator, key_path: object())
    monkeypatch.setattr(cli_mod, "_load_key", lambda key_path: object())
    monkeypatch.setattr(cli_mod, "_record_benchmark_declaration", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_mod,
        "_resolve_experiment",
        lambda client, target: ("exp-mine", "vig-mine-20260709-000001"),
    )
    monkeypatch.setattr(cli_mod, "_wait_for_approval", lambda *a, **k: "approved")

    seen = {}
    monkeypatch.setattr(
        cli_mod.experiment_run, "callback", lambda **kw: seen.update(target=kw.get("target"))
    )

    r = CliRunner().invoke(main, ["experiment", "launch"], standalone_mode=False)
    assert r.exit_code == 0, r.output
    assert seen["target"] == "exp-mine"  # the id it submitted — NOT "latest"
