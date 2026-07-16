"""Detached-driver management: driver_manager core + the ps/stop CLI.

Spawned children are reaped in-test (os.waitpid) — in production the launch parent
exits immediately, so the detached child re-parents to init and is reaped there;
without reaping, kill(pid,0) sees a zombie and reports it alive."""

from __future__ import annotations

import os
import signal
import subprocess

import pytest
from click.testing import CliRunner

from auspexai_tenant import driver_manager as dm
from auspexai_tenant.cli import main


@pytest.fixture
def drivers_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv(dm.ENV_DRIVER_DIR, raising=False)
    return tmp_path / "auspexai-tenant" / "drivers"


def _reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass


def test_drivers_dir_honors_xdg(drivers_home):
    assert dm.drivers_dir() == drivers_home


def test_pid_alive_self_true_reaped_child_false(drivers_home):
    assert dm.pid_alive(os.getpid()) is True
    p = subprocess.Popen(["true"])
    p.wait()
    assert dm.pid_alive(p.pid) is False


def test_spawn_detached_writes_record_and_is_findable(drivers_home):
    rec = dm.spawn_detached(["sleep", "10"], profile="alpha")
    try:
        assert rec.alive
        assert rec.profile == "alpha"
        assert rec.run_id.startswith("alpha-")
        assert (rec.dir / "meta.json").exists()
        assert rec.log_path == rec.dir / "driver.log"
        assert rec.run_id in [r.run_id for r in dm.list_drivers()]
        assert dm.find_drivers("alpha")[0].run_id == rec.run_id
    finally:
        _reap(rec.pid)


def test_stop_driver_signals_then_dead(drivers_home):
    rec = dm.spawn_detached(["sleep", "10"], profile="beta")
    assert dm.stop_driver(rec) is True  # SIGINT a live driver
    os.waitpid(rec.pid, 0)  # reap (init's job in prod)
    assert dm.pid_alive(rec.pid) is False
    assert dm.stop_driver(rec) is False  # already dead


def test_prune_removes_only_dead(drivers_home):
    live = dm.spawn_detached(["sleep", "10"], profile="live")
    dead = dm.spawn_detached(["true"], profile="dead")
    os.waitpid(dead.pid, 0)  # reap the finished one
    try:
        assert dm.prune_dead() == 1
        ids = [r.run_id for r in dm.list_drivers()]
        assert live.run_id in ids and dead.run_id not in ids
    finally:
        _reap(live.pid)


def test_prune_archives_the_driver_log_by_default(drivers_home):
    # The fix: a pruned driver's driver.log — the post-mortem for a driver that died
    # unexpectedly — must SURVIVE the prune (it used to be rmtree'd), archived under
    # _ended/<run-id>/ and out of the active `ps` list.
    dead = dm.spawn_detached(["true"], profile="dead")
    os.waitpid(dead.pid, 0)
    (dead.dir / "driver.log").write_text("Traceback (most recent call last): boom\n")
    assert dm.prune_dead() == 1
    assert dead.run_id not in [r.run_id for r in dm.list_drivers()]  # gone from active
    archived = dm.ended_dir() / dead.dir.name / "driver.log"
    assert archived.exists()  # ...but the log is preserved
    assert "boom" in archived.read_text()


def test_prune_purge_hard_deletes_the_log(drivers_home):
    # keep_logs=False (the `--purge` escape hatch) reclaims space: no archive.
    dead = dm.spawn_detached(["true"], profile="dead")
    os.waitpid(dead.pid, 0)
    (dead.dir / "driver.log").write_text("gone\n")
    assert dm.prune_dead(keep_logs=False) == 1
    assert not (dm.ended_dir() / dead.dir.name).exists()
    assert not dead.dir.exists()


def test_record_experiment_and_status_via_env(drivers_home, monkeypatch):
    run_dir = dm.drivers_dir() / "child-run"
    run_dir.mkdir(parents=True)
    dm._write_meta(
        run_dir,
        {
            "run_id": "child-run",
            "pid": os.getpid(),
            "profile": "p",
            "started_at": "",
            "status": "starting",
            "experiment_id": None,
            "label": None,
        },
    )
    monkeypatch.setenv(dm.ENV_DRIVER_DIR, str(run_dir))
    dm.record_experiment("exp-XYZ", "mylabel")
    dm.set_status("driving")
    rec = next(r for r in dm.list_drivers() if r.run_id == "child-run")
    assert rec.experiment_id == "exp-XYZ" and rec.label == "mylabel" and rec.status == "driving"


def test_record_experiment_noop_without_env(drivers_home):
    dm.record_experiment("exp-X", "l")  # ENV unset → no-op, no crash
    assert dm.list_drivers() == []


# ---- ps / stop CLI ----


def test_ps_empty(drivers_home):
    r = CliRunner().invoke(main, ["experiment", "ps"])
    assert r.exit_code == 0 and "no detached drivers" in r.output


def test_ps_lists_running(drivers_home):
    rec = dm.spawn_detached(["sleep", "10"], profile="gamma")
    try:
        r = CliRunner().invoke(main, ["experiment", "ps"])
        assert r.exit_code == 0
        assert "running" in r.output and "gamma" in r.output and rec.run_id in r.output
    finally:
        _reap(rec.pid)


def test_stop_requires_target(drivers_home):
    assert CliRunner().invoke(main, ["experiment", "stop"]).exit_code == 2


def test_stop_by_profile(drivers_home):
    rec = dm.spawn_detached(["sleep", "10"], profile="delta")
    r = CliRunner().invoke(main, ["experiment", "stop", "delta"])
    os.waitpid(rec.pid, 0)  # reap
    assert r.exit_code == 0 and "stopped" in r.output


def test_stop_unknown_target_errors(drivers_home):
    assert CliRunner().invoke(main, ["experiment", "stop", "nope"]).exit_code == 1
