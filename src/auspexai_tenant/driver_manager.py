"""Detached experiment-driver process management.

A long experiment's driver is the foreground `launch`/`run` process: it must stay
alive for the whole run, one experiment per terminal, and a Ctrl-C or dropped SSH
silently orphans it ("approved with no work units"). `--detach` fixes that by
re-executing the same command (minus `--detach`) as a NEW-SESSION background
process (`start_new_session=True` == POSIX setsid), so the driver survives a closed
terminal — no tmux, no nohup. Each detached driver keeps a small record under
`DRIVERS_DIR` so `experiment ps` shows which drivers are live and `experiment stop`
can signal them. State is local; liveness is a `kill(pid, 0)` probe (no network).

The detached CHILD runs the ordinary `launch`/`run` path with `AUSPEXAI_DRIVER_DIR`
set, and stamps the experiment it resolves into its record (so `ps` can show it).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Set on the detached child so it records into its own run dir (see record_experiment).
ENV_DRIVER_DIR = "AUSPEXAI_DRIVER_DIR"


def drivers_dir() -> Path:
    """`~/.local/share/auspexai-tenant/drivers/` (honors XDG_DATA_HOME)."""
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    d = root / "auspexai-tenant" / "drivers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_alive(pid: int) -> bool:
    """Signal-0 liveness probe. False once the process has exited."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours — treat as alive (shouldn't happen for our children)
    return True


def _meta_path(run_dir: Path) -> Path:
    return run_dir / "meta.json"


def _read_meta(run_dir: Path) -> dict:
    try:
        return json.loads(_meta_path(run_dir).read_text())
    except (OSError, ValueError):
        return {}


def _write_meta(run_dir: Path, meta: dict) -> None:
    tmp = _meta_path(run_dir).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(_meta_path(run_dir))


@dataclass(frozen=True)
class DriverRecord:
    run_id: str
    dir: Path
    pid: int
    profile: str | None
    experiment_id: str | None
    label: str | None
    started_at: str
    status: str
    log_path: Path

    @property
    def alive(self) -> bool:
        return pid_alive(self.pid)

    def uptime_seconds(self, now: float | None = None) -> float | None:
        try:
            started = datetime.fromisoformat(self.started_at).timestamp()
        except (ValueError, TypeError):
            return None
        return max(0.0, (now if now is not None else time.time()) - started)


def _to_record(run_dir: Path, meta: dict) -> DriverRecord:
    return DriverRecord(
        run_id=meta.get("run_id", run_dir.name),
        dir=run_dir,
        pid=int(meta.get("pid", -1)),
        profile=meta.get("profile"),
        experiment_id=meta.get("experiment_id"),
        label=meta.get("label"),
        started_at=meta.get("started_at", ""),
        status=meta.get("status", "unknown"),
        log_path=run_dir / "driver.log",
    )


def spawn_detached(argv: list[str], *, profile: str | None, cwd: str | None = None) -> DriverRecord:
    """Re-exec `argv` as a detached, new-session background driver; return its record.

    Classic daemon spawn: fork → setsid (a new session, so the driver outlives the
    controlling terminal / dropped SSH) → redirect stdio to the log → exec the CLI.
    The parent returns the child's pid; when the launching CLI exits, the child
    re-parents to init. `argv` must already have `--detach` removed (else the child
    would re-detach). POSIX only (the SDK targets macOS + Linux)."""
    work_dir = cwd or os.getcwd()
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{profile or 'base'}-{ts}-{os.getpid()}"
    run_dir = drivers_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "driver.log"

    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # child — set up a detached session, redirect stdio, exec the CLI
        try:
            os.chdir(work_dir)
            os.setsid()  # new session leader: no controlling terminal
            os.environ[ENV_DRIVER_DIR] = str(run_dir)
            devnull = os.open(os.devnull, os.O_RDONLY)
            logfd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(devnull, 0)
            os.dup2(logfd, 1)
            os.dup2(logfd, 2)
            os.execvp(argv[0], argv)
        except BaseException:  # exec failed — don't fall back into the parent's code
            os._exit(127)
        os._exit(127)  # unreachable

    meta = {
        "run_id": run_id,
        "pid": pid,
        "profile": profile,
        "argv": argv,
        "cwd": work_dir,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "starting",
        "experiment_id": None,
        "label": None,
    }
    _write_meta(run_dir, meta)
    return _to_record(run_dir, meta)


def record_experiment(experiment_id: str, label: str, *, status: str = "submitted") -> None:
    """Called by the DETACHED CHILD (when ENV_DRIVER_DIR is set) to stamp the
    experiment it resolved into its own record, so `experiment ps` can show it.
    A no-op in a foreground run (env unset)."""
    d = os.environ.get(ENV_DRIVER_DIR)
    if not d:
        return
    run_dir = Path(d)
    meta = _read_meta(run_dir)
    meta.update(experiment_id=experiment_id, label=label, status=status)
    _write_meta(run_dir, meta)


def set_status(status: str) -> None:
    """Update the detached child's status (e.g. 'driving' after approval). No-op in
    the foreground."""
    d = os.environ.get(ENV_DRIVER_DIR)
    if not d:
        return
    run_dir = Path(d)
    meta = _read_meta(run_dir)
    if meta:
        meta["status"] = status
        _write_meta(run_dir, meta)


def list_drivers() -> list[DriverRecord]:
    """All recorded drivers, newest first."""
    out: list[DriverRecord] = []
    for run_dir in drivers_dir().iterdir():
        if not run_dir.is_dir():
            continue
        meta = _read_meta(run_dir)
        if meta:
            out.append(_to_record(run_dir, meta))
    out.sort(key=lambda r: r.started_at, reverse=True)
    return out


def find_drivers(target: str) -> list[DriverRecord]:
    """Drivers matching `target` by run_id, experiment_id, label, or profile."""
    return [r for r in list_drivers() if target in (r.run_id, r.experiment_id, r.label, r.profile)]


def stop_driver(rec: DriverRecord, sig: int = signal.SIGINT) -> bool:
    """Signal the driver (SIGINT by default → reuses the launch/run Ctrl-C abort
    path). Returns True if a live process was signaled, False if already dead."""
    if not pid_alive(rec.pid):
        return False
    os.kill(rec.pid, sig)
    return True


def prune_dead() -> int:
    """Remove records of drivers whose process has exited. Returns the count removed."""
    n = 0
    for r in list_drivers():
        if not r.alive:
            shutil.rmtree(r.dir, ignore_errors=True)
            n += 1
    return n
