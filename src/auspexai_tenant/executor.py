"""ExecutorHarness — Python helper for writing tenant executors.

A tenant author writes a function `run_one(unit, models_dir) -> dict` that
returns the result payload. The harness wraps that function with the SDK's
command-line conventions (per `sdk_license_boundary_position.md` §6.4):
parses `--input` / `--output` / `--models` / `--timeout` arguments, loads
and validates the work-unit JSON, calls the tenant function, validates and
writes the executor-output JSON, and returns the appropriate exit code.

Usage:

    from auspexai_tenant.executor import ExecutorHarness

    def run_one(unit, models_dir):
        # The tenant's actual research code goes here.
        # `unit` is a validated WorkUnit Pydantic model.
        # `models_dir` is a Path to the read-only model-weights bind mount.
        # Return a dict that becomes the result payload (opaque to platform).
        return {"score": 0.42, "label": "drifted"}

    if __name__ == "__main__":
        import sys
        sys.exit(ExecutorHarness(run_one).main())

The harness is purely convenience. Tenants may write their executor in any
language (Rust, Go, shell, etc.); the published wire-format contracts
(`schemas/workunit_v0_1.json` and `schemas/executor_output_v0_1.json`) are
sufficient to implement an executor in any runtime without using this
harness.

Exit-code convention:
    0  — tenant code ran to completion; output written successfully
    1  — tenant code raised an exception or returned a non-dict
    2  — harness/IO failure (input not found, malformed, models dir missing,
         output write failed); tenant code never ran or its output couldn't
         be persisted

Note that the executor *process* exit code is distinct from the `exit_code`
field inside the executor-output JSON. The latter is tenant-defined and
signals success/failure of the work itself (e.g., a sanity check on model
output); the former signals whether the harness was able to execute at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from auspexai_tenant.workunits import ExecutorOutput, WorkUnit

ExecutorFn = Callable[[WorkUnit, Path], dict[str, Any]]


class ExecutorHarness:
    """Wraps a tenant `run_one(unit, models_dir) -> dict` function with the
    AuspexAI SDK executor-invocation conventions.

    See module docstring for the full contract and usage example.
    """

    def __init__(self, run_one: ExecutorFn) -> None:
        self._run_one = run_one

    def main(self, argv: list[str] | None = None) -> int:
        """Run the harness. Returns the process exit code (0 / 1 / 2).

        The caller is expected to do `sys.exit(ExecutorHarness(fn).main())`
        in a `__main__` block. Returning rather than calling sys.exit makes
        the harness testable from within pytest without subprocess setup.
        """
        args = self._parse_args(argv)

        try:
            unit = self._load_unit(args.input)
        except FileNotFoundError as e:
            self._stderr(f"work-unit input not found: {e}")
            return 2
        except json.JSONDecodeError as e:
            self._stderr(f"work-unit input is not valid JSON: {e}")
            return 2
        except ValidationError as e:
            self._stderr(f"work-unit input failed schema validation:\n{e}")
            return 2

        models_dir = Path(args.models)
        if not models_dir.is_dir():
            self._stderr(f"--models path is not a directory: {models_dir}")
            return 2

        # Invoke the tenant's run_one. Catch broadly: anything raised here
        # is a tenant-code failure, not a harness failure. Exit 1.
        try:
            payload = self._run_one(unit, models_dir)
        except Exception as e:
            self._stderr(f"tenant run_one() raised {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            return 1

        if not isinstance(payload, dict):
            self._stderr(f"tenant run_one() must return dict, got {type(payload).__name__}")
            return 1

        output = ExecutorOutput(
            schema_version="0.1",
            unit_id=unit.unit_id,
            completed_at=datetime.now(UTC),
            exit_code=0,
            payload=payload,
        )

        try:
            self._write_output(Path(args.output), output)
        except OSError as e:
            self._stderr(f"failed to write output to {args.output}: {e}")
            return 2

        return 0

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _parse_args(argv: list[str] | None) -> argparse.Namespace:
        p = argparse.ArgumentParser(
            prog="tenant-executor",
            description="AuspexAI tenant executor (powered by auspexai-tenant)",
        )
        p.add_argument(
            "--input",
            required=True,
            help="Path to work-unit JSON file (read-only bind mount)",
        )
        p.add_argument(
            "--output",
            required=True,
            help="Path to write executor-output JSON file (write-only bind mount)",
        )
        p.add_argument(
            "--models",
            required=True,
            help="Path to model-weights directory (read-only bind mount)",
        )
        p.add_argument(
            "--timeout",
            type=int,
            default=600,
            help="Soft timeout hint in seconds (advisory; the worker enforces "
            "the hard timeout via subprocess kill)",
        )
        return p.parse_args(argv)

    @staticmethod
    def _load_unit(path: str) -> WorkUnit:
        with open(path) as f:
            raw = json.load(f)
        return WorkUnit.model_validate(raw)

    @staticmethod
    def _write_output(path: Path, output: ExecutorOutput) -> None:
        # Atomic-ish: write to <path>.tmp then rename. Worker reads only after
        # rename, so partial writes are never visible.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(output.model_dump_json(indent=2))
        tmp_path.rename(path)

    @staticmethod
    def _stderr(msg: str) -> None:
        print(f"executor: {msg}", file=sys.stderr)
