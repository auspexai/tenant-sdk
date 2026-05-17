"""ReducerHarness — Python helper for writing tenant result reducers.

When a work unit completes with replication factor N, the coordinator collects
N results and decides whether they agree (quorum). For deterministic outputs,
the platform-side built-in hash-agreement reducer is sufficient. For
non-deterministic outputs (agent runs with sampling, etc.), tenants ship a
custom reducer that the coordinator forks in a sandboxed subprocess.

This module wraps the SDK reducer-invocation conventions (per
`sdk_license_boundary_position.md` §6.5): the harness reads all `result_*.json`
files from `--results`, validates each as a Result envelope, calls the
tenant's `reduce(results)` function, validates the returned ReducerDecision,
and writes it to `--output`.

Usage:

    from auspexai_tenant.reducer import ReducerHarness, ReducerDecision

    def reduce(results):
        # The tenant's actual reduction logic — compare results, decide if
        # they constitute agreement, build a merged result if so.
        payloads = [r.payload for r in results]
        if all(p == payloads[0] for p in payloads):
            return ReducerDecision(
                schema_version="0.1",
                verdict="agree",
                merged=payloads[0],
            )
        return ReducerDecision(schema_version="0.1", verdict="disagree")

    if __name__ == "__main__":
        import sys
        sys.exit(ReducerHarness(reduce).main())

Exit-code convention (mirrors ExecutorHarness):
    0  — reducer ran to completion; decision written successfully
    1  — tenant code raised or returned a non-ReducerDecision
    2  — harness/IO failure (results dir missing/empty/malformed, output
         write failed)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from auspexai_tenant.workunits import Result

ReducerFn = Callable[[list[Result]], "ReducerDecision"]


class ReducerDecision(BaseModel):
    """The reducer's decision file shape — written to --output.

    Wire format published as `schemas/reducer_decision_v0_1.json` and
    `cddl/reducer_decision_v0_1.cddl`. Immutable once published.

    When verdict is "agree", `merged` is required and contains the
    canonical merged payload the coordinator stores as the experiment's
    result for this work unit. When verdict is "disagree", `merged` is
    omitted; the coordinator handles disagreement per its own policy
    (typically: re-dispatch to fresh workers, or mark the unit as failed
    after N retries).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    verdict: Literal["agree", "disagree"]
    merged: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _merged_required_on_agree(self) -> ReducerDecision:
        if self.verdict == "agree" and self.merged is None:
            raise ValueError("merged is required when verdict is 'agree'")
        if self.verdict == "disagree" and self.merged is not None:
            raise ValueError("merged must be omitted when verdict is 'disagree'")
        return self


class ReducerHarness:
    """Wraps a tenant `reduce(results) -> ReducerDecision` function with the
    AuspexAI SDK reducer-invocation conventions.

    See module docstring for the full contract and usage example.
    """

    def __init__(self, reduce: ReducerFn) -> None:
        self._reduce = reduce

    def main(self, argv: list[str] | None = None) -> int:
        """Run the harness. Returns the process exit code (0 / 1 / 2)."""
        args = self._parse_args(argv)

        results_dir = Path(args.results)
        if not results_dir.is_dir():
            self._stderr(f"--results path is not a directory: {results_dir}")
            return 2

        try:
            results = self._load_results(results_dir)
        except (json.JSONDecodeError, ValidationError) as e:
            self._stderr(f"failed to load results from {results_dir}: {e}")
            return 2

        if not results:
            self._stderr(f"--results directory is empty: {results_dir}")
            return 2

        try:
            decision = self._reduce(results)
        except Exception as e:
            self._stderr(f"tenant reduce() raised {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            return 1

        if not isinstance(decision, ReducerDecision):
            self._stderr(
                f"tenant reduce() must return a ReducerDecision, got {type(decision).__name__}"
            )
            return 1

        try:
            self._write_decision(Path(args.output), decision)
        except OSError as e:
            self._stderr(f"failed to write decision to {args.output}: {e}")
            return 2

        return 0

    # ---- internals ---------------------------------------------------------

    @staticmethod
    def _parse_args(argv: list[str] | None) -> argparse.Namespace:
        p = argparse.ArgumentParser(
            prog="tenant-reducer",
            description="AuspexAI tenant reducer (powered by auspexai-tenant)",
        )
        p.add_argument(
            "--results",
            required=True,
            help="Path to directory of result_*.json files (read-only bind mount)",
        )
        p.add_argument(
            "--output",
            required=True,
            help="Path to write decision JSON file (write-only bind mount)",
        )
        return p.parse_args(argv)

    @staticmethod
    def _load_results(results_dir: Path) -> list[Result]:
        results: list[Result] = []
        for path in sorted(results_dir.glob("result_*.json")):
            with open(path) as f:
                raw = json.load(f)
            results.append(Result.model_validate(raw))
        return results

    @staticmethod
    def _write_decision(path: Path, decision: ReducerDecision) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(decision.model_dump_json(indent=2))
        tmp_path.rename(path)

    @staticmethod
    def _stderr(msg: str) -> None:
        print(f"reducer: {msg}", file=sys.stderr)
