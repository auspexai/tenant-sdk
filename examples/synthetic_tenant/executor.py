#!/usr/bin/env python3
"""Synthetic test tenant — integer doubler executor.

Demonstrates a minimal AuspexAI tenant executor in ~15 LOC of actual logic.
Used by tests/test_synthetic_tenant.py as the §5.3 SDK-shape forcing function.

The executor is invoked as a subprocess by the worker daemon:

    python executor.py \\
        --input /sandbox/input/work_unit.json \\
        --output /sandbox/output/result.json \\
        --models /sandbox/models \\
        --timeout 60

For this tenant `--models` is unused (no model needed); the executor just
reads payload.value, doubles it, and returns {"doubled": <2*value>}.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from auspexai_tenant.executor import ExecutorHarness
from auspexai_tenant.workunits import WorkUnit


def run_one(unit: WorkUnit, models_dir: Path) -> dict[str, Any]:
    """Tenant work function: double the integer in payload.value."""
    del models_dir  # unused for this tenant
    value = unit.payload["value"]
    if not isinstance(value, int):
        raise TypeError(f"expected int in payload.value, got {type(value).__name__}")
    return {"doubled": value * 2, "input": value}


if __name__ == "__main__":
    sys.exit(ExecutorHarness(run_one).main())
