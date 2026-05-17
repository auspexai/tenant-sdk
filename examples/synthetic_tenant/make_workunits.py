#!/usr/bin/env python3
"""Generate work-unit tarball for the synthetic test tenant.

Produces `work_units.tar.gz` containing unit_0001.json .. unit_0010.json,
each carrying a single integer payload. The SHA-256 of the tarball is the
value tenant authors put in their manifest's `work_unit_source.tarball_sha256`.

Run from the repo root:

    uv run python examples/synthetic_tenant/make_workunits.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from auspexai_tenant.workunits import WorkUnit, tar_writer


def main() -> int:
    out_dir = Path(__file__).parent
    out_path = out_dir / "work_units.tar.gz"

    units: list[WorkUnit] = []
    now = datetime.now(UTC)
    placeholder_manifest_hash = "0" * 64  # real value set after manifest is sha'd
    for i in range(1, 11):
        units.append(
            WorkUnit(
                schema_version="0.1",
                unit_id=f"u{i:04d}",
                tenant_id="synth-doubler",
                experiment_id="synth-doubler-v1",
                manifest_sha256=placeholder_manifest_hash,
                created_at=now,
                payload={"value": i},
            )
        )

    count = tar_writer(units, out_path)
    tarball_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {count} work units to {out_path}")
    print(f"tarball_sha256: {tarball_sha}")
    print("(set this value in manifest.json work_unit_source.tarball_sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
