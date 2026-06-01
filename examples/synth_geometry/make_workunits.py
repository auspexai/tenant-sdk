#!/usr/bin/env python3
"""Generate the work-unit tarball for the synth-geometry tenant.

Each unit pins a distinct PRNG ``seed`` (the #33 determinism contract): all N
replicas of a unit share the payload, hence the seed, hence reproducible draws —
so honest replicas agree under exact-hash quorum. Different units explore
different seeded estimates; the experiment-level reduce (``reduce_experiment.py``)
folds them into one geometry fingerprint.

    python make_workunits.py [--units 12] [--samples 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from auspexai_tenant.workunits import WorkUnit, tar_writer

TENANT_ID = "synth-geometry"
EXPERIMENT_ID = "synth-geometry-v1"


def build_units(n_units: int, n_samples: int) -> list[WorkUnit]:
    now = datetime.now(UTC)
    placeholder_manifest_hash = "0" * 64  # set after manifest is sha'd
    units: list[WorkUnit] = []
    for i in range(1, n_units + 1):
        units.append(
            WorkUnit(
                schema_version="0.1",
                unit_id=f"g{i:04d}",
                tenant_id=TENANT_ID,
                experiment_id=EXPERIMENT_ID,
                manifest_sha256=placeholder_manifest_hash,
                created_at=now,
                # The seed is what makes Monte-Carlo reproducible across replicas.
                payload={
                    "seed": 1000 + i,
                    "n_samples": n_samples,
                    "probe": "mean_abs_cosine_separation",
                },
            )
        )
    return units


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=int, default=12)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args(argv)

    out_path = Path(__file__).parent / "work_units.tar.gz"
    units = build_units(args.units, args.samples)
    count = tar_writer(units, out_path)
    tarball_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {count} work units to {out_path}")
    print(f"tarball_sha256: {tarball_sha}")
    print("(set this value in manifest.json work_unit_source.tarball_sha256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
