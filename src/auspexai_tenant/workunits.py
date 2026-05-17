"""Work-unit and result Pydantic models, plus static work-unit-source packing.

Both envelopes wrap a tenant-defined `payload` field that is opaque to the
AuspexAI platform (Principles §5.1 tenant-neutral core). The platform reads
these as data; the tenant executor (forked in the sandbox subprocess per §5.17)
is the only code that interprets the payload.

For the static work-unit-source pattern (per `sdk_license_boundary_position.md`
§6.3), `tar_writer` packages a sequence of WorkUnit objects into a gzipped tar
archive suitable for uploading as the manifest's `tarball_sha256` content.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkUnit(BaseModel):
    """A unit of experimental work dispatched to a worker.

    `manifest_sha256` content-addresses the manifest this unit is bound to —
    manifest-swap attacks are foreclosed by the worker rejecting units whose
    hash doesn't match the manifest the volunteer accepted (Principles §5.14).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    unit_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
    tenant_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")]
    experiment_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,127}$")]
    manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    created_at: datetime
    payload: dict[str, Any]


class ExecutorOutput(BaseModel):
    """What a tenant executor writes to its --output path.

    The executor-to-worker handoff format. The worker daemon reads this,
    verifies the unit_id matches the work-unit it dispatched, adds
    worker_pubkey + worker_signature, and submits the full Result to the
    coordinator. Distinct from Result (which carries the worker's signature
    over the canonical encoding).

    Tenant authors writing executors in any language target this schema;
    the Python ExecutorHarness in auspexai_tenant.executor wraps the
    CLI/IO boilerplate for tenants who choose Python.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    unit_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
    completed_at: datetime
    exit_code: Annotated[int, Field(ge=-255, le=255)]
    payload: dict[str, Any]


class Result(BaseModel):
    """A result submitted by a worker for a given work unit.

    `worker_signature` is added by the worker daemon at submission time (Ed25519
    signature over the canonical JSON encoding of the other fields). The
    keystore lives in the trusted worker daemon, not the sandbox subprocess
    (Principles §5.17 two-tier process model).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    unit_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")]
    worker_pubkey: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    completed_at: datetime
    exit_code: Annotated[int, Field(ge=-255, le=255)]
    payload: dict[str, Any]
    worker_signature: Annotated[str, Field(pattern=r"^[A-Za-z0-9+/=]+$")]


# ---- Static work-unit-source packing ---------------------------------------


def tar_writer(units: Iterable[WorkUnit], out_path: Path) -> int:
    """Package work units as a gzipped tar archive (`tarball_sha256` source pattern).

    Each WorkUnit is serialized to a JSON file named `unit_<unit_id>.json` and
    added to the archive. Returns the number of units written. The output file
    is suitable for the static work-unit-source pattern in `manifest.json`:
    compute `sha256(out_path)` and set `work_unit_source.tarball_sha256`.

    Naming is `unit_<unit_id>.json` because the coordinator-side unpacker
    scans for this pattern (per `sdk_license_boundary_position.md` §6.3).

    Raises ValueError on a duplicate `unit_id` (which would create archive
    name collisions and ambiguity at dispatch time).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    count = 0
    with tarfile.open(out_path, mode="w:gz") as tf:
        for unit in units:
            if unit.unit_id in seen_ids:
                raise ValueError(f"duplicate unit_id in work-unit set: {unit.unit_id}")
            seen_ids.add(unit.unit_id)
            data = unit.model_dump_json(indent=2).encode("utf-8")
            info = tarfile.TarInfo(name=f"unit_{unit.unit_id}.json")
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
            count += 1
    return count


def tar_reader(tarball_path: Path) -> list[WorkUnit]:
    """Read work units back from a gzipped tar archive produced by `tar_writer`.

    Convenience for testing: lets tenants verify their tarball can be unpacked
    correctly before computing `tarball_sha256` and submitting. Coordinator-
    side unpacking is the platform's responsibility and uses the same naming
    convention.
    """
    units: list[WorkUnit] = []
    with tarfile.open(tarball_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.startswith("unit_"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            raw = json.loads(f.read())
            units.append(WorkUnit.model_validate(raw))
    return units
