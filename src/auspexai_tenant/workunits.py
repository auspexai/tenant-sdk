"""Work-unit and result Pydantic models.

Both envelopes wrap a tenant-defined `payload` field that is opaque to the
AuspexAI platform (Principles §5.1 tenant-neutral core). The platform reads
these as data; the tenant executor (forked in the sandbox subprocess per §5.17)
is the only code that interprets the payload.
"""

from __future__ import annotations

from datetime import datetime
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
