"""Manifest Pydantic models, matching schemas/manifest_v0_1.json.

The manifest is the declarative half of the tenant published contract: it is
pure data (JSON), validated against a published JSON Schema (mirror of the
normative CDDL in cddl/manifest_v0_1.cddl). Tenants submit manifests via the
SDK CLI; the AuspexAI coordinator reads them as data, content-addresses them,
and stores them for cross-reference by work units.

Per Principles §5.14, the schema is content-addressed (manifest_sha256 lives in
each work unit) — manifest mutation after submission requires the worker to
re-accept the new hash.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

SensitiveContentFlag = Literal[
    "dual_use",
    "red_team",
    "harmful_output_generation",
]


class Model(BaseModel):
    """A model the experiment requires workers to have locally (BYOM — §5.8)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    local_weights_required: bool


class ApproverAttestation(BaseModel):
    """Signed attestation from an Approver-pool member (§5.12, §6.5).

    Required when sensitive_content_flags is non-empty.
    """

    model_config = ConfigDict(extra="forbid")

    approver_pubkey: Annotated[str, Field(pattern=r"^ed25519:[A-Za-z0-9+/=]+$")]
    sig: Annotated[str, Field(pattern=r"^[A-Za-z0-9+/=]+$")]


class StaticWorkUnitSource(BaseModel):
    """Work-unit source: tarball of pre-generated unit_<id>.json files.

    Phase 1 default. Strongest separation — tarball hash gives content-address.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["static"]
    tarball_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class HttpWorkUnitSource(BaseModel):
    """Work-unit source: tenant-hosted HTTPS endpoint the coordinator POSTs to."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["http"]
    url: Annotated[str, Field(pattern=r"^https://")]


WorkUnitSource = Annotated[
    StaticWorkUnitSource | HttpWorkUnitSource,
    Field(discriminator="kind"),
]


class Executor(BaseModel):
    """The tenant executor: forked by the worker in the sandbox subprocess (§5.17).

    `command` is the argv to invoke inside the sandbox. The worker passes
    --input/--output/--models/--timeout per the executor invocation convention
    documented in sdk_license_boundary_position.md §6.4.
    """

    model_config = ConfigDict(extra="forbid")

    command: Annotated[list[str], Field(min_length=1)]
    image_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


class BuiltinReducer(BaseModel):
    """Built-in SHA-256 hash-agreement reducer.

    Suitable for deterministic outputs. Most Sentinel result shapes fit this.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["builtin_hash_agreement"]


class CustomReducer(BaseModel):
    """Tenant-supplied reducer: forked by the coordinator in a subprocess.

    Same boundary mechanism as the executor (subprocess + filesystem IO).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["custom"]
    command: Annotated[list[str], Field(min_length=1)]
    image_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


Reducer = Annotated[
    BuiltinReducer | CustomReducer,
    Field(discriminator="kind"),
]


class Manifest(BaseModel):
    """AuspexAI tenant manifest, v0.1.

    Mirrors schemas/manifest_v0_1.json. See sdk_license_boundary_position.md §6.2
    for the published-contract framing.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"]
    tenant_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")]
    tenant_maintainer_contact: EmailStr
    experiment_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,127}$")]
    research_goal_paragraph: Annotated[str, Field(min_length=50, max_length=2000)]
    models: Annotated[list[Model], Field(min_length=1)]
    prompt_set_characteristics: Annotated[str, Field(min_length=10, max_length=1000)]
    sensitive_content_flags: list[SensitiveContentFlag] = Field(default_factory=list)
    approver_attestations: list[ApproverAttestation] | None = None
    expected_duration_hours: Annotated[float, Field(gt=0, le=8760)]
    replication_factor: Annotated[int, Field(ge=1, le=100)]
    work_unit_source: WorkUnitSource
    executor: Executor
    reducer: Reducer

    @model_validator(mode="after")
    def _sensitive_requires_attestation(self) -> Manifest:
        if self.sensitive_content_flags and not self.approver_attestations:
            raise ValueError(
                "approver_attestations is required when sensitive_content_flags is non-empty "
                "(Principles §5.12 research ethics)"
            )
        return self
