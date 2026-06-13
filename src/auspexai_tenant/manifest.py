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

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

SensitiveContentFlag = Literal[
    "dual_use",
    "red_team",
    "harmful_output_generation",
]


class Model(BaseModel):
    """A model the experiment requires workers to have locally (BYOM — §5.8).

    `hf_repo` + `hf_filename` are optional acquisition coordinates (M3 lazy
    auto-acquire): when present, an auto-acquire-enabled worker that lacks the
    model may pull this exact file (the pinned quant) from HuggingFace rather
    than refusing the unit. They are self-describing — a provisioned worker
    reads them from its locally-staged manifest. Absent coords ⇒ the model must
    be staged out-of-band (the pre-M3 BYOM behavior). `hf_filename` pins the
    exact GGUF so hash-agreement consensus runs the identical quant across
    replicas (the acquisition-side analog of the manifest_sha256 pin)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    local_weights_required: bool
    hf_repo: str | None = None
    hf_filename: str | None = None


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
    # Digest over the executor *files* the tenant stages (computed by
    # `compute_package_digest`). When set, the worker re-derives it over the staged
    # package and refuses on mismatch — so a provisioned worker verifies the CODE
    # it runs is what the tenant *signed*, not just that the manifest JSON matches
    # (closes the provenance gap that local manual staging leaves open, and makes
    # coordinator-served fetch a drop-in). Optional + backward-compatible: a
    # manifest without it keeps the Phase-1 "operator is the trust root" behavior.
    package_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None


class BuiltinReducer(BaseModel):
    """Built-in SHA-256 hash-agreement reducer.

    Suitable for deterministic outputs (most hash-agreement result shapes fit).
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


ResearchClass = Literal[
    "behavioral_drift",
    "eval_sweeps",
    "refusal_boundary_mapping",
    "cross_model_comparison",
    "quantization_effects",
    "prompt_sensitivity",
    "other",
]
"""The §11 research-class taxonomy (§9 #48 keys auto-approval on it). Optional on
the manifest; the coordinator is authoritative — it re-validates membership AND
that the class is within the tenant's approved application classes, then decides
auto-approve vs human review by class × tenant tier."""


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
    # §9 #48: the declared research class. Optional for back-compat (a manifest
    # without it routes to human review — the agent can't auto-clear an
    # unclassified experiment).
    research_class: ResearchClass | None = None
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


# Files never part of the executor package digest: the manifest itself (separately
# content-addressed) and Python bytecode caches (runtime pollution).
# manifest.json.sig is excluded because `manifest sign` writes it NEXT TO the
# manifest by default (i.e. into the package dir) AFTER the digest was computed
# — including it would poison executor.package_sha256 for any tenant staging
# their package dir wholesale. The worker reimplementation excludes it
# identically (the contract is the format, not shared code).
_PACKAGE_DIGEST_EXCLUDE = ("manifest.json", "manifest.json.sig")


def _iter_package_files(package_dir: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield `(posix-relpath, path)` for every file that is part of the executor
    package, sorted by relpath, applying the digest exclusions.

    The single enumeration seam shared by `compute_package_digest` and the
    archive builder (`auspexai_tenant.package.build_package_archive`) — using
    one walk guarantees the uploaded archive and the `X-Package-Digest` header
    can never disagree about which files a package contains.
    """
    root = Path(package_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if rel in _PACKAGE_DIGEST_EXCLUDE or rel.endswith(".pyc") or "__pycache__" in parts:
            continue
        yield rel, path


def compute_package_digest(package_dir: str | Path) -> str:
    """Digest over the executor *files* staged in `package_dir`, for the manifest's
    `executor.package_sha256` (the provenance pin the worker verifies).

    Canonical + deterministic: every regular file under `package_dir`, except
    `manifest.json` / `manifest.json.sig` and any `__pycache__/` / `*.pyc`,
    contributes one line
    ``<posix-relpath>\\x00<sha256-hex>`` (lines sorted by relpath, joined by
    ``\\n``); the digest is the SHA-256 of that blob. The worker re-implements this
    byte-for-byte (`auspexai_worker.provisioning.compute_package_digest`) — the
    shared contract is the format, not shared code (SDK is Apache, worker AGPL).
    """
    lines: list[str] = []
    for rel, path in _iter_package_files(package_dir):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{rel}\x00{file_hash}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
