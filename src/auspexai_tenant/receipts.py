"""Contribution receipts — CBOR encode/decode + the Receipt Pydantic model.

A contribution receipt is the platform's signed, verifiable record that a
specific worker performed accepted work for a specific experiment. Per
Principles & Scope §6.8.1, receipts are the trust substrate of the network —
they accumulate into a track record that gates tier promotion, vouching, and
Approver eligibility.

Wire format (per §5.16 + Q20 resolution): receipts are CBOR-encoded (per
`receipt_v0_1.cddl`), embedded as the predicate body of an in-toto v1 Statement
with predicate type `https://auspexai.network/receipt/v0`, COSE-signed
(RFC 9052) by the coordinator's long-lived receipt-signing key, and recorded
in Rekor for transparency.

**v0.1 SDK scope:** raw CBOR encode/decode + Pydantic validation against the
published receipt schema. Full COSE + in-toto + Rekor verification lands in
milestone 5 or 6, alongside the platform-side signing infrastructure. Until
then, the SDK reads "naked" CBOR receipts — useful for testing and for tenants
who want to inspect the receipt structure without verification.

When the platform issues real receipts (Phase 1 lab mode → Phase 2 closed
beta → Phase 3 public), the SDK reader will be extended to:
  1. Decode COSE_Sign1 → extract payload bytes (no verification path: shows
     receipt as-is; verification path: check signature against
     AUTHORIZED_SIGNERS.md)
  2. Decode payload as CBOR → in-toto v1 Statement
  3. Extract `predicate` field → CBOR receipt bytes (this module)
  4. Pydantic-validate and return Receipt
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

import cbor2
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class TimeWindow(BaseModel):
    """Receipt time-window — start and end of the work attested."""

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime


class QuorumAgreement(BaseModel):
    """Quorum-agreement metadata for the receipt — how many workers agreed."""

    model_config = ConfigDict(extra="forbid")

    replication_factor: Annotated[int, Field(gt=0)]
    agreeing_workers: Annotated[int, Field(gt=0)]
    method: str


class ResultHashAnchor(BaseModel):
    """A Rekor anchor for one of the agreeing results."""

    model_config = ConfigDict(extra="forbid")

    rekor_log_index: Annotated[int, Field(ge=0)]
    rekor_entry_uuid: str
    result_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Receipt(BaseModel):
    """An AuspexAI contribution receipt.

    Wire format published as `cddl/receipt_v0_1.cddl`. Per the schema-
    versioning policy (§11 Q20), this format is immutable once published;
    future minor/major versions are additive and parser code grows
    monotonically ("honor any recognized version forever").

    `worker_pubkey` is the raw 32-byte Ed25519 public key. The receipt
    itself is identity-bound — see §6.8.2 cryptographic structural defense.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal["0.1"]
    tenant_id: str
    experiment_id: str
    worker_pubkey: Annotated[bytes, Field(min_length=32, max_length=32)]
    work_unit_ids: Annotated[list[str], Field(min_length=1)]
    time_window: TimeWindow
    quorum_agreement: QuorumAgreement
    result_hash_anchors: Annotated[list[ResultHashAnchor], Field(min_length=1)]

    @field_validator("worker_pubkey", mode="before")
    @classmethod
    def _decode_worker_pubkey(cls, v: Any) -> bytes:
        """Accept either raw bytes (from CBOR) or lowercase-hex string (from JSON)."""
        if isinstance(v, str):
            return bytes.fromhex(v)
        return v

    @field_serializer("worker_pubkey", when_used="json")
    def _serialize_worker_pubkey(self, v: bytes) -> str:
        """JSON form: lowercase hex. CBOR/python form uses the raw bytes directly."""
        return v.hex()


# ---- CBOR encode / decode ----------------------------------------------------


def encode_cbor(receipt: Receipt) -> bytes:
    """Encode a Receipt to raw CBOR bytes (per receipt_v0_1.cddl).

    Datetime fields are encoded as CBOR Tag 0 (RFC 3339 string). The 32-byte
    worker_pubkey is encoded as native CBOR bytes. The output is the predicate
    body that the platform's signing pipeline will wrap in an in-toto v1
    Statement and COSE-sign — see module docstring.
    """
    return cbor2.dumps(receipt.model_dump(mode="python"), canonical=True)


def decode_cbor(data: bytes) -> Receipt:
    """Decode raw CBOR bytes to a Receipt.

    Validates against the Pydantic schema; raises ValidationError if the
    bytes do not match `receipt_v0_1.cddl`.
    """
    raw = cbor2.loads(data)
    return Receipt.model_validate(raw)
