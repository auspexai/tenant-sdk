"""Result-set completion attestation — tenant-side fetch + verify (#34 §6.3).

The coordinator emits a model-blind, COSE-signed in-toto statement over the
Merkle root of an experiment's per-unit consensus set (see the platform's
`receipts/attestation.py`). This module lets a tenant **independently verify**
it: recompute the root the same way, check the COSE Ed25519 signature against an
authorized signer, and confirm the signed root matches. That closes the
reproducibility loop — "aggregate A = reduce(set with root R); R is
coordinator-attested" — without trusting the coordinator's word.

The Merkle construction here is byte-identical to the coordinator's:
    leaf = SHA-256( 0x00 || canonical_json({unit_id, consensus_result_hash, receipt_id}) )
    node = SHA-256( 0x01 || left || right )      # odd level → duplicate last
    root = top node (hex); empty set → SHA-256(0x00).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

RESULT_SET_ALGORITHM = "sha256-merkle-v0"
RESULT_SET_PREDICATE_TYPE = "https://auspexai.network/result-set/v0"

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"

# COSE_Sign1 / RFC 9052 constants (must match the coordinator's signing.py).
_COSE_HEADER_ALG = 1
_COSE_HEADER_KID = 4
_COSE_ALG_EDDSA = -8


def _leaf_bytes(unit: dict[str, Any]) -> bytes:
    canonical = json.dumps(
        {
            "unit_id": unit["unit_id"],
            "consensus_result_hash": unit["consensus_result_hash"],
            "receipt_id": unit["receipt_id"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_LEAF_PREFIX + canonical).digest()


def merkle_root(units: list[dict[str, Any]]) -> str:
    """Recompute the result-set Merkle root over `units` (each a dict with
    unit_id / consensus_result_hash / receipt_id), sorted by unit_id. Identical
    to the coordinator's construction so a match proves the set is unchanged."""
    if not units:
        return hashlib.sha256(_LEAF_PREFIX).hexdigest()
    ordered = sorted(units, key=lambda u: u["unit_id"])
    level = [_leaf_bytes(u) for u in ordered]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            hashlib.sha256(_NODE_PREFIX + level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


@dataclass(frozen=True)
class ResultSetAttestation:
    """Parsed `GET /experiments/{id}/attestation` response."""

    attestation_id: str
    experiment_id: str
    tenant_id: str
    merkle_root: str
    algorithm: str
    unit_count: int
    units: list[dict[str, Any]]
    cose_b64: str
    signing_key_pubkey_hex: str
    rekor_log_index: int
    rekor_entry_uuid: str
    # M9 leg 2: True when this is a checkpoint over a not-yet-complete experiment
    # (consensus-so-far). The same flag is in the COSE-signed predicate.
    partial: bool = False

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> ResultSetAttestation:
        return cls(
            attestation_id=body["attestation_id"],
            experiment_id=body["experiment_id"],
            tenant_id=body["tenant_id"],
            merkle_root=body["merkle_root"],
            algorithm=body.get("algorithm", RESULT_SET_ALGORITHM),
            unit_count=body.get("unit_count", 0),
            units=body.get("units") or [],
            cose_b64=body["cose_b64"],
            signing_key_pubkey_hex=body["signing_key_pubkey_hex"],
            rekor_log_index=body.get("rekor_log_index", 0),
            rekor_entry_uuid=body.get("rekor_entry_uuid", ""),
            partial=bool(body.get("partial", False)),
        )


@dataclass(frozen=True)
class AttestationVerification:
    """Outcome of `verify_attestation`. `ok` is the AND of every check that was
    requested."""

    signature_valid: bool
    signer_authorized: bool
    root_matches_units: bool  # recomputed root over the response units == signed root
    signed_root_matches: bool  # the COSE-signed predicate's root == response root
    signer_pubkey_hex: str | None
    attested_root: str
    recomputed_root: str

    @property
    def ok(self) -> bool:
        return (
            self.signature_valid
            and self.signer_authorized
            and self.root_matches_units
            and self.signed_root_matches
        )


class AttestationVerificationError(Exception):
    """Raised when the COSE blob is malformed (not when a check merely fails —
    those are reported in the AttestationVerification flags)."""


def _cose_verify(cose_bytes: bytes) -> tuple[bool, str | None, bytes]:
    """Decode a COSE_Sign1, verify its Ed25519 signature against the pubkey in
    the protected header's kid. Returns (signature_valid, signer_pubkey_hex,
    payload_bytes). Mirrors the coordinator's `cose_sign1_encode` structure."""
    try:
        outer = cbor2.loads(cose_bytes)
    except Exception as exc:
        raise AttestationVerificationError(f"COSE outer CBOR malformed: {exc}") from exc
    if not isinstance(outer, list) or len(outer) != 4:
        raise AttestationVerificationError("COSE_Sign1 must be a 4-element array")
    protected_bytes, _unprotected, payload, signature = outer
    if not (
        isinstance(protected_bytes, bytes)
        and isinstance(payload, bytes)
        and isinstance(signature, bytes)
    ):
        raise AttestationVerificationError("COSE_Sign1 fields have wrong types")
    try:
        protected = cbor2.loads(protected_bytes) if protected_bytes else {}
    except Exception as exc:
        raise AttestationVerificationError(f"COSE protected header malformed: {exc}") from exc
    kid = protected.get(_COSE_HEADER_KID) if isinstance(protected, dict) else None
    signer_hex = kid.decode("ascii") if isinstance(kid, bytes) else None
    if signer_hex is None:
        return False, None, payload
    sig_structure = cbor2.dumps(["Signature1", protected_bytes, b"", payload], canonical=True)
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(signer_hex))
        pubkey.verify(signature, sig_structure)
        valid = True
    except (InvalidSignature, ValueError):
        valid = False
    return valid, signer_hex, payload


def verify_attestation(
    att: ResultSetAttestation,
    *,
    authorized_signers: list[str] | None = None,
) -> AttestationVerification:
    """Independently verify a result-set attestation:

      1. recompute the Merkle root over the response `units` → must equal the
         attestation's `merkle_root` (the response is internally consistent);
      2. verify the COSE Ed25519 signature against the signer in the protected
         header, and (if `authorized_signers` is given) that the signer is
         on the trusted list (e.g. AUTHORIZED_SIGNERS.md pubkeys);
      3. confirm the COSE-signed predicate's `merkle_root` equals the response
         root (the response wasn't altered relative to the signed artifact).

    `ok` is true only if all checks pass. To prove the *set itself* is the one
    you reduced, also recompute `merkle_root(your_pulled_units)` and compare to
    `att.merkle_root` — see `verify_against_results`."""
    from base64 import b64decode

    recomputed = merkle_root(att.units)
    root_matches_units = recomputed == att.merkle_root

    signature_valid, signer_hex, payload = _cose_verify(b64decode(att.cose_b64))
    if authorized_signers is None:
        signer_authorized = True
    else:
        signer_authorized = signer_hex is not None and signer_hex.lower() in {
            s.lower() for s in authorized_signers
        }

    signed_root_matches = False
    try:
        statement = cbor2.loads(payload)
        predicate = cbor2.loads(statement["predicate"])
        signed_root_matches = (
            statement.get("predicateType") == RESULT_SET_PREDICATE_TYPE
            and predicate.get("merkle_root") == att.merkle_root
        )
    except Exception:
        signed_root_matches = False

    return AttestationVerification(
        signature_valid=signature_valid,
        signer_authorized=signer_authorized,
        root_matches_units=root_matches_units,
        signed_root_matches=signed_root_matches,
        signer_pubkey_hex=signer_hex,
        attested_root=att.merkle_root,
        recomputed_root=recomputed,
    )


def verify_against_results(att: ResultSetAttestation, results: list[dict[str, Any]]) -> bool:
    """True iff a Merkle root recomputed over a *separately pulled* result set
    equals the attested root. `results` are result rows (e.g. from
    `TenantClient.iter_results`) carrying unit_id / semantic_hash / receipt_id;
    `semantic_hash` is the consensus_result_hash. This is the strong
    reproducibility check: it proves the set you reduced is the attested one."""
    units = [
        {
            "unit_id": r["unit_id"],
            "consensus_result_hash": r.get("semantic_hash") or r.get("consensus_result_hash"),
            "receipt_id": r["receipt_id"],
        }
        for r in results
    ]
    return merkle_root(units) == att.merkle_root
