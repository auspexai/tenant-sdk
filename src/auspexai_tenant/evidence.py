"""Evidence-bundle verification (EB-1, principles §9 #47).

The coordinator's `GET /experiments/{id}/results/export` returns the
**evidence bundle** (`schema: auspexai-evidence-bundle/v1`): the signed
manifest + work-unit INPUTS + consensus results + COSE receipts + the COSE
result-set attestation (with its Rekor anchor and, when captured, the
inclusion proof) + a signed proof-of-transfer. `verify_bundle` runs the full
chain a researcher needs for custody confidence, offline by default:

  1. proof-of-transfer signature — and (with `authorized_signers`) that the
     signing key is EXTERNALLY pinned, not merely the one the bundle claims;
  2. the attestation itself (COSE signature, root recompute over its units,
     signed-root match, optional online Rekor inclusion);
  3. root unification — the custody record signs the attestation's merkle
     root (`transfer.root_kind` != flat-v0), binding data ↔ custody ↔ Rekor;
  4. completeness — the delivered consensus set is EXACTLY the attested leaf
     set: nothing missing, nothing extra;
  5. input binding (v1 attestations) — every leaf's `unit_payload_sha256`
     recomputes from the bundled work-unit payloads: "result R was produced
     from parameters P" verified from bytes in hand;
  6. worker signatures — the only signatures in the chain NOT made by the
     coordinator's key (the worker daemon's `signing/result.py` convention),
     removing the coordinator from the trusted base for data authenticity.

Aged-off results (payload withheld) can't have their worker signature or
payload recomputed — they are counted as skipped, never as failures: the
receipt + semantic_hash still prove the unit ran.
"""

from __future__ import annotations

import json
from base64 import b64decode
from dataclasses import dataclass
from typing import Any

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auspexai_tenant.attestation import (
    DEFAULT_REKOR_URL,
    RESULT_SET_ALGORITHM_V1,
    AttestationVerification,
    ResultSetAttestation,
    unit_payload_sha256,
    verify_attestation,
)

EVIDENCE_BUNDLE_SCHEMA = "auspexai-evidence-bundle/v1"
FLAT_ROOT_KIND = "flat-v0"


@dataclass(frozen=True)
class WorkerSignatureReport:
    """Per-result worker-signature verification over the bundle's consensus
    rows. `failed` lists result_ids whose signature did NOT verify — any entry
    is a hard failure. Skips are not failures: aged-off rows have no payload to
    recompute over; missing-field rows come from pre-EB-1 coordinators that
    didn't ship `worker_pubkey_hex`/`exit_code` in the bundle."""

    verified: int
    failed: list[str]
    skipped_aged_off: int
    skipped_missing_fields: int

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass(frozen=True)
class BundleVerification:
    """The named checks of `verify_bundle`. Tri-state fields are None when the
    check was not applicable (no attestation in the bundle / flat-root custody
    / v0 attestation without input binding) — None never fails `ok`; an
    explicit False always does."""

    transfer_signature_valid: bool
    transfer_signer_authorized: bool
    attestation: AttestationVerification | None
    root_unified: bool | None
    completeness_ok: bool | None
    inputs_bound_ok: bool | None
    worker_signatures: WorkerSignatureReport

    @property
    def ok(self) -> bool:
        return (
            self.transfer_signature_valid
            and self.transfer_signer_authorized
            and (self.attestation is None or self.attestation.ok)
            and self.root_unified is not False
            and self.completeness_ok is not False
            and self.inputs_bound_ok is not False
            and self.worker_signatures.ok
        )


def _attestation_from_bundle(block: dict[str, Any]) -> ResultSetAttestation:
    """Reconstruct a `ResultSetAttestation` from the bundle's attestation
    block. The block carries the COSE artifact but not the convenience `units`
    list — recover units (and the predicate-only fields) from the SIGNED
    statement itself, which is the stronger source anyway."""
    cose = cbor2.loads(b64decode(block["cose_b64"]))
    statement = cbor2.loads(cose[2])  # [protected, unprotected, payload, sig]
    predicate = cbor2.loads(statement["predicate"])
    return ResultSetAttestation(
        attestation_id=block["attestation_id"],
        experiment_id=predicate.get("experiment_id", ""),
        tenant_id=predicate.get("tenant_id", ""),
        merkle_root=block["merkle_root"],
        algorithm=block["algorithm"],
        unit_count=predicate.get("unit_count", 0),
        units=predicate.get("units") or [],
        cose_b64=block["cose_b64"],
        signing_key_pubkey_hex=block["signing_key_pubkey_hex"],
        rekor_log_index=block.get("rekor_log_index", 0),
        rekor_entry_uuid=block.get("rekor_entry_uuid", ""),
        partial=bool(predicate.get("partial", False)),
        rekor_inclusion_proof=block.get("rekor_inclusion_proof"),
    )


def _canonical_result_bytes(r: dict[str, Any]) -> bytes:
    """The worker daemon's canonical signing input (signing/result.py) —
    reproduced byte-for-byte from bundle fields."""
    body = {
        "unit_id": r["unit_id"],
        "worker_pubkey": r["worker_pubkey_hex"].lower(),
        "completed_at": r["completed_at"],
        "exit_code": int(r["exit_code"]),
        "payload": r["payload"],
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_worker_signatures(consensus_results: list[dict[str, Any]]) -> WorkerSignatureReport:
    """Verify each consensus result's worker signature — the only signature in
    the chain not made by the coordinator (defense in depth against an at-rest
    or coordinator-level tamper)."""
    verified = 0
    failed: list[str] = []
    skipped_aged = 0
    skipped_missing = 0
    for r in consensus_results:
        if r.get("aged_off") or r.get("payload") is None:
            skipped_aged += 1
            continue
        if not r.get("worker_pubkey_hex") or r.get("exit_code") is None:
            skipped_missing += 1
            continue
        try:
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(r["worker_pubkey_hex"]))
            pub.verify(b64decode(r["worker_signature"]), _canonical_result_bytes(r))
            verified += 1
        except (InvalidSignature, KeyError, ValueError):
            failed.append(r.get("result_id") or r.get("unit_id") or "<unknown>")
    return WorkerSignatureReport(
        verified=verified,
        failed=failed,
        skipped_aged_off=skipped_aged,
        skipped_missing_fields=skipped_missing,
    )


def verify_bundle(
    bundle: dict[str, Any],
    *,
    authorized_signers: list[str] | None = None,
    check_rekor: bool = False,
    rekor_url: str = DEFAULT_REKOR_URL,
) -> BundleVerification:
    """Run the full evidence-bundle verification chain (module docstring).
    Offline by default; `check_rekor=True` adds the online transparency-log
    inclusion check. Pass `authorized_signers` (e.g. the AUTHORIZED_SIGNERS.md
    pubkeys) to pin BOTH the custody and attestation signing keys externally —
    without it, a valid signature only proves consistency with the key the
    bundle itself carries."""
    t = bundle["transfer"]

    # 1. proof-of-transfer signature + external pinning
    record = (
        f"{t['result_set_root']}|{t['collected_by_pubkey']}|"
        f"{t['collected_at']}|{t['manifest_hash']}"
    ).encode()
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(t["coordinator_pubkey_hex"]))
        pub.verify(bytes.fromhex(t["coordinator_signature"]), record)
        transfer_sig_ok = True
    except (InvalidSignature, KeyError, ValueError):
        transfer_sig_ok = False
    if authorized_signers is None:
        signer_authorized = True
    else:
        signer_authorized = t["coordinator_pubkey_hex"].lower() in {
            s.lower() for s in authorized_signers
        }

    consensus = bundle.get("consensus_results") or []

    # 2-5. attestation chain (when the bundle carries one)
    att_verification: AttestationVerification | None = None
    root_unified: bool | None = None
    completeness: bool | None = None
    inputs_bound: bool | None = None
    block = bundle.get("attestation")
    if block:
        att = _attestation_from_bundle(block)
        att_verification = verify_attestation(
            att,
            authorized_signers=authorized_signers,
            check_rekor=check_rekor,
            rekor_url=rekor_url,
        )
        if t.get("root_kind") and t["root_kind"] != FLAT_ROOT_KIND:
            root_unified = t["result_set_root"] == att.merkle_root
        delivered = {(r["unit_id"], r.get("semantic_hash") or "") for r in consensus}
        attested = {(u["unit_id"], u["consensus_result_hash"]) for u in att.units}
        completeness = delivered == attested
        if att.algorithm == RESULT_SET_ALGORITHM_V1 and att.units:
            payloads = {w["unit_id"]: w.get("payload") for w in bundle.get("work_units") or []}
            inputs_bound = all(
                bool(u.get("unit_payload_sha256"))
                and payloads.get(u["unit_id"]) is not None
                and unit_payload_sha256(payloads[u["unit_id"]]) == u["unit_payload_sha256"]
                for u in att.units
            )

    # 6. worker signatures
    ws = verify_worker_signatures(consensus)

    return BundleVerification(
        transfer_signature_valid=transfer_sig_ok,
        transfer_signer_authorized=signer_authorized,
        attestation=att_verification,
        root_unified=root_unified,
        completeness_ok=completeness,
        inputs_bound_ok=inputs_bound,
        worker_signatures=ws,
    )


# ---- the evidence loader (§9 #47 §6) ----------------------------------------


class BundleVerificationError(Exception):
    """Raised by `load_verified` when the bundle fails verification. Carries
    the full `BundleVerification` so the caller can see WHICH check failed —
    but the loader never returns data from a bundle that doesn't verify."""

    def __init__(self, verification: BundleVerification) -> None:
        self.verification = verification
        failed = []
        v = verification
        if not v.transfer_signature_valid:
            failed.append("transfer signature")
        if not v.transfer_signer_authorized:
            failed.append("transfer signer not in authorized list")
        if v.attestation is not None and not v.attestation.ok:
            failed.append("attestation")
        if v.root_unified is False:
            failed.append("custody/attestation root mismatch")
        if v.completeness_ok is False:
            failed.append("delivered set != attested set")
        if v.inputs_bound_ok is False:
            failed.append("input binding (work-unit payload hashes)")
        if v.worker_signatures.failed:
            failed.append(f"worker signatures ({', '.join(v.worker_signatures.failed)})")
        super().__init__("evidence bundle failed verification: " + "; ".join(failed))


def _read_bundle(bundle: Any) -> dict[str, Any]:
    """Accept a parsed bundle dict or a path to a saved bundle JSON file."""
    if isinstance(bundle, dict):
        return bundle
    from pathlib import Path

    return json.loads(Path(bundle).read_text(encoding="utf-8"))


def load_verified(
    bundle: Any,
    *,
    authorized_signers: list[str] | None = None,
    check_rekor: bool = False,
    rekor_url: str = DEFAULT_REKOR_URL,
):
    """Verify an evidence bundle, then return its results as a pandas
    DataFrame — analysis that BEGINS from a cryptographically verified
    dataset, in one call. Refuses (raises `BundleVerificationError`) if any
    check fails; there is deliberately no force/skip flag — to inspect a bad
    bundle, call `verify_bundle` directly.

    `bundle` is the dict from `TenantClient.export()` or a path to the JSON
    file `auspexai-tenant experiment export` saved.

    One row per consensus result. Columns: `unit_id`, `result_id`,
    `receipt_id`, `completed_at` (datetime), `semantic_hash`, `aged_off`,
    plus the work-unit INPUT payload flattened under `input.*` and the result
    OUTPUT payload flattened under `output.*` (NaN for aged-off rows — their
    receipt + semantic_hash still verify, but the payload is gone; collection
    is custody transfer). Round/sweep coordinates are tenant semantics — they
    live in your `input.*` columns or your unit-id convention, not in the
    platform's schema.

    Requires the `analysis` extra: `pip install auspexai-tenant[analysis]`.
    From here, `df.to_csv(...)` / `df.to_parquet(...)` open the Excel /
    Tableau / R door — or use the `auspexai-tenant bundle table` CLI.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "load_verified needs pandas — install the analysis extra: "
            "pip install 'auspexai-tenant[analysis]'"
        ) from e

    data = _read_bundle(bundle)
    verification = verify_bundle(
        data,
        authorized_signers=authorized_signers,
        check_rekor=check_rekor,
        rekor_url=rekor_url,
    )
    if not verification.ok:
        raise BundleVerificationError(verification)

    inputs = {w["unit_id"]: w.get("payload") or {} for w in data.get("work_units") or []}
    rows = []
    for r in data.get("consensus_results") or []:
        row: dict[str, Any] = {
            "unit_id": r["unit_id"],
            "result_id": r.get("result_id"),
            "receipt_id": r.get("receipt_id"),
            "completed_at": r.get("completed_at"),
            "semantic_hash": r.get("semantic_hash"),
            "aged_off": bool(r.get("aged_off")),
        }
        for k, v in (inputs.get(r["unit_id"]) or {}).items():
            row[f"input.{k}"] = v
        for k, v in (r.get("payload") or {}).items():
            row[f"output.{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["completed_at"] = pd.to_datetime(df["completed_at"], format="ISO8601")
        df = df.sort_values("unit_id", kind="stable").reset_index(drop=True)
    return df
