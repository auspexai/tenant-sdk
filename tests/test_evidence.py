"""Evidence-bundle verification (EB-1, §9 #47) — the full custody chain.

Builds synthetic bundles signed exactly as the coordinator + worker do
(coordinator: COSE-Sign1 attestation + Ed25519 proof-of-transfer; worker:
signing/result.py canonical-bytes convention) and exercises every named check
of `verify_bundle`, including the failure modes the design doc ratified:
external key pinning (D4), at-rest payload tamper (worker sigs, D2),
input-binding tamper, and completeness (missing rows).
"""

from __future__ import annotations

import hashlib
import json
from base64 import b64decode, b64encode

import cbor2
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_tenant.attestation import (
    RESULT_SET_ALGORITHM_V1,
    RESULT_SET_PREDICATE_TYPE_V1,
    merkle_root,
    unit_payload_sha256,
)
from auspexai_tenant.evidence import verify_bundle, verify_worker_signatures

_ALG, _KID, _EDDSA = 1, 4, -8


def _pub_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _sign_v1_attestation(
    units: list[dict],
    key: Ed25519PrivateKey,
    *,
    footprint: dict | None = None,
    diverged_units: list[dict] | None = None,
) -> dict:
    """A bundle `attestation` block, COSE-signed as the coordinator's v1 path."""
    root = merkle_root(units, algorithm=RESULT_SET_ALGORITHM_V1)
    predicate = {
        "merkle_root": root,
        "algorithm": RESULT_SET_ALGORITHM_V1,
        "experiment_id": "exp-label",
        "tenant_id": "tenant-a",
        "unit_count": len(units),
        "units": sorted(units, key=lambda u: u["unit_id"]),
    }
    if diverged_units:
        predicate["diverged_units"] = sorted(diverged_units, key=lambda d: d["unit_id"])
    if footprint is not None:
        predicate["governance_footprint"] = footprint
    predicate_cbor = cbor2.dumps(predicate, canonical=True)
    statement = {
        "_type": "https://www.in-toto.io/Statement/v1",
        "subject": [
            {
                "name": "auspexai:result-set/att-eb1",
                "digest": {"sha256": hashlib.sha256(predicate_cbor).hexdigest()},
            }
        ],
        "predicateType": RESULT_SET_PREDICATE_TYPE_V1,
        "predicate": predicate_cbor,
    }
    statement_cbor = cbor2.dumps(statement, canonical=True)
    protected = cbor2.dumps({_ALG: _EDDSA, _KID: _pub_hex(key).encode("ascii")}, canonical=True)
    sig_structure = cbor2.dumps(["Signature1", protected, b"", statement_cbor], canonical=True)
    cose = cbor2.dumps([protected, {}, statement_cbor, key.sign(sig_structure)], canonical=True)
    return {
        "attestation_id": "att-eb1",
        "merkle_root": root,
        "algorithm": RESULT_SET_ALGORITHM_V1,
        "cose_b64": b64encode(cose).decode(),
        "signing_key_pubkey_hex": _pub_hex(key),
        "rekor_log_index": 0,
        "rekor_entry_uuid": "lab-mode-no-rekor",
        "rekor_inclusion_proof": None,
    }


def _sign_worker_result(worker_key: Ed25519PrivateKey, r: dict) -> str:
    body = {
        "unit_id": r["unit_id"],
        "worker_pubkey": r["worker_pubkey_hex"].lower(),
        "completed_at": r["completed_at"],
        "exit_code": int(r["exit_code"]),
        "payload": r["payload"],
    }
    sv = r.get("schema_version")
    if sv and int(sv) >= 1:  # §9 #13a: v1 binds the served-weights digest
        body["schema_version"] = int(sv)
        body["served_weights"] = {
            str(k): str(v).lower() for k, v in (r.get("served_weights") or {}).items()
        }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return b64encode(worker_key.sign(canonical)).decode()


def _make_bundle(
    coordinator_key: Ed25519PrivateKey,
    worker_key: Ed25519PrivateKey,
    *,
    n: int = 2,
    with_attestation: bool = True,
    unit_basis: str | None = None,
    footprint: dict | None = None,
    diverged_units: list[dict] | None = None,
    served_weights: dict | None = None,
) -> dict:
    work_units = [{"unit_id": f"u{i}", "payload": {"q": i}} for i in range(n)]
    consensus = []
    att_units = []
    for i in range(n):
        sem = hashlib.sha256(f"consensus-{i}".encode()).hexdigest()
        r = {
            "result_id": f"res-u{i}",
            "unit_id": f"u{i}",
            "semantic_hash": sem,
            "payload": {"a": i},
            "aged_off": False,
            "worker_pubkey_hex": _pub_hex(worker_key),
            "exit_code": 0,
            "receipt_id": f"rcpt-u{i}",
            "completed_at": f"2026-06-11T0{i}:00:00+00:00",
        }
        if served_weights is not None:  # §9 #13a: v1 worker-attested digest
            r["schema_version"] = 1
            r["served_weights"] = served_weights
        r["worker_signature"] = _sign_worker_result(worker_key, r)
        consensus.append(r)
        att_unit = {
            "unit_id": f"u{i}",
            "consensus_result_hash": sem,
            "receipt_id": f"rcpt-u{i}",
            "unit_payload_sha256": unit_payload_sha256({"q": i}),
        }
        if unit_basis is not None:
            att_unit["integrity_basis"] = unit_basis
        att_units.append(att_unit)
    bundle: dict = {
        "schema": "auspexai-evidence-bundle/v1",
        "experiment_id": "exp-123",
        "manifest_hash": "ab" * 32,
        "manifest": {"experiment_id": "exp-label"},
        "work_units": work_units,
        "consensus_results": consensus,
        "receipts": [],
        "attestation": None,
    }
    if with_attestation:
        bundle["attestation"] = _sign_v1_attestation(
            att_units, coordinator_key, footprint=footprint, diverged_units=diverged_units
        )
        root = bundle["attestation"]["merkle_root"]
        root_kind = RESULT_SET_ALGORITHM_V1
    else:
        items = sorted((r["unit_id"], r["semantic_hash"]) for r in consensus)
        root = hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()
        root_kind = "flat-v0"
    collected_by = "cd" * 32
    collected_at = "2026-06-11T12:00:00+00:00"
    record = f"{root}|{collected_by}|{collected_at}|{'ab' * 32}".encode()
    bundle["transfer"] = {
        "transfer_id": "xfer-test",
        "result_set_root": root,
        "root_kind": root_kind,
        "attestation_id": "att-eb1" if with_attestation else None,
        "collected_at": collected_at,
        "collected_by_pubkey": collected_by,
        "manifest_hash": "ab" * 32,
        "receipt_count": 0,
        "coordinator_signature": coordinator_key.sign(record).hex(),
        "coordinator_pubkey_hex": _pub_hex(coordinator_key),
    }
    return bundle


def _keys() -> tuple[Ed25519PrivateKey, Ed25519PrivateKey]:
    return Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()


def test_full_bundle_verifies():
    ck, wk = _keys()
    v = verify_bundle(_make_bundle(ck, wk))
    assert v.transfer_signature_valid
    assert v.attestation is not None and v.attestation.ok
    assert v.root_unified is True
    assert v.completeness_ok is True
    assert v.inputs_bound_ok is True
    assert v.worker_signatures.verified == 2 and not v.worker_signatures.failed
    assert v.ok


def _counts(exact=0, tolerance=0, process=0, diverged=0):
    return {
        "within_cell_exact": exact,
        "within_cell_tolerance": tolerance,
        "process_only": process,
        "diverged": diverged,
    }


def test_footprint_recompute_passes_and_surfaces():
    """Firewall #2 F6 (consumer): a footprint whose integrity_basis counts match a
    recount of the signed predicate's per-unit basis verifies, and the footprint
    surfaces for the researcher."""
    ck, wk = _keys()
    fp = {
        "schema_version": 1,
        "tenant": {"tier": "T2"},
        "integrity_basis": {"counts": _counts(exact=2)},
    }
    v = verify_bundle(_make_bundle(ck, wk, n=2, unit_basis="within_cell_exact", footprint=fp))
    assert v.footprint_ok is True
    assert v.governance_footprint["tenant"]["tier"] == "T2"
    assert v.ok


def test_footprint_counts_include_diverged_units():
    """diverged_units count toward the `diverged` basis — a footprint claiming the
    divergence recomputes."""
    ck, wk = _keys()
    fp = {"integrity_basis": {"counts": _counts(exact=2, diverged=1)}}
    diverged = [
        {
            "unit_id": "u9",
            "unit_payload_sha256": "",
            "result_hashes": ["a", "b"],
            "integrity_basis": "diverged",
        }
    ]
    v = verify_bundle(
        _make_bundle(
            ck, wk, n=2, unit_basis="within_cell_exact", footprint=fp, diverged_units=diverged
        )
    )
    assert v.footprint_ok is True
    assert v.ok


def test_footprint_count_tamper_fails():
    """A signed footprint that overstates its counts (vs the signed unit list)
    fails — the consumer recount catches a coordinator-side inconsistency even
    though the COSE signature is valid."""
    ck, wk = _keys()
    fp = {"integrity_basis": {"counts": _counts(exact=5)}}  # lies — really 2
    v = verify_bundle(_make_bundle(ck, wk, n=2, unit_basis="within_cell_exact", footprint=fp))
    assert v.footprint_ok is False
    assert not v.ok


def test_no_footprint_is_lenient():
    """A pre-firewall attestation (no governance_footprint) is not failed for it —
    footprint_ok is None, the bundle still verifies."""
    ck, wk = _keys()
    v = verify_bundle(_make_bundle(ck, wk))
    assert v.footprint_ok is None
    assert v.governance_footprint is None
    assert v.ok


def test_external_pinning_rejects_unlisted_coordinator_key():
    """D4: a valid signature by a key the bundle itself carries proves nothing —
    pinning against an external signer list must fail an unlisted key."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    v = verify_bundle(bundle, authorized_signers=["ef" * 32])
    assert v.transfer_signature_valid  # the math is fine...
    assert not v.transfer_signer_authorized  # ...but the key isn't pinned
    assert v.signer_pin_mode == "explicit" and not v.signer_grounded
    assert not v.ok
    # and with the real key listed, it passes.
    ok = verify_bundle(bundle, authorized_signers=[_pub_hex(ck)])
    assert ok.transfer_signer_authorized and ok.ok
    assert ok.signer_pin_mode == "explicit" and ok.signer_grounded


def test_default_pin_unpinned_for_unknown_key():
    """Default (no args): an unrecognized signer is reported UNPINNED but does
    NOT fail — the bundle is still self-consistent."""
    ck, wk = _keys()
    v = verify_bundle(_make_bundle(ck, wk))
    assert v.signer_pin_mode == "unpinned"
    assert not v.signer_grounded
    assert v.transfer_signer_authorized  # soft pin never turns into a false FAIL
    assert v.ok


def test_default_pin_grounds_known_public_key(monkeypatch):
    """Default: a signer in the embedded KNOWN_PUBLIC_SIGNERS grounds trust
    ('known') with no caller --signer — the public-network ergonomics win."""
    import auspexai_tenant.evidence as ev

    ck, wk = _keys()
    monkeypatch.setattr(ev, "KNOWN_PUBLIC_SIGNERS", (_pub_hex(ck),))
    v = verify_bundle(_make_bundle(ck, wk))
    assert v.signer_pin_mode == "known"
    assert v.signer_grounded
    assert v.ok


def test_no_pin_skips_grounding():
    """--no-pin path: grounding is skipped entirely (self-consistency only),
    still never a false FAIL."""
    ck, wk = _keys()
    v = verify_bundle(_make_bundle(ck, wk), no_pin=True)
    assert v.signer_pin_mode == "skipped"
    assert not v.signer_grounded
    assert v.ok


def test_tampered_payload_fails_worker_signature():
    """D2: the worker signature is the only non-coordinator key in the chain —
    an at-rest payload tamper must surface here."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"][0]["payload"] = {"a": "TAMPERED"}
    v = verify_bundle(bundle)
    assert v.worker_signatures.failed == ["res-u0"]
    assert not v.ok


def test_v1_served_weights_covered_by_worker_signature():
    """§9 #13a: a v1 result binds its served-weights digest into the signed
    body, so verifying the worker signature also verifies the digest — and a
    tampered served_weights surfaces as a signature failure (the coordinator
    never touches this key, so this is the researcher's independent check that
    the declared model is the one that ran)."""
    wk = Ed25519PrivateKey.generate()
    r = {
        "result_id": "res-u0",
        "unit_id": "u0",
        "payload": {"a": 0},
        "aged_off": False,
        "worker_pubkey_hex": _pub_hex(wk),
        "exit_code": 0,
        "completed_at": "2026-06-15T00:00:00+00:00",
        "schema_version": 1,
        "served_weights": {"gemma": "abc123"},
    }
    r["worker_signature"] = _sign_worker_result(wk, r)

    ok = verify_worker_signatures([r])
    assert ok.verified == 1 and not ok.failed

    tampered = dict(r, served_weights={"gemma": "deadbeef"})
    bad = verify_worker_signatures([tampered])
    assert bad.failed == ["res-u0"] and not bad.ok


def test_tampered_work_unit_breaks_input_binding():
    """Reproducibility triple, input leg: 'result R came from parameters P'."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["work_units"][1]["payload"] = {"q": "NOT-WHAT-RAN"}
    v = verify_bundle(bundle)
    assert v.inputs_bound_ok is False
    assert not v.ok


def test_missing_result_breaks_completeness():
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"].pop()
    v = verify_bundle(bundle)
    assert v.completeness_ok is False
    assert not v.ok


def test_flat_root_bundle_skips_attestation_checks():
    """Pre-completion export: no attestation, flat custody root — tri-state
    checks are n/a (None), and the bundle still verifies on its own terms."""
    ck, wk = _keys()
    v = verify_bundle(_make_bundle(ck, wk, with_attestation=False))
    assert v.attestation is None
    assert v.root_unified is None
    assert v.completeness_ok is None
    assert v.inputs_bound_ok is None
    assert v.transfer_signature_valid and v.worker_signatures.ok
    assert v.ok


def test_aged_off_rows_are_skipped_not_failed():
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"][0]["payload"] = None
    bundle["consensus_results"][0]["aged_off"] = True
    report = verify_worker_signatures(bundle["consensus_results"])
    assert report.skipped_aged_off == 1
    assert report.verified == 1
    assert report.ok


def test_unified_root_mismatch_detected():
    """A custody record claiming attestation-root binding must actually match
    the attestation's root."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    fake_root = "00" * 32
    t = bundle["transfer"]
    record = (
        f"{fake_root}|{t['collected_by_pubkey']}|{t['collected_at']}|{t['manifest_hash']}"
    ).encode()
    t["result_set_root"] = fake_root
    t["coordinator_signature"] = ck.sign(record).hex()
    v = verify_bundle(bundle)
    assert v.transfer_signature_valid  # the record itself is well-signed...
    assert v.root_unified is False  # ...but it doesn't bind the attestation
    assert not v.ok


# ---- adversarial negative pass (2026-06-12, reviewer rec ~80% pre-tenant-#2) --


def test_stripped_worker_sig_fields_fail_on_v1_bundle():
    """Strip-the-signature tamper: removing worker_pubkey_hex/exit_code used to
    pass vacuously (verified=0, failed=[], skipped_missing>0). On a v1-schema
    bundle the members are guaranteed, so absence must FAIL verification."""
    coord, worker = _keys()
    bundle = _make_bundle(coord, worker)
    for r in bundle["consensus_results"]:
        r.pop("worker_pubkey_hex", None)
        r.pop("exit_code", None)
    v = verify_bundle(bundle)
    assert v.worker_signatures.verified == 0
    assert v.worker_signatures.skipped_missing_fields > 0
    assert not v.worker_signatures.ok
    assert not v.ok


def test_stripped_fields_stay_lenient_without_schema_member():
    """Pre-EB-1 bundles never carried the worker-sig members — absence there
    is an old coordinator, not a tamper; the legacy lenient skip survives."""
    coord, worker = _keys()
    bundle = _make_bundle(coord, worker)
    bundle.pop("schema")
    for r in bundle["consensus_results"]:
        r.pop("worker_pubkey_hex", None)
        r.pop("exit_code", None)
    v = verify_bundle(bundle)
    assert v.worker_signatures.skipped_missing_fields > 0
    assert v.worker_signatures.ok  # lenient: legacy bundle
    assert v.ok


def test_unknown_future_schema_refuses_with_upgrade_hint():
    """Cross-version: an SDK must refuse a bundle schema it doesn't know
    rather than verify whatever subset happens to parse."""
    coord, worker = _keys()
    bundle = _make_bundle(coord, worker)
    bundle["schema"] = "auspexai-evidence-bundle/v2"
    with pytest.raises(ValueError, match="upgrade the SDK"):
        verify_bundle(bundle)


def test_tampered_transfer_signature_fails():
    coord, worker = _keys()
    bundle = _make_bundle(coord, worker)
    sig = bytearray(bytes.fromhex(bundle["transfer"]["coordinator_signature"]))
    sig[0] ^= 0xFF
    bundle["transfer"]["coordinator_signature"] = bytes(sig).hex()
    v = verify_bundle(bundle)
    assert not v.transfer_signature_valid
    assert not v.ok


def test_tampered_attestation_cose_fails():
    coord, worker = _keys()
    bundle = _make_bundle(coord, worker)
    blob = bytearray(b64decode(bundle["attestation"]["cose_b64"]))
    blob[-1] ^= 0xFF
    bundle["attestation"]["cose_b64"] = b64encode(bytes(blob)).decode()
    v = verify_bundle(bundle)
    assert v.attestation is not None and not v.attestation.ok
    assert not v.ok
