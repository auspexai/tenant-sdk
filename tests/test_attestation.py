"""Result-set completion attestation — tenant-side verify (#34 §6.3).

Self-contained: builds a COSE-signed in-toto statement the exact way the
coordinator does (a local Ed25519 key), then exercises `verify_attestation` /
`verify_against_results` without needing a live coordinator.
"""

from __future__ import annotations

import hashlib
import json
from base64 import b64encode

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_tenant.attestation import (
    RESULT_SET_PREDICATE_TYPE,
    ResultSetAttestation,
    merkle_root,
    verify_against_results,
    verify_attestation,
)

_ALG, _KID, _EDDSA = 1, 4, -8


def _unit(uid: str, h: str = "deadbeef", rid: str = "rcpt-x") -> dict:
    return {"unit_id": uid, "consensus_result_hash": h, "receipt_id": rid}


def _sign_attestation(units: list[dict], key: Ed25519PrivateKey) -> dict:
    """Produce a `GET /attestation` response body, COSE-signed exactly as the
    coordinator's signing.py does."""
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    root = merkle_root(units)
    predicate = {
        "merkle_root": root,
        "algorithm": "sha256-merkle-v0",
        "experiment_id": "exp-label",
        "tenant_id": "tenant-a",
        "unit_count": len(units),
        "units": sorted(units, key=lambda u: u["unit_id"]),
    }
    predicate_cbor = cbor2.dumps(predicate, canonical=True)
    digest = hashlib.sha256(predicate_cbor).hexdigest()
    statement = {
        "_type": "https://www.in-toto.io/Statement/v1",
        "subject": [{"name": "auspexai:result-set/att-x", "digest": {"sha256": digest}}],
        "predicateType": RESULT_SET_PREDICATE_TYPE,
        "predicate": predicate_cbor,
    }
    statement_cbor = cbor2.dumps(statement, canonical=True)
    protected = cbor2.dumps({_ALG: _EDDSA, _KID: pub_hex.encode("ascii")}, canonical=True)
    sig_structure = cbor2.dumps(["Signature1", protected, b"", statement_cbor], canonical=True)
    signature = key.sign(sig_structure)
    cose = cbor2.dumps([protected, {}, statement_cbor, signature], canonical=True)
    return {
        "attestation_id": "att-x",
        "experiment_id": "exp-label",
        "tenant_id": "tenant-a",
        "merkle_root": root,
        "algorithm": "sha256-merkle-v0",
        "unit_count": len(units),
        "units": sorted(units, key=lambda u: u["unit_id"]),
        "cose_b64": b64encode(cose).decode(),
        "signing_key_pubkey_hex": pub_hex,
        "rekor_log_index": 0,
        "rekor_entry_uuid": "lab-mode-no-rekor",
    }


# ---- merkle ----------------------------------------------------------------


def test_merkle_order_independent_and_sensitive():
    a = [_unit("u1"), _unit("u2"), _unit("u3")]
    assert merkle_root(a) == merkle_root(list(reversed(a)))
    assert merkle_root(a) != merkle_root([_unit("u1"), _unit("u2"), _unit("u3", "CHANGED")])


def test_merkle_empty_sentinel():
    assert merkle_root([]) == hashlib.sha256(b"\x00").hexdigest()


def test_merkle_leaf_matches_documented_construction():
    # Guards the byte-exact leaf form the coordinator also uses.
    u = _unit("u1", "h1", "rcpt-1")
    canonical = json.dumps(
        {"unit_id": "u1", "consensus_result_hash": "h1", "receipt_id": "rcpt-1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected_leaf = hashlib.sha256(b"\x00" + canonical).digest()
    assert merkle_root([u]) == expected_leaf.hex()


# ---- verify ----------------------------------------------------------------


def test_verify_attestation_happy_path():
    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    att = ResultSetAttestation.from_response(_sign_attestation([_unit("u1"), _unit("u2")], key))
    v = verify_attestation(att, authorized_signers=[pub_hex])
    assert v.ok
    assert v.signature_valid and v.signer_authorized
    assert v.root_matches_units and v.signed_root_matches
    assert v.signer_pubkey_hex == pub_hex


def test_verify_rejects_unauthorized_signer():
    key = Ed25519PrivateKey.generate()
    att = ResultSetAttestation.from_response(_sign_attestation([_unit("u1")], key))
    v = verify_attestation(att, authorized_signers=["00" * 32])
    assert v.signature_valid  # the signature itself is valid...
    assert not v.signer_authorized  # ...but the signer isn't on the trusted list
    assert not v.ok


def test_verify_detects_tampered_unit_hash():
    key = Ed25519PrivateKey.generate()
    body = _sign_attestation([_unit("u1", "h1"), _unit("u2", "h2")], key)
    # Tamper a unit hash in the response (but not the signed root).
    body["units"][0]["consensus_result_hash"] = "TAMPERED"
    att = ResultSetAttestation.from_response(body)
    v = verify_attestation(att)
    assert not v.root_matches_units  # recompute over tampered units != attested root
    assert not v.ok


def test_verify_detects_tampered_signature():
    key = Ed25519PrivateKey.generate()
    body = _sign_attestation([_unit("u1")], key)
    raw = bytearray(__import__("base64").b64decode(body["cose_b64"]))
    raw[-1] ^= 0xFF  # flip a signature byte
    body["cose_b64"] = b64encode(bytes(raw)).decode()
    att = ResultSetAttestation.from_response(body)
    v = verify_attestation(att)
    assert not v.signature_valid
    assert not v.ok


def test_verify_against_pulled_results():
    key = Ed25519PrivateKey.generate()
    units = [_unit("u1", "h1", "rcpt-1"), _unit("u2", "h2", "rcpt-2")]
    att = ResultSetAttestation.from_response(_sign_attestation(units, key))
    # Result rows as the delivery route returns them (semantic_hash == consensus hash).
    results = [
        {"unit_id": "u1", "semantic_hash": "h1", "receipt_id": "rcpt-1"},
        {"unit_id": "u2", "semantic_hash": "h2", "receipt_id": "rcpt-2"},
    ]
    assert verify_against_results(att, results) is True
    results[1]["semantic_hash"] = "wrong"
    assert verify_against_results(att, results) is False
