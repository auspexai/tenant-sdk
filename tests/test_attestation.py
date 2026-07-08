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


def _sign_attestation(units: list[dict], key: Ed25519PrivateKey, *, partial: bool = False) -> dict:
    """Produce a `GET /attestation` response body, COSE-signed exactly as the
    coordinator's signing.py does. `partial=True` mirrors the M9 leg-2 checkpoint
    (the predicate carries `partial: true`, omitted otherwise)."""
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
    if partial:
        predicate["partial"] = True
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
        **({"partial": True} if partial else {}),
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
    assert att.partial is False  # default response has no partial key → False


def test_partial_checkpoint_attestation_parses_and_verifies():
    """M9 leg 2: a checkpoint response carries partial=True; the SDK surfaces it
    and verification still passes (the extra signed predicate key is tolerated)."""
    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    body = _sign_attestation([_unit("u1")], key, partial=True)
    assert body["partial"] is True
    att = ResultSetAttestation.from_response(body)
    assert att.partial is True
    v = verify_attestation(att, authorized_signers=[pub_hex])
    assert v.ok  # signature + root checks unaffected by the partial flag
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


# ---- A3: Rekor inclusion check ---------------------------------------------

import httpx  # noqa: E402

from auspexai_tenant.attestation import (  # noqa: E402
    REKOR_PLACEHOLDER_UUID,
    _verify_merkle_inclusion_proof,
    verify_rekor_inclusion,
)

# Independent RFC 6962 Merkle tree + audit-path oracle (NOT the impl under test)
# so the verifier is checked against a separate construction, not a mirror.


def _h_leaf(d: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + d).digest()


def _h_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _mth(data: list[bytes]) -> bytes:
    n = len(data)
    if n == 1:
        return _h_leaf(data[0])
    k = 1
    while k * 2 < n:
        k *= 2
    return _h_node(_mth(data[:k]), _mth(data[k:]))


def _audit_path(m: int, data: list[bytes]) -> list[bytes]:
    n = len(data)
    if n == 1:
        return []
    k = 1
    while k * 2 < n:
        k *= 2
    if m < k:
        return [*_audit_path(m, data[:k]), _mth(data[k:])]
    return [*_audit_path(m - k, data[k:]), _mth(data[:k])]


def test_rfc6962_inclusion_proof_matches_oracle():
    for n in (1, 2, 3, 5, 8, 9):
        data = [f"leaf-{i}".encode() for i in range(n)]
        root = _mth(data)
        for m in range(n):
            proof = _audit_path(m, data)
            assert _verify_merkle_inclusion_proof(
                leaf_hash=_h_leaf(data[m]),
                leaf_index=m,
                tree_size=n,
                proof=proof,
                root_hash=root,
            ), f"valid proof rejected for n={n} m={m}"
        # A tampered root must fail.
        assert not _verify_merkle_inclusion_proof(
            leaf_hash=_h_leaf(data[0]),
            leaf_index=0,
            tree_size=n,
            proof=_audit_path(0, data),
            root_hash=bytes(32),
        )


def _rekor_entry(
    cose_blob: bytes,
    *,
    leaf_index=1,
    n_leaves=4,
    global_index=42,
    commit=True,
    kind="rekord",
):
    """A realistic Rekor entry whose leaf is our COSE artifact, with a valid
    inclusion proof built by the oracle. Default shape = rekord:0.0.1 (what the
    coordinator anchors as — the only Rekor kind accepting pure-Ed25519; its
    canonicalized body stores sha256(artifact) at spec.data.hash.value);
    `kind="intoto"` produces the legacy spec.content shape the verifier still
    tolerates. `commit=False` records a different artifact hash (binding should
    then fail)."""
    env_hash = hashlib.sha256(cose_blob).hexdigest()
    recorded = {"hash": {"algorithm": "sha256", "value": env_hash if commit else "11" * 32}}
    if kind == "rekord":
        body = {"apiVersion": "0.0.1", "kind": "rekord", "spec": {"data": recorded}}
    else:
        body = {"apiVersion": "0.0.2", "kind": "intoto", "spec": {"content": recorded}}
    body_bytes = json.dumps(body).encode()
    data = [f"other-{i}".encode() for i in range(n_leaves)]
    data[leaf_index] = (
        body_bytes  # leaf = sha256(0x00 || body_bytes) — what the verifier recomputes
    )
    proof = _audit_path(leaf_index, data)
    return {
        "body": b64encode(body_bytes).decode(),
        "logIndex": global_index,
        "verification": {
            "inclusionProof": {
                "logIndex": leaf_index,
                "treeSize": n_leaves,
                "hashes": [h.hex() for h in proof],
                "rootHash": _mth(data).hex(),
            }
        },
    }


def _mock_rekor(uuid: str | None, entry: dict | None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if uuid is not None and request.url.path.endswith(f"/log/entries/{uuid}"):
            return httpx.Response(200, json={uuid: entry})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_verify_rekor_inclusion_ok():
    cose = b"\xde\xad\xbe\xef cose artifact"
    entry = _rekor_entry(cose, global_index=42)
    ri = verify_rekor_inclusion(
        cose, log_index=42, entry_uuid="uuid-1", client=_mock_rekor("uuid-1", entry)
    )
    assert ri.included is True
    assert ri.entry_found and ri.log_index_matches and ri.artifact_committed
    assert ri.inclusion_proof_verified and ri.tree_size == 4


def test_verify_rekor_inclusion_legacy_intoto_shape():
    """The binding fallback still reads the legacy intoto spec.content shape."""
    cose = b"\xde\xad\xbe\xef cose artifact"
    entry = _rekor_entry(cose, global_index=42, kind="intoto")
    ri = verify_rekor_inclusion(
        cose, log_index=42, entry_uuid="uuid-1", client=_mock_rekor("uuid-1", entry)
    )
    assert ri.included is True and ri.artifact_committed


def test_verify_rekor_inclusion_not_anchored():
    ri = verify_rekor_inclusion(b"x", log_index=0, entry_uuid=REKOR_PLACEHOLDER_UUID)
    assert ri.checked is False and ri.included is False


def test_verify_rekor_inclusion_entry_not_found():
    ri = verify_rekor_inclusion(
        b"x", log_index=1, entry_uuid="missing", client=_mock_rekor(None, None)
    )
    assert ri.checked is True and ri.entry_found is False and ri.included is False


def test_verify_rekor_inclusion_wrong_artifact():
    cose = b"the real artifact"
    entry = _rekor_entry(cose, commit=False)  # body commits to a different hash
    ri = verify_rekor_inclusion(
        cose, log_index=42, entry_uuid="uuid-1", client=_mock_rekor("uuid-1", entry)
    )
    assert ri.artifact_committed is False and ri.included is False


def test_verify_rekor_inclusion_tampered_proof():
    cose = b"artifact"
    entry = _rekor_entry(cose)
    entry["verification"]["inclusionProof"]["rootHash"] = "00" * 32  # wrong root
    ri = verify_rekor_inclusion(
        cose, log_index=42, entry_uuid="uuid-1", client=_mock_rekor("uuid-1", entry)
    )
    assert ri.inclusion_proof_verified is False and ri.included is False


def test_verify_attestation_check_rekor_folds_into_ok():
    key = Ed25519PrivateKey.generate()
    body = _sign_attestation([_unit("u1", "h1", "rcpt-1")], key)
    body["rekor_log_index"] = 42
    body["rekor_entry_uuid"] = "uuid-1"
    att = ResultSetAttestation.from_response(body)
    cose = __import__("base64").b64decode(att.cose_b64)

    # Anchored + included → ok True (offline checks already pass for this build).
    good = _mock_rekor("uuid-1", _rekor_entry(cose, global_index=42))
    v = verify_attestation(att, check_rekor=True, rekor_client=good)
    assert v.rekor_inclusion is not None and v.rekor_inclusion.included and v.ok is True

    # Entry missing → inclusion fails → ok False even though offline checks pass.
    v2 = verify_attestation(att, check_rekor=True, rekor_client=_mock_rekor(None, None))
    assert v2.rekor_inclusion.included is False and v2.ok is False

    # check_rekor=False → offline-only, ok True, no rekor result.
    v3 = verify_attestation(att)
    assert v3.rekor_inclusion is None and v3.ok is True


def test_partial_flag_bound_to_signed_predicate():
    """AUD-35: a checkpoint (partial) attestation re-served with body.partial=false
    must NOT verify — `partial` is read from the COSE-signed predicate, not the
    unsigned response body, so a partial set can't masquerade as final."""
    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    body = _sign_attestation([_unit("u1")], key, partial=True)  # signed predicate: partial=True
    # Tamper the UNSIGNED body: drop the partial flag so it reads as a complete set.
    body.pop("partial", None)
    att = ResultSetAttestation.from_response(body)
    assert att.partial is False  # the body now claims the set is complete
    v = verify_attestation(att, authorized_signers=[pub_hex])
    assert not v.ok  # ...but the signed predicate says partial=True → mismatch
    assert not v.signed_root_matches
