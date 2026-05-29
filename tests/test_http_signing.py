"""Tests for the RFC 9421 Ed25519 request signer.

Correctness here means *byte-compatibility with the coordinator's verifier*.
The SDK can't import the AGPL platform, so these tests re-derive the signature
base independently and verify the Ed25519 signature over it, and assert the
exact RFC 9421 wire format the coordinator parses. (A live cross-codebase
acceptance check against the real verifier lives in the signing harness.)
"""

from __future__ import annotations

import base64
import hashlib

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auspexai_tenant.http_signing import (
    Rfc9421Auth,
    build_signature_base,
    compute_content_digest,
    sign_request,
)
from auspexai_tenant.signing import MaintainerKey


@pytest.fixture
def key() -> MaintainerKey:
    return MaintainerKey.generate()


def _pubkey(key: MaintainerKey) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(key.pubkey_hex))


def _parse_input(value: str) -> tuple[str, dict[str, str]]:
    """Minimal split of `sig1=(...);k=v;...` into (raw_covered_and_params, params)."""
    assert value.startswith("sig1=")
    rest = value[len("sig1=") :]
    covered_and_params = rest
    params: dict[str, str] = {}
    for chunk in rest.split(";")[1:]:
        k, _, v = chunk.partition("=")
        params[k.strip()] = v.strip().strip('"')
    return covered_and_params, params


def test_content_digest_format(key: MaintainerKey) -> None:
    body = b'{"a":1}'
    expected = base64.b64encode(hashlib.sha256(body).digest()).decode()
    assert compute_content_digest(body) == f"sha-256=:{expected}:"


def test_sign_request_headers_and_format(key: MaintainerKey) -> None:
    body = b'{"manifest":{}}'
    headers = sign_request(
        key=key,
        method="post",  # lowercased on input; base must uppercase it
        path="/api/v0/experiments",
        authority="Coord.AuspexAI.Network",  # base must lowercase it
        body=body,
        created=1716123456,
    )
    assert set(headers) == {"Signature-Input", "Signature", "Content-Digest"}

    raw, params = _parse_input(headers["Signature-Input"])
    assert params["alg"] == "ed25519"
    assert params["created"] == "1716123456"
    assert params["keyid"] == key.pubkey_hex.lower()
    # Body non-empty → content-digest is covered.
    assert '"content-digest"' in raw
    assert headers["Content-Digest"] == compute_content_digest(body)


def test_signature_verifies_over_canonical_base(key: MaintainerKey) -> None:
    body = b'{"x":1}'
    headers = sign_request(
        key=key,
        method="POST",
        path="/api/v0/experiments",
        authority="coord.auspexai.network",
        body=body,
        created=1716123456,
    )
    raw, _ = _parse_input(headers["Signature-Input"])
    base = build_signature_base(
        covered=("@method", "@path", "@authority", "content-digest"),
        raw_covered_and_params=raw,
        method="POST",
        path="/api/v0/experiments",
        authority="coord.auspexai.network",
        content_digest_header=headers["Content-Digest"],
    )
    # Expected exact base lines (this is what the coordinator reconstructs).
    expected = (
        '"@method": POST\n'
        '"@path": /api/v0/experiments\n'
        '"@authority": coord.auspexai.network\n'
        f'"content-digest": {headers["Content-Digest"]}\n'
        f'"@signature-params": {raw}'
    ).encode()
    assert base == expected

    sig = base64.b64decode(headers["Signature"][len("sig1=:") : -1])
    _pubkey(key).verify(sig, base)  # raises on failure


def test_empty_body_omits_content_digest(key: MaintainerKey) -> None:
    headers = sign_request(
        key=key, method="GET", path="/api/v0/experiments", authority="h", body=b""
    )
    assert "Content-Digest" not in headers
    raw, _ = _parse_input(headers["Signature-Input"])
    assert "content-digest" not in raw


def test_rfc9421_auth_signs_outgoing_request(key: MaintainerKey) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(201, text="{}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, auth=Rfc9421Auth(key)) as client:
        client.post("https://coord.auspexai.network/api/v0/experiments", json={"manifest": {}})

    assert "signature-input" in seen
    assert "signature" in seen
    assert "content-digest" in seen
    assert key.pubkey_hex.lower() in seen["signature-input"]
