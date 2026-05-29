"""RFC 9421 HTTP Message Signatures — Ed25519 request signer (tenant side).

The coordinator authenticates researcher requests with a deliberately narrow
subset of RFC 9421 (`auspexai_platform.auth.signature`). This module is the
tenant-side signer that produces byte-identical signatures the coordinator
verifies. It is a standalone **replica** of the coordinator's canonicalization,
not an import: the SDK is Apache-2.0 and must not depend on the AGPL-3.0
platform (see Documentation/AuspexAI/v0.1.0/sdk_license_boundary_position.md).
The shared contract is the wire format, not shared code.

Accepted subset (must match the coordinator exactly):

  - Algorithm: ed25519 only.
  - Covered components: @method, @path, @authority, and content-digest (the
    last only when the body is non-empty). @signature-params is appended last
    per RFC 9421 §2.5.
  - `created` (Unix seconds) is REQUIRED; the coordinator rejects signatures
    outside a +/- 5-minute window.
  - `keyid` is the lowercase hex of the 32-byte Ed25519 public key.
  - One signature per request (label `sig1`); no `nonce`.
  - content-digest is SHA-256 of the body per RFC 9530, format `sha-256=:b64:`.

`Rfc9421Auth` is an `httpx.Auth` that signs outgoing requests automatically;
`sign_request` is the lower-level primitive it (and tests) call.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Generator
from datetime import UTC, datetime

import httpx

from auspexai_tenant.signing import MaintainerKey

SUPPORTED_ALG = "ed25519"
SIGNATURE_LABEL = "sig1"
DEFAULT_COVERED: tuple[str, ...] = ("@method", "@path", "@authority")


def compute_content_digest(body: bytes) -> str:
    """Build a Content-Digest header value for `body` (RFC 9530 `sha-256`)."""
    digest = hashlib.sha256(body).digest()
    return f"sha-256=:{base64.b64encode(digest).decode('ascii')}:"


def build_signature_base(
    *,
    covered: tuple[str, ...],
    raw_covered_and_params: str,
    method: str,
    path: str,
    authority: str,
    content_digest_header: str | None,
) -> bytes:
    """Reconstruct the RFC 9421 §2.5 signature base. Byte-identical to the
    coordinator's `build_signature_base`."""
    lines: list[str] = []
    for component in covered:
        if component == "@method":
            lines.append(f'"@method": {method.upper()}')
        elif component == "@path":
            lines.append(f'"@path": {path}')
        elif component == "@authority":
            lines.append(f'"@authority": {authority.lower()}')
        elif component == "content-digest":
            if content_digest_header is None:
                raise ValueError(
                    "signature covers `content-digest` but no Content-Digest "
                    "header value was provided"
                )
            lines.append(f'"content-digest": {content_digest_header.strip()}')
        else:
            raise ValueError(
                f"unsupported covered component {component!r} (supported: "
                "@method, @path, @authority, content-digest)"
            )
    lines.append(f'"@signature-params": {raw_covered_and_params}')
    return "\n".join(lines).encode("utf-8")


def sign_request(
    *,
    key: MaintainerKey,
    method: str,
    path: str,
    authority: str,
    body: bytes,
    created: int | None = None,
    covered: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Produce the Signature-Input + Signature (+ Content-Digest when the body
    is non-empty) headers for an Ed25519-signed request.

    Returns a dict of headers to merge onto the outgoing request. `key` is the
    tenant maintainer keypair; its public key becomes the `keyid` the
    coordinator resolves to the tenant.
    """
    if created is None:
        created = int(datetime.now(UTC).timestamp())
    if covered is None:
        covered = DEFAULT_COVERED + (("content-digest",) if body else ())

    pubkey_hex = key.pubkey_hex.lower()
    covered_list = " ".join(f'"{c}"' for c in covered)
    raw_covered_and_params = (
        f'({covered_list});created={created};alg="{SUPPORTED_ALG}";keyid="{pubkey_hex}"'
    )

    content_digest_header: str | None = None
    if "content-digest" in covered:
        content_digest_header = compute_content_digest(body)

    base = build_signature_base(
        covered=covered,
        raw_covered_and_params=raw_covered_and_params,
        method=method,
        path=path,
        authority=authority,
        content_digest_header=content_digest_header,
    )
    sig = key.sign(base)

    headers = {
        "Signature-Input": f"{SIGNATURE_LABEL}={raw_covered_and_params}",
        "Signature": f"{SIGNATURE_LABEL}=:{base64.b64encode(sig).decode('ascii')}:",
    }
    if content_digest_header:
        headers["Content-Digest"] = content_digest_header
    return headers


def _request_authority(request: httpx.Request) -> str:
    """The @authority for an httpx request: the Host header httpx will send
    (host, plus port only when non-default), matching what the coordinator
    reconstructs server-side."""
    host_header = request.headers.get("host")
    if host_header:
        return host_header
    url = request.url
    return url.host if url.port is None else f"{url.host}:{url.port}"


class Rfc9421Auth(httpx.Auth):
    """An `httpx.Auth` that signs every outgoing request with a tenant
    maintainer key, so the coordinator authenticates it as that tenant's
    researcher credential.

    Usage:
        key = MaintainerKey.load(path)
        with httpx.Client(auth=Rfc9421Auth(key)) as client:
            client.post("https://coord.auspexai.network/api/v0/experiments", json=body)
    """

    # We need the (already-encoded) request body to compute content-digest.
    requires_request_body = True

    def __init__(self, key: MaintainerKey, *, covered: tuple[str, ...] | None = None) -> None:
        self._key = key
        self._covered = covered

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        body = request.content or b""
        headers = sign_request(
            key=self._key,
            method=request.method,
            path=request.url.path,  # path only; the coordinator does not cover @query
            authority=_request_authority(request),
            body=body,
            covered=self._covered,
        )
        request.headers.update(headers)
        yield request
