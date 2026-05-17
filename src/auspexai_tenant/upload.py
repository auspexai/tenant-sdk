"""Manifest upload — POST a signed manifest to the AuspexAI coordinator.

v0.1: no auth (the coordinator doesn't exist yet; this is the SDK side of a
future API surface). Multipart POST to `{coordinator}/api/v0/manifests` with
`manifest` and (optional) `signature` form parts. Returns an UploadResult
with status code, response body, and success flag.

The coordinator-side endpoint shape is defined by this client (the SDK is
the reference implementation of the tenant side of the protocol); when the
coordinator daemon is built, it will implement the matching server side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class UploadResult:
    """Outcome of an upload_manifest call."""

    status_code: int
    body: str
    ok: bool


def upload_manifest(
    manifest_path: Path,
    coordinator_url: str,
    signature_path: Path | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> UploadResult:
    """POST a manifest (and optional detached signature) to the coordinator.

    Returns an UploadResult. Raises httpx.RequestError on network failure
    (DNS, TLS, connect, read timeout, etc.); HTTP-level non-2xx responses
    are reported via UploadResult.ok=False rather than raising.

    The `client` parameter is for testability — pass a pre-configured
    `httpx.Client` (e.g., backed by `httpx.MockTransport`) to test without
    real networking.
    """
    endpoint = f"{coordinator_url.rstrip('/')}/api/v0/manifests"
    manifest_bytes = manifest_path.read_bytes()
    files: dict[str, tuple[str, bytes, str]] = {
        "manifest": (manifest_path.name, manifest_bytes, "application/json"),
    }
    if signature_path is not None:
        sig_bytes = signature_path.read_bytes()
        files["signature"] = (signature_path.name, sig_bytes, "application/json")

    if client is None:
        with httpx.Client(timeout=timeout) as c:
            response = c.post(endpoint, files=files)
    else:
        response = client.post(endpoint, files=files)

    return UploadResult(
        status_code=response.status_code,
        body=response.text,
        ok=response.is_success,
    )
