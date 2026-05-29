"""Experiment submission — POST a signed manifest to the AuspexAI coordinator.

A researcher submits an experiment by POSTing its manifest (and detached
ManifestSignature) as JSON to `POST /api/v0/experiments`, authenticated with
an RFC 9421 HTTP Message Signature over the request (the researcher's tenant
maintainer key — see `http_signing`). The coordinator resolves the signing
key to a tenant and enforces `manifest.tenant_id == credential.tenant_id`.

This replaces the earlier multipart `/api/v0/manifests` shape, which targeted
an endpoint the coordinator never implemented and carried no request-level
auth (researcher_dashboard_design.md §7.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from auspexai_tenant.http_signing import Rfc9421Auth
from auspexai_tenant.signing import MaintainerKey

DEFAULT_TIMEOUT_S = 30.0
EXPERIMENTS_PATH = "/api/v0/experiments"


@dataclass(frozen=True)
class UploadResult:
    """Outcome of a submit_experiment call."""

    status_code: int
    body: str
    ok: bool


def submit_experiment(
    manifest: dict[str, Any],
    signature: dict[str, Any],
    coordinator_url: str,
    key: MaintainerKey,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> UploadResult:
    """POST a manifest + signature to `POST /api/v0/experiments`, signed with
    `key` (RFC 9421).

    Returns an UploadResult. Raises httpx.RequestError on network failure;
    HTTP-level non-2xx responses are reported via UploadResult.ok=False.

    Pass a pre-configured `client` (e.g., backed by httpx.MockTransport) for
    testing. The request is signed regardless of whether a custom client is
    supplied, because the auth is attached per-request.
    """
    endpoint = f"{coordinator_url.rstrip('/')}{EXPERIMENTS_PATH}"
    auth = Rfc9421Auth(key)
    payload = {"manifest": manifest, "signature": signature}

    if client is None:
        with httpx.Client(timeout=timeout) as c:
            response = c.post(endpoint, json=payload, auth=auth)
    else:
        response = client.post(endpoint, json=payload, auth=auth)

    return UploadResult(
        status_code=response.status_code,
        body=response.text,
        ok=response.is_success,
    )


def submit_experiment_from_files(
    manifest_path: Path,
    signature_path: Path,
    coordinator_url: str,
    key: MaintainerKey,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.Client | None = None,
) -> UploadResult:
    """Convenience wrapper: load manifest + signature JSON from disk, then
    `submit_experiment`."""
    manifest = json.loads(manifest_path.read_text())
    signature = json.loads(signature_path.read_text())
    return submit_experiment(
        manifest, signature, coordinator_url, key, timeout=timeout, client=client
    )
