"""Signed read client for the AuspexAI coordinator (M-Results retrieval).

The SDK's first *read* surface. A researcher fetches their experiments, results,
receipts, and the offload bundle over RFC 9421-signed GETs — authenticated with
the same tenant key that signs experiment submission. Mirrors
`upload.py`'s httpx + `Rfc9421Auth` pattern; pass `client` (an `httpx.Client`
backed by `httpx.MockTransport`) to exercise it offline in tests.

Retrieval maps to the coordinator routes:
  - list_experiments / get_experiment  → GET /api/v0/experiments[/{id}]
  - get_results / iter_results          → GET .../{id}/results[?include=raw]
  - get_receipts                        → GET .../{id}/receipts
  - export                              → GET .../{id}/results/export  (the offload
                                          bundle; collection transfers data custody
                                          to the researcher — verify_transfer checks
                                          the coordinator's signed custody record)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auspexai_tenant.attestation import ResultSetAttestation
from auspexai_tenant.http_signing import Rfc9421Auth
from auspexai_tenant.manifest import compute_package_digest
from auspexai_tenant.package import build_package_archive
from auspexai_tenant.signing import TenantKey

DEFAULT_TIMEOUT_S = 30.0


class CoordinatorError(Exception):
    """A non-2xx response from the coordinator."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"coordinator returned HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class TransferVerification:
    """Result of checking an export bundle's signed custody record."""

    valid: bool
    transfer_id: str
    result_set_root: str
    collected_at: str
    coordinator_pubkey_hex: str


class TenantClient:
    """Signed-GET client over the coordinator's tenant-scoped read routes.

    Every call is RFC 9421-signed with `key`; the coordinator resolves the
    signature to this tenant's researcher credential and scopes the response.
    """

    def __init__(
        self,
        coordinator_url: str,
        key: TenantKey,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = coordinator_url.rstrip("/")
        self._auth = Rfc9421Auth(key)
        self._timeout = timeout
        self._client = client

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        # The signature covers @path only (not @query), so query params ride
        # unsigned — safe per the coordinator's RFC 9421 verification.
        if self._client is None:
            with httpx.Client(timeout=self._timeout) as c:
                resp = c.get(url, auth=self._auth, params=params)
        else:
            resp = self._client.get(url, auth=self._auth, params=params)
        if not resp.is_success:
            raise CoordinatorError(resp.status_code, resp.text)
        return resp.json()

    # ---- experiments ----

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._get("/api/v0/experiments").get("experiments") or []

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._get(f"/api/v0/experiments/{experiment_id}")

    def get_activity(self, experiment_id: str) -> dict[str, Any]:
        """The experiment's liveness rollup (anonymized contributor/unit counts,
        replication fill, network size, and the caller's own-account workers).
        The cheap snapshot the autonomic driver's `progress()` polls."""
        return self._get(f"/api/v0/experiments/{experiment_id}/activity")

    # ---- results ----

    def get_results(
        self, experiment_id: str, *, include: str = "consensus", cursor: str | None = None
    ) -> dict[str, Any]:
        """One page of results. `include` is 'consensus' (default, one per unit)
        or 'raw' (all replicas). Returns {results, next_cursor}."""
        params: dict[str, Any] = {"include": include}
        if cursor:
            params["cursor"] = cursor
        return self._get(f"/api/v0/experiments/{experiment_id}/results", params)

    def iter_results(
        self, experiment_id: str, *, include: str = "consensus"
    ) -> Iterator[dict[str, Any]]:
        """All results, transparently following `next_cursor`."""
        cursor: str | None = None
        while True:
            page = self.get_results(experiment_id, include=include, cursor=cursor)
            yield from page.get("results") or []
            cursor = page.get("next_cursor")
            if not cursor:
                return

    # ---- receipts ----

    def get_receipts(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._get(f"/api/v0/experiments/{experiment_id}/receipts").get("receipts") or []

    # ---- completion attestation (#34 §6.3) ----

    def get_attestation(
        self, experiment_id: str, *, checkpoint: bool = False
    ) -> ResultSetAttestation:
        """Fetch the result-set attestation. By default available once the
        experiment is COMPLETED (the final set). **M9 leg 2** — pass
        `checkpoint=True` for a partial consensus-so-far attestation over a
        not-yet-complete experiment (`.partial is True`), the integrity anchor for
        a partial collection when a pause/capacity-collapse stalls the run. Verify
        with `auspexai_tenant.attestation.verify_attestation`."""
        params = {"checkpoint": "true"} if checkpoint else None
        body = self._get(f"/api/v0/experiments/{experiment_id}/attestation", params=params)
        return ResultSetAttestation.from_response(body)

    # ---- offload bundle ----

    def export(self, experiment_id: str) -> dict[str, Any]:
        """Fetch the offload bundle (consensus payloads + receipts + manifest +
        signed custody record). Collecting it transfers data custody to the
        researcher and arms collection-anchored age-off coordinator-side."""
        return self._get(f"/api/v0/experiments/{experiment_id}/results/export")

    # ---- executor packages (coordinator-served provisioning, §9 #40a) ----

    def upload_package(self, package_dir: str | Path) -> dict[str, Any]:
        """Upload the executor package staged in `package_dir` to the
        coordinator (`POST /api/v0/packages`).

        Builds the deterministic tar.gz (`build_package_archive`) and sends it
        as a raw `application/gzip` body, RFC 9421-signed — the binary body is
        bound into the signature via `Content-Digest` (SHA-256 of the raw
        bytes), exactly like the JSON paths. `X-Package-Digest` carries the
        package *tree* digest (`compute_package_digest`, the same value as the
        manifest's `executor.package_sha256` pin); the coordinator re-derives
        it over the extracted tree and refuses on mismatch (422). Returns the
        coordinator's envelope — `{"package_digest": ..., "status": "stored"}`
        on 201, `status: "already_exists"` on 200; 422 (digest mismatch) and
        413 (too large) raise `CoordinatorError`."""
        archive = build_package_archive(package_dir)
        digest = compute_package_digest(package_dir)
        url = f"{self._base}/api/v0/packages"
        headers = {
            "Content-Type": "application/gzip",
            "X-Package-Digest": digest,
        }
        if self._client is None:
            with httpx.Client(timeout=self._timeout) as c:
                resp = c.post(url, auth=self._auth, content=archive, headers=headers)
        else:
            resp = self._client.post(url, auth=self._auth, content=archive, headers=headers)
        if not resp.is_success:
            raise CoordinatorError(resp.status_code, resp.text)
        return resp.json()

    # ---- demand-board (model requests, M2 / §9 #32/#39) ----

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self._base}{path}"
        if self._client is None:
            with httpx.Client(timeout=self._timeout) as c:
                resp = c.post(url, auth=self._auth, json=body)
        else:
            resp = self._client.post(url, auth=self._auth, json=body)
        if not resp.is_success:
            raise CoordinatorError(resp.status_code, resp.text)
        return resp.json()

    def get_catalog(self) -> dict[str, Any]:
        """The network's bottom-up model catalog — the emergent set of models
        active workers declare: {models:[{model_id, worker_count}], total_active_workers}."""
        return self._get("/api/v0/models/catalog")

    def request_model(
        self, model_id: str, *, reason: str, hf_repo: str | None = None
    ) -> dict[str, Any]:
        """Signal demand for a model (BYOM, §5.8). `model_id` is the worker store
        id (`<repo-slug>-<quant>`). Recorded as a demand signal; if no active
        worker holds it the request enters the maintainer review queue
        (status `pending`), otherwise it's `available`."""
        body: dict[str, Any] = {"model_id": model_id, "reason": reason}
        if hf_repo:
            body["hf_repo"] = hf_repo
        return self._post("/api/v0/model-requests", body)

    def list_my_model_requests(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Your tenant's model requests (the coordinator scopes the list to the
        signing credential's tenant)."""
        params = {"status": status} if status else None
        return self._get("/api/v0/model-requests", params).get("requests") or []

    # ---- software requests (code-plane requirements, §9 #46) ----

    def request_software(self, title: str, *, description: str, reason: str) -> dict[str, Any]:
        """Signal demand for a worker-baseline CAPABILITY the network doesn't
        provide (e.g. an inference backend, a new runtime) — the code-plane
        analog of `request_model`. Always enters the maintainer review queue
        (status `pending`); the maintainer attaches a dependencies/security/
        alternatives assessment before approving/declining, and a recorded
        release later fulfils it (status `released`, with `release_version`)."""
        return self._post(
            "/api/v0/software-requests",
            {"title": title, "description": description, "reason": reason},
        )

    def list_my_software_requests(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Your tenant's software requests, incl. assessment + resolution +
        fulfilling release_version (tenant-scoped by the coordinator)."""
        params = {"status": status} if status else None
        return self._get("/api/v0/software-requests", params).get("requests") or []

    # ---- tenant applications (apply-from-CLI onboarding, Option D) ----------

    def apply_for_tenant(
        self,
        *,
        github_access_token: str,
        requested_tenant_id: str,
        contact_name: str,
        affiliation: str,
        research_summary: str | None = None,
        research_classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit a tenant application signed by THIS client's key.

        The signing key need not be registered yet: the coordinator verifies
        the RFC 9421 signature against the request's own keyid, so the
        application carries built-in proof-of-possession of the (possibly
        fresh) tenant key and no public key is ever hand-passed out of band.
        The GitHub access token (identity scope only) lets the coordinator
        resolve who applied; the SDK never persists it.

        At least one of `research_classes` (taxonomy ids, e.g.
        "behavioral_drift") or `research_summary` (free text) is required;
        empty/None values are omitted from the wire body. Returns
        {application_id, status} (HTTP 201)."""
        if not research_classes and not research_summary:
            raise ValueError("at least one of research_classes or research_summary is required")
        body: dict[str, Any] = {
            "github_access_token": github_access_token,
            "requested_tenant_id": requested_tenant_id,
            "contact_name": contact_name,
            "affiliation": affiliation,
        }
        if research_summary:
            body["research_summary"] = research_summary
        if research_classes:
            body["research_classes"] = list(research_classes)
        return self._post("/api/v0/tenant-applications", body)

    def my_tenant_applications(self) -> list[dict[str, Any]]:
        """The applications submitted by this signing key (the coordinator
        scopes the list to the request's keyid). Each item carries
        {application_id, status, resolution_reason, created_tenant_id}."""
        return self._get("/api/v0/tenant-applications/mine").get("applications") or []


def verify_transfer(bundle: dict[str, Any]) -> TransferVerification:
    """Verify the coordinator's Ed25519 signature on an export bundle's transfer
    (custody) record — proof Auspex delivered exactly these results to you.

    The signed message is `result_set_root|collected_by|collected_at|manifest_hash`
    (matching the coordinator's `api/results.export_results`)."""
    t = bundle["transfer"]
    record = (
        f"{t['result_set_root']}|{t['collected_by_pubkey']}|"
        f"{t['collected_at']}|{t['manifest_hash']}"
    ).encode()
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(t["coordinator_pubkey_hex"]))
    valid = True
    try:
        pub.verify(bytes.fromhex(t["coordinator_signature"]), record)
    except InvalidSignature:
        valid = False
    return TransferVerification(
        valid=valid,
        transfer_id=t["transfer_id"],
        result_set_root=t["result_set_root"],
        collected_at=t["collected_at"],
        coordinator_pubkey_hex=t["coordinator_pubkey_hex"],
    )
