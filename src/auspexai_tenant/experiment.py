"""Stateful per-experiment client — the autonomic driver's coordinator handle (M8 §3.1).

The SDK was publish + read only (`upload.submit_experiment`, `TenantClient`).
The autonomic control loop (`run_until`, design note §3.2) needs to *drive* one
experiment: submit adaptive work-unit batches, read agreed results to fold, stop
(finalize) or abort, and fetch the result-set attestation. `Experiment` is that
handle — it composes a `TenantClient` for the read surface and adds the signed
submit + lifecycle-action POSTs the SDK lacked, over the same httpx +
`Rfc9421Auth` pattern as `upload.py`. Pass `client` (an `httpx.Client` backed by
`httpx.MockTransport`) to drive it offline in tests.

**Idempotency contract (load-bearing for crash-resume, design note §3.4 / §10.1):**
the coordinator *rejects* a duplicate `unit_id` with `409 unit_id_already_submitted`
— it does **not** silently dedupe, and the batch is all-or-nothing. So a driver
that, after a crash, re-submits a round's *pinned* `unit_id`s must treat
`UnitsAlreadySubmitted` as **success** (the work is already on the coordinator).
`submit_units` raises typed errors so the driver can distinguish that recoverable
case from a terminal `MaxUnitsExceeded` / `SubmissionsFinalized`.

`experiment_id` is the **coordinator** id (e.g. ``"exp-CmK1YYqC"``). Work-unit
envelopes instead carry the tenant's experiment **label** (`tenant_experiment_label`)
and the manifest hash; this client resolves both from the experiment metadata
(immutable — cached on first use) so the caller supplies only `unit_id` + payload.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from auspexai_tenant.attestation import ResultSetAttestation
from auspexai_tenant.client import CoordinatorError, TenantClient
from auspexai_tenant.http_signing import Rfc9421Auth
from auspexai_tenant.retry import call_with_retry
from auspexai_tenant.signing import TenantKey

DEFAULT_TIMEOUT_S = 30.0

# Terminal experiment statuses — no further lifecycle change is possible.
TERMINAL_STATUSES = frozenset({"completed", "aborted", "archived"})
# The status at which the final result-set attestation is available and the
# experiment-level reduce is final (design note §6.4).
REDUCE_READY_STATUS = "completed"
# Reserved opaque-payload key carrying round provenance until the M6 `round`
# column lands (design note §4). The coordinator persists only unit_id + payload,
# so the payload is the one tenant-controlled field round can survive in;
# consensus-safe (every replica of a unit shares its payload). `submit_units`'
# signature is stable across the eventual promotion to a dedicated column.
ROUND_PAYLOAD_KEY = "_auspexai_round"


class MaxUnitsExceededError(CoordinatorError):
    """The batch would exceed the experiment's `max_units` hard cap (HTTP 409
    `max_units_exceeded`). Terminal — a runaway / over-eager generator."""


class UnitsAlreadySubmittedError(CoordinatorError):
    """One or more `unit_id`s were already submitted (HTTP 409
    `unit_id_already_submitted`). The crash-resume idempotency signal: re-submitting
    a round's pinned `unit_id`s raises this, and the driver treats it as success —
    the work is already on the coordinator."""


class SubmissionsFinalizedError(CoordinatorError):
    """Submissions are finalized; no new units accepted (HTTP 409
    `submissions_finalized`)."""


class LifecycleConflictError(CoordinatorError):
    """A lifecycle action was not applicable in the experiment's current state
    (HTTP 409 — e.g. `finalize_not_applicable` or a disallowed transition)."""


@dataclass(frozen=True)
class Unit:
    """A work unit to submit: a tenant-chosen `unit_id` + opaque `payload`. The
    `Experiment` wraps it into the coordinator envelope, filling in `tenant_id`,
    the experiment label, and the manifest hash."""

    unit_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of a `submit_units` call (the coordinator's accepted set)."""

    submitted_unit_ids: list[str]
    count: int


@dataclass(frozen=True)
class ResultPage:
    """One page of consensus results plus the cursor to resume after it. The
    driver journals `next_cursor` as its exactly-once watermark (design §3.4)."""

    results: list[dict[str, Any]]
    next_cursor: str | None


@dataclass(frozen=True)
class Progress:
    """Cheap experiment snapshot for hot-loop polling (design note §3.1).

    `reduce_ready` is derived client-side from the experiment status (no
    coordinator change in M8 — the activity snapshot's `reduce_ready` flag is M6)."""

    status: str | None
    reduce_ready: bool
    submissions_finalized: bool
    total_work_units: int | None
    work_unit_counts: dict[str, int]
    completions_total: int | None
    replication_target_total: int | None
    active_contributor_count: int | None
    network_active_workers: int | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _error_code(response: httpx.Response) -> str | None:
    """Best-effort extraction of the coordinator's machine error code
    (`{"error": {"code": ...}}`); None when the body isn't that shape."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    # The coordinator's HTTPException envelope nests under "detail"
    # ({"detail": {"error": {...}}}); accept the bare shape too. This mismatch
    # made every typed 409 (unit_id_already_submitted / max_units_exceeded /
    # submissions_finalized) fall through to generic CoordinatorError — the
    # driver's resume idempotency was dead code until the 2026-07-04 campaign
    # exposed it (the wq2-prefix workaround retires with this fix).
    detail = body.get("detail")
    if isinstance(detail, dict) and "error" in detail:
        body = detail
    error = body.get("error")
    return error.get("code") if isinstance(error, dict) else None


class Experiment:
    """Signed client scoped to one coordinator experiment (design note §3.1)."""

    def __init__(
        self,
        coordinator_url: str,
        key: TenantKey,
        experiment_id: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = coordinator_url.rstrip("/")
        self._key = key
        self._auth = Rfc9421Auth(key)
        self._timeout = timeout
        self._client = client
        self.experiment_id = experiment_id
        # The read surface (list/get/results/receipts/export/attestation/activity).
        self.reader = TenantClient(coordinator_url, key, timeout=timeout, client=client)
        self._meta: dict[str, Any] | None = None

    # ---- experiment metadata (immutable trio, cached) ----------------------

    def _metadata(self) -> dict[str, Any]:
        """Tenant id, experiment label, and manifest hash — immutable, resolved
        once and cached (used to build valid work-unit envelopes)."""
        if self._meta is None:
            self._meta = self.reader.get_experiment(self.experiment_id)
        return self._meta

    # ---- signed POST -------------------------------------------------------

    def _post(self, path: str, *, json: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self._base}{path}"

        def _do() -> httpx.Response:
            if self._client is None:
                with httpx.Client(timeout=self._timeout) as c:
                    return c.post(url, auth=self._auth, json=json)
            return self._client.post(url, auth=self._auth, json=json)

        # D18: retry transient tunnel failures. Safe: the coordinator's submit
        # idempotency turns a re-sent-but-already-landed POST into a 409 the caller
        # treats as success.
        return call_with_retry(_do)

    # ---- submit ------------------------------------------------------------

    def submit_units(self, units: Iterable[Unit], *, round: int | None = None) -> SubmitResult:
        """Submit a batch of work units (idempotent on `unit_id` — see the module
        docstring). `round` tags this batch's provenance (design note §4); it is
        carried in each unit's opaque payload under `ROUND_PAYLOAD_KEY` until the
        M6 `round` column supersedes it. Raises `MaxUnitsExceededError` /
        `UnitsAlreadySubmittedError` / `SubmissionsFinalizedError` on the matching 409s."""
        meta = self._metadata()
        envelopes: list[dict[str, Any]] = []
        for unit in units:
            payload = unit.payload
            if round is not None:
                payload = {**unit.payload, ROUND_PAYLOAD_KEY: round}
            envelopes.append(
                {
                    "schema_version": "0.1",
                    "unit_id": unit.unit_id,
                    "tenant_id": meta["tenant_id"],
                    "experiment_id": meta["tenant_experiment_label"],
                    "manifest_sha256": meta["manifest_hash"],
                    "payload": payload,
                }
            )
        if not envelopes:
            raise ValueError("submit_units requires at least one unit")

        resp = self._post(
            f"/api/v0/experiments/{self.experiment_id}/work-units",
            json={"work_units": envelopes},
        )
        if resp.status_code == httpx.codes.CONFLICT:
            code = _error_code(resp)
            if code == "unit_id_already_submitted":
                raise UnitsAlreadySubmittedError(resp.status_code, resp.text)
            if code == "max_units_exceeded":
                raise MaxUnitsExceededError(resp.status_code, resp.text)
            if code == "submissions_finalized":
                raise SubmissionsFinalizedError(resp.status_code, resp.text)
        if not resp.is_success:
            raise CoordinatorError(resp.status_code, resp.text)
        data = resp.json()
        return SubmitResult(
            submitted_unit_ids=data.get("submitted_unit_ids") or [],
            count=data.get("count") or 0,
        )

    # ---- lifecycle actions -------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Stop accepting submissions (`POST /actions/finalize-submissions`). The
        coordinator auto-completes once the last in-flight unit finishes — this is
        the autonomic loop's computed termination (design note §1, §3.2)."""
        return self._action("finalize-submissions")

    def declare_deviation(self, *, what_changed: str, why: str) -> dict[str, Any]:
        """D16.2-D (§5): declare an append-only deviation from the pre-registered
        design — what changed and why, SIGNED by this tenant key over the
        canonical declaration (the coordinator verifies it against the
        authenticated caller and anchors the record in Rekor, so WHEN the
        analysis changed is publicly provable). Never edits the original;
        exploratory analysis is allowed — it just may not masquerade as
        confirmatory. Raises CoordinatorError on refusal (e.g. the experiment
        was never pre-registered)."""
        from base64 import b64encode

        from auspexai_tenant.signing import canonical_deviation_bytes

        meta = self._metadata()
        declaration = canonical_deviation_bytes(meta["manifest_hash"], what_changed, why)
        resp = self._post(
            f"/api/v0/experiments/{self.experiment_id}/pre-registration/deviations",
            json={
                "what_changed": what_changed,
                "why": why,
                "tenant_signature_b64": b64encode(self._key.sign(declaration)).decode(),
            },
        )
        if resp.status_code != 201:
            raise CoordinatorError(resp.status_code, resp.text)
        return resp.json()

    def abort(self) -> dict[str, Any]:
        """Abort the experiment (`POST /actions/abort`)."""
        return self._action("abort")

    def _action(self, action: str) -> dict[str, Any]:
        resp = self._post(f"/api/v0/experiments/{self.experiment_id}/actions/{action}")
        if resp.status_code == httpx.codes.CONFLICT:
            raise LifecycleConflictError(resp.status_code, resp.text)
        if not resp.is_success:
            raise CoordinatorError(resp.status_code, resp.text)
        return resp.json()

    # ---- reads (poll = source of truth, design note §2) --------------------

    def progress(self) -> Progress:
        """A cheap snapshot for hot-loop polling — experiment status (→
        `reduce_ready`) plus the anonymized activity counts. Always fetched fresh
        (status/counts change); the immutable metadata stays cached."""
        exp = self.reader.get_experiment(self.experiment_id)
        activity = self.reader.get_activity(self.experiment_id)
        status = exp.get("status")
        return Progress(
            status=status,
            reduce_ready=status == REDUCE_READY_STATUS,
            submissions_finalized=bool(exp.get("submissions_finalized")),
            total_work_units=activity.get("total_work_units"),
            work_unit_counts=activity.get("work_unit_counts") or {},
            completions_total=activity.get("completions_total"),
            replication_target_total=activity.get("replication_target_total"),
            active_contributor_count=activity.get("active_contributor_count"),
            network_active_workers=activity.get("network_active_workers"),
        )

    def results_page(self, *, cursor: str | None = None, include: str = "consensus") -> ResultPage:
        """One page of results from `cursor` (the driver's exactly-once primitive
        — it folds the page, then journals `next_cursor`). `include` is
        'consensus' (one agreed payload per unit — the set to fold) or 'raw'."""
        page = self.reader.get_results(self.experiment_id, include=include, cursor=cursor)
        return ResultPage(results=page.get("results") or [], next_cursor=page.get("next_cursor"))

    def results(
        self, *, since: str | None = None, include: str = "consensus"
    ) -> Iterator[dict[str, Any]]:
        """All results from the `since` cursor onward, transparently following
        pagination. Convenience over `results_page`; the driver prefers
        `results_page` so it can journal the cursor between folds."""
        cursor = since
        while True:
            page = self.results_page(cursor=cursor, include=include)
            yield from page.results
            if not page.next_cursor:
                return
            cursor = page.next_cursor

    def attestation(self, *, checkpoint: bool = False) -> ResultSetAttestation:
        """The result-set attestation. By default the final attestation (once
        COMPLETED); `checkpoint=True` is the M9 partial attestation over
        consensus-so-far — the integrity anchor for finalize-on-partial (§3.5)."""
        return self.reader.get_attestation(self.experiment_id, checkpoint=checkpoint)
