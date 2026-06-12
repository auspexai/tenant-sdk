"""GitHub Device Authorization Flow (RFC 8628) for `auspexai-tenant apply`.

A researcher proves control of a GitHub identity entirely from the CLI: the SDK
requests a device code, the researcher enters the short user code at
github.com/login/device in any browser, and the SDK polls until GitHub issues
an access token. The token rides inside the RFC 9421-signed tenant application
to the coordinator and is never persisted locally.

This is a standalone Apache-2.0 implementation of the same wire protocol the
AGPL-3.0 worker speaks (`auspexai_worker.oauth.device_flow`) — as with
`http_signing`, the shared contract is the wire format (RFC 8628 + the
network's GitHub OAuth App), not shared code. The default Client ID below is
the auspexai-org OAuth App's public-by-design identifier (Device Flow uses the
public-client model; no client secret is involved), the same App the worker's
`login` uses. Override it with the AUSPEXAI_GITHUB_CLIENT_ID environment
variable. Identity scope only (`read:user`), matching the worker.

RFC 8628 §3.5 poll semantics:

  - authorization_pending → keep polling at the current interval
  - slow_down             → add 5 seconds to the interval, keep polling
  - expired_token         → terminal (the user didn't approve in time)
  - access_denied         → terminal (the user explicitly declined)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

#: Public-by-design Client ID of the auspexai-org GitHub OAuth App (the same
#: App the worker uses for volunteer login; Device Flow enabled, no secret).
GITHUB_CLIENT_ID = "Ov23lierutLLeF9skyHu"
#: Environment variable that overrides the Client ID (e.g. for a staging App).
CLIENT_ID_ENV = "AUSPEXAI_GITHUB_CLIENT_ID"

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
#: Identity scope only — enough for the coordinator to resolve who applied.
GITHUB_SCOPE = "read:user"

_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_REQUEST_TIMEOUT_S = 10.0


def default_client_id() -> str:
    """The Client ID to use: AUSPEXAI_GITHUB_CLIENT_ID when set, else the
    network's public OAuth App id."""
    return os.environ.get(CLIENT_ID_ENV) or GITHUB_CLIENT_ID


class DeviceFlowError(Exception):
    """Base class for Device Flow failures."""


class ExpiredTokenError(DeviceFlowError):
    """The device code expired before the user authorized (RFC 8628 §3.5)."""


class AccessDeniedError(DeviceFlowError):
    """The user explicitly declined the authorization (RFC 8628 §3.5)."""


@dataclass(frozen=True)
class DeviceCode:
    """The codes returned by the initial device-code request."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


def request_device_code(
    *,
    client_id: str,
    scope: str = GITHUB_SCOPE,
    transport: httpx.BaseTransport | None = None,
) -> DeviceCode:
    """Start a Device Flow: fetch the user_code + verification_uri to display."""
    with httpx.Client(transport=transport, timeout=_REQUEST_TIMEOUT_S) as client:
        try:
            response = client.post(
                GITHUB_DEVICE_CODE_URL,
                data={"client_id": client_id, "scope": scope},
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise DeviceFlowError(f"device-code request failed: {exc}") from exc
    if response.status_code != 200:
        raise DeviceFlowError(
            f"device-code request returned HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    try:
        return DeviceCode(
            device_code=payload["device_code"],
            user_code=payload["user_code"],
            verification_uri=payload["verification_uri"],
            expires_in=int(payload["expires_in"]),
            interval=int(payload["interval"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeviceFlowError(f"device-code response missing/invalid fields: {payload!r}") from exc


def run_device_flow(
    *,
    on_code: Callable[[DeviceCode], None],
    client_id: str | None = None,
    scope: str = GITHUB_SCOPE,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run the full Device Flow; return the GitHub access token.

    `on_code` is called once with the DeviceCode so the caller can show the
    user_code + verification_uri. `clock` and `sleep` are injectable so tests
    never wait on real time; `transport` lets tests script both endpoints
    through an `httpx.MockTransport`.
    """
    cid = client_id if client_id is not None else default_client_id()
    code = request_device_code(client_id=cid, scope=scope, transport=transport)
    on_code(code)

    deadline = clock() + code.expires_in
    interval = float(code.interval)
    with httpx.Client(transport=transport, timeout=_REQUEST_TIMEOUT_S) as client:
        while clock() < deadline:
            sleep(interval)
            try:
                response = client.post(
                    GITHUB_TOKEN_URL,
                    data={
                        "client_id": cid,
                        "device_code": code.device_code,
                        "grant_type": _GRANT_TYPE,
                    },
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise DeviceFlowError(f"token poll request failed: {exc}") from exc
            if response.status_code not in (200, 400):
                raise DeviceFlowError(
                    f"token poll returned unexpected HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            payload = response.json()
            if payload.get("access_token"):
                return str(payload["access_token"])
            error = str(payload.get("error", "unknown_error"))
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                raise ExpiredTokenError(
                    "device code expired before authorization; run `auspexai-tenant apply` again"
                )
            if error == "access_denied":
                raise AccessDeniedError("the GitHub authorization request was declined")
            raise DeviceFlowError(f"token poll returned unexpected error code {error!r}")
    raise ExpiredTokenError("device code expired before authorization completed")
