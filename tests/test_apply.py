"""Tests for `auspexai-tenant apply` (apply-from-CLI tenant onboarding, Option D).

Three layers, all offline (no live network anywhere):

  - the GitHub Device Flow helper against an httpx.MockTransport with scripted
    poll sequences (injected clock/sleep — no real waits),
  - TenantClient.apply_for_tenant / my_tenant_applications wire shape: the POST
    is RFC 9421-signed by the LOCAL (possibly unregistered) key with the body
    under content-digest,
  - the `apply` CLI via CliRunner with the device-flow + client seams patched
    (fresh-key generation, existing-key reuse, --status rendering, fail-fast on
    missing fields).
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
from click.testing import CliRunner

from auspexai_tenant import cli as cli_mod
from auspexai_tenant.cli import main
from auspexai_tenant.client import TenantClient
from auspexai_tenant.github_device_flow import (
    CLIENT_ID_ENV,
    GITHUB_CLIENT_ID,
    GITHUB_DEVICE_CODE_URL,
    GITHUB_TOKEN_URL,
    AccessDeniedError,
    DeviceCode,
    DeviceFlowError,
    ExpiredTokenError,
    default_client_id,
    run_device_flow,
)
from auspexai_tenant.signing import MaintainerKey

COORD = "https://coord.test"


# ---- device flow helper ------------------------------------------------------


def _device_code_json() -> dict[str, object]:
    return {
        "device_code": "dev-123",
        "user_code": "ABCD-9999",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }


def _scripted(responses: list[httpx.Response]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A MockTransport that returns `responses` in order, recording requests.
    Any extra call fails loudly."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if len(seen) >= len(responses):
            raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
        seen.append(request)
        return responses[len(seen) - 1]

    return httpx.MockTransport(handler), seen


def test_device_flow_poll_loop_pending_slow_down_then_token() -> None:
    transport, seen = _scripted(
        [
            httpx.Response(200, json=_device_code_json()),
            httpx.Response(400, json={"error": "authorization_pending"}),
            httpx.Response(400, json={"error": "slow_down"}),
            httpx.Response(200, json={"access_token": "gho_ok", "token_type": "bearer"}),
        ]
    )
    sleeps: list[float] = []
    codes: list[DeviceCode] = []
    token = run_device_flow(
        on_code=codes.append,
        client_id="cid-test",
        transport=transport,
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )
    assert token == "gho_ok"
    # the caller was shown the user code + verification url exactly once
    assert len(codes) == 1
    assert codes[0].user_code == "ABCD-9999"
    assert codes[0].verification_uri == "https://github.com/login/device"
    # pending keeps the interval; slow_down adds 5s (RFC 8628 §3.5)
    assert sleeps == [5, 5, 10]
    # wire shape: device-code request carries the client_id + identity-only scope
    assert str(seen[0].url) == GITHUB_DEVICE_CODE_URL
    start = parse_qs(seen[0].content.decode())
    assert start["client_id"] == ["cid-test"]
    assert start["scope"] == ["read:user"]
    # token polls carry the device_code + RFC 8628 grant type
    assert str(seen[1].url) == GITHUB_TOKEN_URL
    poll = parse_qs(seen[1].content.decode())
    assert poll["client_id"] == ["cid-test"]
    assert poll["device_code"] == ["dev-123"]
    assert poll["grant_type"] == ["urn:ietf:params:oauth:grant-type:device_code"]


def test_device_flow_access_denied_is_terminal() -> None:
    transport, _ = _scripted(
        [
            httpx.Response(200, json=_device_code_json()),
            httpx.Response(400, json={"error": "access_denied"}),
        ]
    )
    with pytest.raises(AccessDeniedError):
        run_device_flow(
            on_code=lambda c: None,
            client_id="cid-test",
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )


def test_device_flow_expired_token_is_terminal() -> None:
    transport, _ = _scripted(
        [
            httpx.Response(200, json=_device_code_json()),
            httpx.Response(400, json={"error": "expired_token"}),
        ]
    )
    with pytest.raises(ExpiredTokenError):
        run_device_flow(
            on_code=lambda c: None,
            client_id="cid-test",
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )


def test_device_flow_deadline_exceeded_raises_expired() -> None:
    transport, _ = _scripted(
        [
            httpx.Response(200, json=_device_code_json()),
            httpx.Response(400, json={"error": "authorization_pending"}),
        ]
    )
    now = {"t": 0.0}

    def sleep(seconds: float) -> None:
        now["t"] += seconds + 1000  # push past the 900s expires_in

    with pytest.raises(ExpiredTokenError, match="expired"):
        run_device_flow(
            on_code=lambda c: None,
            client_id="cid-test",
            transport=transport,
            clock=lambda: now["t"],
            sleep=sleep,
        )


def test_device_flow_unknown_error_raises() -> None:
    transport, _ = _scripted(
        [
            httpx.Response(200, json=_device_code_json()),
            httpx.Response(400, json={"error": "weird_error"}),
        ]
    )
    with pytest.raises(DeviceFlowError, match="weird_error"):
        run_device_flow(
            on_code=lambda c: None,
            client_id="cid-test",
            transport=transport,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )


def test_default_client_id_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CLIENT_ID_ENV, raising=False)
    # default = the network's public OAuth App id (same App the worker uses)
    assert default_client_id() == GITHUB_CLIENT_ID
    monkeypatch.setenv(CLIENT_ID_ENV, "cid-override")
    assert default_client_id() == "cid-override"


# ---- client wire shape -------------------------------------------------------


def _tenant_client(handler, key: MaintainerKey) -> TenantClient:
    return TenantClient(COORD, key, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_apply_for_tenant_posts_signed_body() -> None:
    key = MaintainerKey.generate()
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["content"] = request.content
        return httpx.Response(201, json={"application_id": "app-1", "status": "pending"})

    out = _tenant_client(handler, key).apply_for_tenant(
        github_access_token="gho_secret",
        requested_tenant_id="drift-lab",
        contact_name="Ada Lovelace",
        affiliation="Analytical Engines Ltd",
        research_summary="Output drift across quantized local models.",
    )
    assert out == {"application_id": "app-1", "status": "pending"}
    assert (seen["method"], seen["path"]) == ("POST", "/api/v0/tenant-applications")
    # RFC 9421: signed by the LOCAL (not-yet-registered) key — the keyid IS the
    # applicant's pubkey, which is how the coordinator learns it.
    assert "Signature" in seen["headers"]
    sig_input = seen["headers"]["Signature-Input"]
    assert f'keyid="{key.pubkey_hex}"' in sig_input
    # the body is covered by the signature via content-digest
    assert '"content-digest"' in sig_input
    digest = base64.b64encode(hashlib.sha256(seen["content"]).digest()).decode("ascii")
    assert seen["headers"]["Content-Digest"] == f"sha-256=:{digest}:"
    assert json.loads(seen["content"]) == {
        "github_access_token": "gho_secret",
        "requested_tenant_id": "drift-lab",
        "contact_name": "Ada Lovelace",
        "affiliation": "Analytical Engines Ltd",
        "research_summary": "Output drift across quantized local models.",
    }


def test_my_tenant_applications_signed_get() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["signed"] = "Signature" in request.headers
        return httpx.Response(
            200, json={"applications": [{"application_id": "app-1", "status": "pending"}]}
        )

    apps = _tenant_client(handler, MaintainerKey.generate()).my_tenant_applications()
    assert apps == [{"application_id": "app-1", "status": "pending"}]
    assert seen["path"] == "/api/v0/tenant-applications/mine"
    assert seen["signed"]


# ---- CLI ----------------------------------------------------------------------


def _patch_device_flow(
    monkeypatch: pytest.MonkeyPatch, *, token: str = "gho_test", calls: list | None = None
) -> None:
    """Replace the CLI's device flow with one that 'shows' a code and returns
    `token`, recording the client_id it was handed."""

    def fake_run_device_flow(*, on_code, client_id=None, **kwargs):
        if calls is not None:
            calls.append(client_id)
        on_code(
            DeviceCode(
                device_code="dev-123",
                user_code="ABCD-9999",
                verification_uri="https://github.com/login/device",
                expires_in=900,
                interval=5,
            )
        )
        return token

    monkeypatch.setattr(cli_mod, "run_device_flow", fake_run_device_flow)


def _patch_apply(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """Capture TenantClient.apply_for_tenant kwargs + the pubkey of the key the
    client signs with; return a canned 201 body."""

    def fake_apply(self, **kwargs):
        captured.update(kwargs)
        captured["signing_pubkey"] = self._auth._key.pubkey_hex
        return {"application_id": "app-9", "status": "pending"}

    monkeypatch.setattr(TenantClient, "apply_for_tenant", fake_apply)


_APPLY_FIELDS = [
    "--tenant-id",
    "drift-lab",
    "--name",
    "Ada Lovelace",
    "--affiliation",
    "Analytical Engines Ltd",
    "--summary",
    "Output drift across quantized local models.",
]


def test_cli_apply_generates_fresh_key_and_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CLIENT_ID_ENV, raising=False)
    key_path = tmp_path / "tenant_key.pem"
    client_ids: list[str] = []
    _patch_device_flow(monkeypatch, token="gho_fresh", calls=client_ids)
    captured: dict = {}
    _patch_apply(monkeypatch, captured)

    result = CliRunner().invoke(
        main,
        ["apply", "--coordinator", COORD, "--key", str(key_path), *_APPLY_FIELDS],
    )
    assert result.exit_code == 0, result.output
    # a fresh key was created at the given path and the user was told
    assert key_path.exists()
    assert "Generated a new tenant key" in result.output
    saved = MaintainerKey.load(key_path)
    assert saved.pubkey_hex in result.output
    # the application was signed by THAT key and carried all the fields + token
    assert captured["signing_pubkey"] == saved.pubkey_hex
    assert captured["github_access_token"] == "gho_fresh"
    assert captured["requested_tenant_id"] == "drift-lab"
    assert captured["contact_name"] == "Ada Lovelace"
    assert captured["affiliation"] == "Analytical Engines Ltd"
    assert captured["research_summary"] == "Output drift across quantized local models."
    # the device-flow UX surfaced the code + url, using the public client id
    assert "ABCD-9999" in result.output
    assert "https://github.com/login/device" in result.output
    assert client_ids == [GITHUB_CLIENT_ID]
    # submission receipt + how to track
    assert "app-9" in result.output
    assert "apply --status" in result.output


def test_cli_apply_reuses_existing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = tmp_path / "tenant_key.pem"
    existing = MaintainerKey.generate()
    existing.save(key_path)
    _patch_device_flow(monkeypatch)
    captured: dict = {}
    _patch_apply(monkeypatch, captured)

    result = CliRunner().invoke(
        main,
        ["apply", "--coordinator", COORD, "--key", str(key_path), *_APPLY_FIELDS],
    )
    assert result.exit_code == 0, result.output
    assert "Generated a new tenant key" not in result.output
    assert captured["signing_pubkey"] == existing.pubkey_hex


def test_cli_apply_missing_field_fails_before_key_or_device_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-interactively (no tty), a missing field errors out without creating
    a key or burning a device code."""

    def must_not_run(**kwargs):
        raise AssertionError("device flow must not run when fields are missing")

    monkeypatch.setattr(cli_mod, "run_device_flow", must_not_run)
    key_path = tmp_path / "tenant_key.pem"
    result = CliRunner().invoke(
        main,
        # --tenant-id deliberately absent
        ["apply", "--coordinator", COORD, "--key", str(key_path), *_APPLY_FIELDS[2:]],
    )
    assert result.exit_code == 1
    assert "--tenant-id" in result.output
    assert not key_path.exists()


def test_cli_apply_device_flow_failure_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def declined(**kwargs):
        raise AccessDeniedError("the GitHub authorization request was declined")

    monkeypatch.setattr(cli_mod, "run_device_flow", declined)
    result = CliRunner().invoke(
        main,
        ["apply", "--coordinator", COORD, "--key", str(tmp_path / "k.pem"), *_APPLY_FIELDS],
    )
    assert result.exit_code == 1
    assert "declined" in result.output


def test_cli_apply_status_renders_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = tmp_path / "tenant_key.pem"
    MaintainerKey.generate().save(key_path)
    apps = [
        {
            "application_id": "app-1",
            "status": "pending",
            "resolution_reason": None,
            "created_tenant_id": None,
        },
        {
            "application_id": "app-2",
            "status": "approved",
            "resolution_reason": None,
            "created_tenant_id": "drift-lab",
        },
        {
            "application_id": "app-3",
            "status": "declined",
            "resolution_reason": "no research summary provided",
            "created_tenant_id": None,
        },
    ]
    monkeypatch.setattr(TenantClient, "my_tenant_applications", lambda self: apps)
    result = CliRunner().invoke(
        main, ["apply", "--status", "--coordinator", COORD, "--key", str(key_path)]
    )
    assert result.exit_code == 0, result.output
    assert "app-1" in result.output and "pending" in result.output
    # approved → created_tenant_id surfaced; declined → resolution_reason surfaced
    assert "app-2" in result.output and "drift-lab" in result.output
    assert "app-3" in result.output and "no research summary provided" in result.output


def test_cli_apply_status_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = tmp_path / "tenant_key.pem"
    MaintainerKey.generate().save(key_path)
    monkeypatch.setattr(TenantClient, "my_tenant_applications", lambda self: [])
    result = CliRunner().invoke(
        main, ["apply", "--status", "--coordinator", COORD, "--key", str(key_path)]
    )
    assert result.exit_code == 0, result.output
    assert "(no applications)" in result.output
