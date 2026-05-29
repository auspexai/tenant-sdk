"""Tests for experiment submission — signed POST to the coordinator API.

Uses httpx.MockTransport to test without a real network. The request is
RFC 9421-signed; tests assert the signing headers are present and the JSON
body shape matches `POST /api/v0/experiments`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from click.testing import CliRunner

from auspexai_tenant.cli import main
from auspexai_tenant.signing import MaintainerKey
from auspexai_tenant.upload import submit_experiment, submit_experiment_from_files

FIXTURES = Path(__file__).parent / "fixtures"

_SIG = {"sig_v": "0.1", "maintainer_pubkey": "0" * 64, "signature": "AAAA"}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _manifest() -> dict:
    return json.loads((FIXTURES / "valid_minimal.json").read_text())


# ---- submit_experiment ------------------------------------------------------


def test_submit_posts_signed_json_to_experiments() -> None:
    key = MaintainerKey.generate()
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers.get("content-type", "")
        seen["has_sig_input"] = "signature-input" in request.headers
        seen["has_content_digest"] = "content-digest" in request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, text='{"status":"accepted"}')

    result = submit_experiment(
        _manifest(), _SIG, "https://coord.test/", key, client=_client(handler)
    )
    assert result.ok is True
    assert result.status_code == 201
    assert seen["method"] == "POST"
    assert seen["url"] == "https://coord.test/api/v0/experiments"
    assert "application/json" in seen["content_type"]
    assert seen["has_sig_input"] is True
    assert seen["has_content_digest"] is True
    assert set(seen["body"]) == {"manifest", "signature"}
    assert seen["body"]["signature"] == _SIG


def test_submit_keyid_matches_signing_key() -> None:
    key = MaintainerKey.generate()
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["sig_input"] = request.headers.get("signature-input", "")
        return httpx.Response(201, text="{}")

    submit_experiment(_manifest(), _SIG, "https://coord.test", key, client=_client(handler))
    assert f'keyid="{key.pubkey_hex.lower()}"' in seen["sig_input"]


def test_submit_returns_status_for_4xx() -> None:
    key = MaintainerKey.generate()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text='{"error":"manifest_tenant_mismatch"}')

    result = submit_experiment(
        _manifest(), _SIG, "https://coord.test", key, client=_client(handler)
    )
    assert result.ok is False
    assert result.status_code == 403
    assert "manifest_tenant_mismatch" in result.body


def test_submit_strips_trailing_slash() -> None:
    key = MaintainerKey.generate()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(201, text="{}")

    submit_experiment(_manifest(), _SIG, "https://coord.test/////", key, client=_client(handler))
    assert seen[0] == "https://coord.test/api/v0/experiments"


def test_submit_from_files(tmp_path: Path) -> None:
    key = MaintainerKey.generate()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())
    sig_path = tmp_path / "manifest.json.sig"
    sig_path.write_text(json.dumps(_SIG))

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, text="{}")

    result = submit_experiment_from_files(
        manifest_path, sig_path, "https://coord.test", key, client=_client(handler)
    )
    assert result.ok is True
    assert seen["body"]["signature"] == _SIG


# ---- CLI: manifest upload ---------------------------------------------------


def test_cli_manifest_upload_dry_run(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    sig_path.write_text(json.dumps(_SIG))

    result = runner.invoke(
        main,
        ["manifest", "upload", str(manifest_path), "--coordinator", "https://coord.test", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert "POST https://coord.test/api/v0/experiments" in result.output
    assert "signed" in result.output


def test_cli_manifest_upload_requires_signature(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    result = runner.invoke(
        main,
        ["manifest", "upload", str(manifest_path), "--coordinator", "https://coord.test", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "signature file not found" in result.output


def test_cli_manifest_upload_rejects_missing_sig_path(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    result = runner.invoke(
        main,
        [
            "manifest", "upload", str(manifest_path),
            "--coordinator", "https://coord.test",
            "--sig", "/nonexistent/sig.sig",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "signature file not found" in result.output
