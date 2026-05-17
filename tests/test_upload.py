"""Tests for manifest upload — POST to the coordinator API.

Uses httpx.MockTransport to test without a real network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from click.testing import CliRunner

from auspexai_tenant.cli import main
from auspexai_tenant.upload import upload_manifest

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=_mock_transport(handler))


# ---- upload_manifest --------------------------------------------------------


def test_upload_manifest_only_no_signature(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    received: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["method"] = request.method
        received["url"] = str(request.url)
        received["content_type"] = request.headers.get("content-type", "")
        received["body_len"] = len(request.content)
        return httpx.Response(201, text='{"status":"accepted"}')

    result = upload_manifest(
        manifest_path,
        "https://coord.test/",
        signature_path=None,
        client=_client(handler),
    )
    assert result.ok is True
    assert result.status_code == 201
    assert received["method"] == "POST"
    assert received["url"] == "https://coord.test/api/v0/manifests"
    assert "multipart/form-data" in received["content_type"]


def test_upload_with_signature(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())
    sig_path = tmp_path / "manifest.json.sig"
    sig_path.write_text('{"sig_v":"0.1","maintainer_pubkey":"' + "0" * 64 + '","signature":"AAAA"}')

    body_seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body_seen.append(request.content)
        return httpx.Response(200, text="ok")

    result = upload_manifest(manifest_path, "https://coord.test", sig_path, client=_client(handler))
    assert result.ok is True
    body = body_seen[0].decode("utf-8", errors="replace")
    # Both form parts should be in the multipart body
    assert "manifest.json" in body
    assert "manifest.json.sig" in body


def test_upload_returns_status_for_4xx(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"bad manifest"}')

    result = upload_manifest(manifest_path, "https://coord.test", client=_client(handler))
    assert result.ok is False
    assert result.status_code == 400
    assert "bad manifest" in result.body


def test_upload_strips_trailing_slash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    seen_url: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_url.append(str(request.url))
        return httpx.Response(200, text="")

    upload_manifest(manifest_path, "https://coord.test/////", client=_client(handler))
    assert seen_url[0] == "https://coord.test/api/v0/manifests"


# ---- CLI: manifest upload ---------------------------------------------------


def test_cli_manifest_upload_dry_run(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    result = runner.invoke(
        main,
        [
            "manifest",
            "upload",
            str(manifest_path),
            "--coordinator",
            "https://coord.test",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output
    assert "POST https://coord.test/api/v0/manifests" in result.output
    assert "manifest:" in result.output


def test_cli_manifest_upload_dry_run_picks_up_default_sig(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    sig_path.write_text('{"sig_v":"0.1","maintainer_pubkey":"' + "0" * 64 + '","signature":"AAAA"}')

    result = runner.invoke(
        main,
        [
            "manifest",
            "upload",
            str(manifest_path),
            "--coordinator",
            "https://coord.test",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "signature:" in result.output
    assert str(sig_path) in result.output


def test_cli_manifest_upload_rejects_missing_sig_path(tmp_path: Path) -> None:
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    result = runner.invoke(
        main,
        [
            "manifest",
            "upload",
            str(manifest_path),
            "--coordinator",
            "https://coord.test",
            "--sig",
            "/nonexistent/sig.sig",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output
