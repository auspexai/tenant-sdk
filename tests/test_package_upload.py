"""Tests for executor-package upload (#40a researcher leg).

Three layers, all offline:

  - `build_package_archive` canonical form: byte-identical rebuilds (incl.
    across mtime / creation-order differences), digest-lockstep exclusions,
    fixed entry metadata, symlink dereferencing, gzip-header determinism;
  - `TenantClient.upload_package` wire shape against httpx.MockTransport:
    raw application/gzip body, X-Package-Digest = the tree digest, the RFC 9421
    signature binding the BINARY body via Content-Digest (verified end-to-end
    against the pubkey), and 422/413 envelope surfacing;
  - the `package upload` CLI happy path via CliRunner.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tarfile
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auspexai_tenant import cli as cli_mod
from auspexai_tenant.cli import main
from auspexai_tenant.client import CoordinatorError, TenantClient
from auspexai_tenant.http_signing import build_signature_base
from auspexai_tenant.manifest import compute_package_digest
from auspexai_tenant.package import build_package_archive
from auspexai_tenant.signing import TenantKey

COORD = "https://coord.test"

_FILES: tuple[tuple[str, bytes], ...] = (
    ("executor.py", b"print('run')\n"),
    ("helper.py", b"# helper\n"),
    ("lib/util.py", b"X = 1\n"),
)


def _make_pkg(root: Path, files: tuple[tuple[str, bytes], ...] = _FILES) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _members(archive: bytes) -> list[tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        return tf.getmembers()


# ---- archive determinism -----------------------------------------------------


def test_two_builds_are_byte_identical(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path / "pkg")
    assert build_package_archive(pkg) == build_package_archive(pkg)


def test_archive_independent_of_mtime_and_creation_order(tmp_path: Path) -> None:
    """Same tree → identical bytes, even when the trees were written in a
    different order with wildly different timestamps and modes."""
    pkg_a = _make_pkg(tmp_path / "a", _FILES)
    pkg_b = _make_pkg(tmp_path / "b", tuple(reversed(_FILES)))
    for i, (rel, _) in enumerate(_FILES):
        os.utime(pkg_a / rel, (0, 0))
        os.utime(pkg_b / rel, (1_700_000_000 + i, 1_700_000_000 + i))
    (pkg_a / "executor.py").chmod(0o755)  # exec bit must not leak into the bytes
    assert build_package_archive(pkg_a) == build_package_archive(pkg_b)


def test_archive_changes_when_content_changes(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path / "pkg")
    before = build_package_archive(pkg)
    (pkg / "executor.py").write_bytes(b"print('edited')\n")
    assert build_package_archive(pkg) != before


def test_gzip_header_has_no_timestamp(tmp_path: Path) -> None:
    """MTIME bytes 4-7 of the gzip header are zero — no wall clock in the bytes."""
    archive = build_package_archive(_make_pkg(tmp_path / "pkg"))
    assert archive[:2] == b"\x1f\x8b"
    assert archive[4:8] == b"\x00\x00\x00\x00"


# ---- archive canonical form ---------------------------------------------------


def test_archive_entries_sorted_with_fixed_metadata(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path / "pkg")
    members = _members(build_package_archive(pkg))
    names = [m.name for m in members]
    assert names == sorted(rel for rel, _ in _FILES)
    for m in members:
        assert m.isfile()
        assert (m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "")
        assert m.mtime == 0
        assert m.mode == 0o644


def test_archive_round_trips_content(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path / "pkg")
    with tarfile.open(fileobj=io.BytesIO(build_package_archive(pkg)), mode="r:gz") as tf:
        for rel, content in _FILES:
            extracted = tf.extractfile(rel)
            assert extracted is not None
            assert extracted.read() == content


def test_archive_excludes_what_the_digest_excludes(tmp_path: Path) -> None:
    """The archive's file set is the digest's file set — manifest.json[.sig],
    __pycache__/ and *.pyc never ship."""
    pkg = _make_pkg(tmp_path / "pkg", (("executor.py", b"print('run')\n"),))
    digest_before = compute_package_digest(pkg)
    (pkg / "manifest.json").write_bytes(b"{}")
    (pkg / "manifest.json.sig").write_bytes(b'{"sig": "..."}')
    (pkg / "stray.pyc").write_bytes(b"\x00")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "executor.cpython-312.pyc").write_bytes(b"\x00\x01")
    members = _members(build_package_archive(pkg))
    assert [m.name for m in members] == ["executor.py"]
    # ...and the digest the upload pins is equally unaffected.
    assert compute_package_digest(pkg) == digest_before


def test_archive_dereferences_symlinks(tmp_path: Path) -> None:
    """No link entries: a symlink ships as a regular file with the target's
    bytes — the digest's content-addressed view of the tree."""
    pkg = _make_pkg(tmp_path / "pkg", (("executor.py", b"print('run')\n"),))
    (pkg / "alias.py").symlink_to(pkg / "executor.py")
    archive = build_package_archive(pkg)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        alias = tf.getmember("alias.py")
        assert alias.isfile() and not alias.issym()
        extracted = tf.extractfile("alias.py")
        assert extracted is not None
        assert extracted.read() == b"print('run')\n"


# ---- TenantClient.upload_package ----------------------------------------------


def _upload(tmp_path: Path, handler) -> tuple[dict, Path, TenantKey]:
    pkg = _make_pkg(tmp_path / "pkg")
    key = TenantKey.generate()
    client = TenantClient(COORD, key, client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client.upload_package(pkg), pkg, key


def test_upload_sends_signed_binary_post(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["content_type"] = request.headers.get("content-type")
        seen["x_package_digest"] = request.headers.get("x-package-digest")
        seen["content_digest"] = request.headers.get("content-digest")
        seen["sig_input"] = request.headers.get("signature-input", "")
        seen["body"] = request.content
        return httpx.Response(
            201,
            json={"package_digest": request.headers["x-package-digest"], "status": "stored"},
        )

    out, pkg, _key = _upload(tmp_path, handler)
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/v0/packages"
    assert seen["content_type"] == "application/gzip"
    # The body is the deterministic archive; the digest header is the TREE
    # digest (== the manifest's executor.package_sha256 pin), not the tar hash.
    assert seen["body"] == build_package_archive(pkg)
    assert seen["x_package_digest"] == compute_package_digest(pkg)
    # Content-Digest is SHA-256 of the RAW BINARY bytes, and it is covered.
    body = seen["body"]
    assert isinstance(body, bytes)
    expected_cd = f"sha-256=:{base64.b64encode(hashlib.sha256(body).digest()).decode()}:"
    assert seen["content_digest"] == expected_cd
    assert '"content-digest"' in str(seen["sig_input"])
    assert out == {"package_digest": compute_package_digest(pkg), "status": "stored"}


def test_upload_signature_verifies_over_binary_body(tmp_path: Path) -> None:
    """End-to-end: re-derive the RFC 9421 signature base the coordinator builds
    and verify the Ed25519 signature — proof the gzip body is cryptographically
    bound, not just hashed into an uncovered header."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["sig_input"] = request.headers["signature-input"]
        seen["signature"] = request.headers["signature"]
        seen["content_digest"] = request.headers["content-digest"]
        return httpx.Response(201, json={"package_digest": "x", "status": "stored"})

    _out, _pkg, key = _upload(tmp_path, handler)
    raw = seen["sig_input"][len("sig1=") :]
    base = build_signature_base(
        covered=("@method", "@path", "@authority", "content-digest"),
        raw_covered_and_params=raw,
        method="POST",
        path="/api/v0/packages",
        authority="coord.test",
        content_digest_header=seen["content_digest"],
    )
    sig = base64.b64decode(seen["signature"][len("sig1=:") : -1])
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key.pubkey_hex))
    pub.verify(sig, base)  # raises InvalidSignature on failure


def test_upload_returns_already_exists_envelope(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "package_digest": request.headers["x-package-digest"],
                "status": "already_exists",
            },
        )

    out, pkg, _key = _upload(tmp_path, handler)
    assert out == {"package_digest": compute_package_digest(pkg), "status": "already_exists"}


@pytest.mark.parametrize(
    ("status", "detail"),
    [(422, "digest_mismatch"), (413, "package exceeds the size limit")],
)
def test_upload_surfaces_4xx_envelopes(tmp_path: Path, status: int, detail: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": detail})

    with pytest.raises(CoordinatorError) as excinfo:
        _upload(tmp_path, handler)
    assert excinfo.value.status_code == status
    assert detail in excinfo.value.body


# ---- CLI ----------------------------------------------------------------------


def test_cli_package_upload_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = _make_pkg(tmp_path / "pkg")
    key_path = tmp_path / "key.pem"
    TenantKey.generate().save(key_path)
    digest = compute_package_digest(pkg)
    captured: dict[str, Path] = {}

    def fake_upload(self: TenantClient, package_dir: str | Path) -> dict[str, str]:
        captured["package_dir"] = Path(package_dir)
        return {"package_digest": compute_package_digest(package_dir), "status": "stored"}

    monkeypatch.setattr(TenantClient, "upload_package", fake_upload)
    result = CliRunner().invoke(
        main,
        ["package", "upload", str(pkg), "--coordinator", COORD, "--key", str(key_path)],
    )
    assert result.exit_code == 0, result.output
    assert captured["package_dir"] == pkg
    assert digest in result.output
    assert "stored" in result.output
    assert "executor.package_sha256" in result.output  # the pin-this-digest hint


def test_cli_package_upload_defaults_to_public_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--coordinator is optional (front-door default, mirroring `apply`)."""
    pkg = _make_pkg(tmp_path / "pkg")
    seen: dict[str, str] = {}

    class FakeClient:
        def upload_package(self, package_dir: str | Path) -> dict[str, str]:
            return {"package_digest": "ab" * 32, "status": "already_exists"}

    def fake_make_client(coordinator: str, key_path: Path) -> FakeClient:
        seen["coordinator"] = coordinator
        return FakeClient()

    monkeypatch.delenv("AUSPEXAI_COORDINATOR_URL", raising=False)
    monkeypatch.setattr(cli_mod, "_make_client", fake_make_client)
    result = CliRunner().invoke(main, ["package", "upload", str(pkg)])
    assert result.exit_code == 0, result.output
    assert seen["coordinator"] == "https://coord.auspexai.network"
    assert "already_exists" in result.output
