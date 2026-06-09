"""Tests for the executor-package digest (`compute_package_digest`) + the
manifest `executor.package_sha256` provenance pin.

The digest is the shared contract the worker re-implements byte-for-byte, so this
also pins a known vector (a fixed file set → fixed digest) that the worker test
asserts against — if either side drifts, a staged executor would wrongly refuse.
"""

from __future__ import annotations

import hashlib

from auspexai_tenant.manifest import Executor, compute_package_digest

# A fixed two-file package and its digest, computed by hand from the documented
# canonical form (`<relpath>\x00<sha256>` lines, sorted, joined by \n, hashed).
_A = b"print('a')\n"
_B = b"# helper\n"


def _expected(*files: tuple[str, bytes]) -> str:
    lines = [f"{rel}\x00{hashlib.sha256(content).hexdigest()}" for rel, content in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_digest_of_known_package(tmp_path) -> None:
    (tmp_path / "executor.py").write_bytes(_A)
    (tmp_path / "helper.py").write_bytes(_B)
    assert compute_package_digest(tmp_path) == _expected(("executor.py", _A), ("helper.py", _B))


def test_digest_excludes_manifest_and_pycache(tmp_path) -> None:
    (tmp_path / "executor.py").write_bytes(_A)
    (tmp_path / "manifest.json").write_bytes(b'{"x":1}')
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "executor.cpython-311.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "stray.pyc").write_bytes(b"\x00")
    # Only executor.py counts.
    assert compute_package_digest(tmp_path) == _expected(("executor.py", _A))


def test_digest_is_order_independent_and_recursive(tmp_path) -> None:
    (tmp_path / "b.py").write_bytes(_B)
    (tmp_path / "a.py").write_bytes(_A)
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "c.py").write_bytes(_A)
    assert compute_package_digest(tmp_path) == _expected(
        ("a.py", _A), ("b.py", _B), ("lib/c.py", _A)
    )


def test_digest_changes_when_content_changes(tmp_path) -> None:
    f = tmp_path / "executor.py"
    f.write_bytes(_A)
    before = compute_package_digest(tmp_path)
    f.write_bytes(_A + b"# edited\n")
    assert compute_package_digest(tmp_path) != before


def test_empty_package_is_stable(tmp_path) -> None:
    assert compute_package_digest(tmp_path) == hashlib.sha256(b"").hexdigest()


def test_executor_accepts_package_sha256() -> None:
    digest = "ab" * 32
    ex = Executor(command=["python", "executor.py"], package_sha256=digest)
    assert ex.package_sha256 == digest
    assert Executor(command=["python", "executor.py"]).package_sha256 is None


def test_manifest_sig_is_excluded(tmp_path):
    """`manifest sign` writes <manifest>.sig INTO the package dir by default,
    after the digest was computed — it must not poison executor.package_sha256
    (every unit would otherwise be refused on the worker's re-derivation)."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "executor.py").write_text("print('hi')")
    before = compute_package_digest(pkg)
    (pkg / "manifest.json").write_text("{}")
    (pkg / "manifest.json.sig").write_text('{"sig": "..."}')
    assert compute_package_digest(pkg) == before
