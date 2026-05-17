"""Tests for tar_writer / tar_reader (static work-unit-source helpers)."""

from __future__ import annotations

import hashlib
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auspexai_tenant.workunits import WorkUnit, tar_reader, tar_writer


def _make_unit(unit_id: str, value: int) -> WorkUnit:
    return WorkUnit(
        schema_version="0.1",
        unit_id=unit_id,
        tenant_id="synth-doubler",
        experiment_id="synth-doubler-v1",
        manifest_sha256="0" * 64,
        created_at=datetime(2026, 5, 17, 20, 0, 0, tzinfo=UTC),
        payload={"value": value},
    )


def test_tar_writer_round_trip(tmp_path: Path) -> None:
    units = [_make_unit(f"u{i:04d}", i) for i in range(1, 6)]
    out_path = tmp_path / "work_units.tar.gz"

    count = tar_writer(units, out_path)
    assert count == 5
    assert out_path.is_file()

    read_back = tar_reader(out_path)
    assert len(read_back) == 5
    assert {u.unit_id for u in read_back} == {f"u{i:04d}" for i in range(1, 6)}
    assert {u.payload["value"] for u in read_back} == {1, 2, 3, 4, 5}


def test_tar_writer_rejects_duplicate_unit_id(tmp_path: Path) -> None:
    units = [_make_unit("u0001", 1), _make_unit("u0001", 2)]
    out_path = tmp_path / "dupes.tar.gz"
    with pytest.raises(ValueError, match="duplicate unit_id"):
        tar_writer(units, out_path)


def test_tar_writer_creates_parent_dir(tmp_path: Path) -> None:
    units = [_make_unit("u0001", 1)]
    out_path = tmp_path / "nested" / "deep" / "work_units.tar.gz"
    tar_writer(units, out_path)
    assert out_path.is_file()


def test_tar_writer_produces_unit_named_files(tmp_path: Path) -> None:
    """Coordinator-side unpacker scans for `unit_*.json` — verify naming."""
    units = [_make_unit("u0001", 1), _make_unit("u0042", 42)]
    out_path = tmp_path / "work_units.tar.gz"
    tar_writer(units, out_path)

    with tarfile.open(out_path, mode="r:gz") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())
    assert names == ["unit_u0001.json", "unit_u0042.json"]


def test_tar_writer_sha256_is_stable(tmp_path: Path) -> None:
    """The same input should produce a tarball whose SHA-256 the tenant can
    cite in manifest.work_unit_source.tarball_sha256.

    Tarball headers include mtime, so consecutive writes need not be
    bit-identical, but the content of each member file is deterministic
    via Pydantic's model_dump_json.
    """
    units = [_make_unit("u0001", 1)]
    out1 = tmp_path / "a.tar.gz"
    out2 = tmp_path / "b.tar.gz"
    tar_writer(units, out1)
    tar_writer(units, out2)

    # Re-read both and compare member content
    units_a = tar_reader(out1)
    units_b = tar_reader(out2)
    assert [u.model_dump() for u in units_a] == [u.model_dump() for u in units_b]


def test_tar_reader_ignores_non_unit_files(tmp_path: Path) -> None:
    """A tarball that happens to contain unrelated files should still be readable."""
    out_path = tmp_path / "mixed.tar.gz"
    with tarfile.open(out_path, mode="w:gz") as tf:
        # Add a real unit
        unit = _make_unit("u0001", 1)
        data = unit.model_dump_json().encode("utf-8")
        info = tarfile.TarInfo(name=f"unit_{unit.unit_id}.json")
        info.size = len(data)
        from io import BytesIO

        tf.addfile(info, BytesIO(data))
        # Add a stray non-unit file
        extra = b"readme content"
        info2 = tarfile.TarInfo(name="README.txt")
        info2.size = len(extra)
        tf.addfile(info2, BytesIO(extra))

    units = tar_reader(out_path)
    assert len(units) == 1
    assert units[0].unit_id == "u0001"


def test_tarball_sha256_for_manifest_use(tmp_path: Path) -> None:
    """Demonstrates the workflow: write tar, compute sha256, that's the
    value the tenant puts in manifest.work_unit_source.tarball_sha256."""
    out_path = tmp_path / "work_units.tar.gz"
    tar_writer([_make_unit("u1", 1)], out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)
