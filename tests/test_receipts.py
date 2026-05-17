"""Tests for receipt CBOR encode/decode + `receipts show` CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from auspexai_tenant.cli import main
from auspexai_tenant.receipts import (
    QuorumAgreement,
    Receipt,
    ResultHashAnchor,
    TimeWindow,
    decode_cbor,
    encode_cbor,
)


def _make_receipt() -> Receipt:
    return Receipt(
        version="0.1",
        tenant_id="synth-doubler",
        experiment_id="synth-doubler-v1",
        worker_pubkey=bytes(range(32)),
        work_unit_ids=["u0001", "u0002"],
        time_window=TimeWindow(
            start=datetime(2026, 5, 17, 20, 0, 0, tzinfo=UTC),
            end=datetime(2026, 5, 17, 21, 0, 0, tzinfo=UTC),
        ),
        quorum_agreement=QuorumAgreement(
            replication_factor=3,
            agreeing_workers=3,
            method="builtin_hash_agreement",
        ),
        result_hash_anchors=[
            ResultHashAnchor(
                rekor_log_index=42,
                rekor_entry_uuid="abc-def-ghi",
                result_sha256="a" * 64,
            )
        ],
    )


# ---- Pydantic model ---------------------------------------------------------


def test_receipt_construction_round_trip() -> None:
    r = _make_receipt()
    serialized = json.loads(r.model_dump_json())
    assert serialized["worker_pubkey"] == bytes(range(32)).hex()
    r2 = Receipt.model_validate(serialized)
    assert r2.worker_pubkey == r.worker_pubkey


def test_receipt_accepts_hex_pubkey_from_json() -> None:
    r = _make_receipt()
    d = json.loads(r.model_dump_json())
    # Round-trip through JSON: pubkey is hex, validator decodes
    assert isinstance(d["worker_pubkey"], str)
    r2 = Receipt.model_validate(d)
    assert isinstance(r2.worker_pubkey, bytes)
    assert len(r2.worker_pubkey) == 32


def test_receipt_rejects_wrong_pubkey_size() -> None:
    with pytest.raises(ValidationError):
        Receipt(
            version="0.1",
            tenant_id="t",
            experiment_id="e",
            worker_pubkey=bytes(16),  # wrong size
            work_unit_ids=["u1"],
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC)),
            quorum_agreement=QuorumAgreement(
                replication_factor=1, agreeing_workers=1, method="builtin"
            ),
            result_hash_anchors=[
                ResultHashAnchor(rekor_log_index=0, rekor_entry_uuid="x", result_sha256="a" * 64)
            ],
        )


def test_receipt_rejects_empty_work_unit_ids() -> None:
    with pytest.raises(ValidationError):
        Receipt(
            version="0.1",
            tenant_id="t",
            experiment_id="e",
            worker_pubkey=bytes(32),
            work_unit_ids=[],
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC)),
            quorum_agreement=QuorumAgreement(
                replication_factor=1, agreeing_workers=1, method="builtin"
            ),
            result_hash_anchors=[
                ResultHashAnchor(rekor_log_index=0, rekor_entry_uuid="x", result_sha256="a" * 64)
            ],
        )


def test_receipt_rejects_zero_replication_factor() -> None:
    with pytest.raises(ValidationError):
        QuorumAgreement(replication_factor=0, agreeing_workers=0, method="x")


def test_receipt_rejects_bad_sha256() -> None:
    with pytest.raises(ValidationError):
        ResultHashAnchor(rekor_log_index=0, rekor_entry_uuid="x", result_sha256="zzz")


def test_receipt_rejects_extra_fields() -> None:
    r = _make_receipt()
    d = json.loads(r.model_dump_json())
    d["extra"] = "field"
    with pytest.raises(ValidationError):
        Receipt.model_validate(d)


# ---- CBOR encode / decode ---------------------------------------------------


def test_cbor_round_trip() -> None:
    r = _make_receipt()
    encoded = encode_cbor(r)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    decoded = decode_cbor(encoded)
    assert decoded.tenant_id == r.tenant_id
    assert decoded.worker_pubkey == r.worker_pubkey
    assert decoded.time_window.start == r.time_window.start
    assert decoded.quorum_agreement.method == r.quorum_agreement.method
    assert decoded.result_hash_anchors[0].rekor_log_index == 42


def test_cbor_encoding_keeps_pubkey_as_bytes() -> None:
    """The CBOR form uses raw bytes for worker_pubkey (per receipt_v0_1.cddl),
    not the hex string used in the JSON form."""
    import cbor2

    r = _make_receipt()
    encoded = encode_cbor(r)
    raw = cbor2.loads(encoded)
    assert isinstance(raw["worker_pubkey"], bytes)
    assert len(raw["worker_pubkey"]) == 32


def test_cbor_decode_rejects_garbage() -> None:
    import cbor2

    with pytest.raises((cbor2.CBORDecodeError, ValidationError)):
        decode_cbor(b"\xff\xff\xff\xff")


def test_cbor_decode_rejects_wrong_shape() -> None:
    import cbor2

    encoded = cbor2.dumps({"not_a_receipt": True})
    with pytest.raises(ValidationError):
        decode_cbor(encoded)


# ---- CLI: receipts show -----------------------------------------------------


def test_cli_receipts_show_happy_path(tmp_path: Path) -> None:
    runner = CliRunner()
    receipt_path = tmp_path / "receipt.cbor"
    receipt_path.write_bytes(encode_cbor(_make_receipt()))

    result = runner.invoke(main, ["receipts", "show", str(receipt_path)])
    assert result.exit_code == 0, result.output

    # Pretty-printed JSON includes hex pubkey (not bytes) and our test data
    parsed = json.loads(result.output)
    assert parsed["tenant_id"] == "synth-doubler"
    assert parsed["worker_pubkey"] == bytes(range(32)).hex()
    assert parsed["quorum_agreement"]["agreeing_workers"] == 3


def test_cli_receipts_show_rejects_invalid_cbor(tmp_path: Path) -> None:
    runner = CliRunner()
    bad_path = tmp_path / "bad.cbor"
    bad_path.write_bytes(b"\x00\x01\x02\x03")

    result = runner.invoke(main, ["receipts", "show", str(bad_path)])
    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_cli_receipts_show_rejects_wrong_schema(tmp_path: Path) -> None:
    import cbor2

    runner = CliRunner()
    wrong_path = tmp_path / "wrong.cbor"
    wrong_path.write_bytes(cbor2.dumps({"not_a_receipt": True}))

    result = runner.invoke(main, ["receipts", "show", str(wrong_path)])
    assert result.exit_code == 1
    assert "validation" in result.output.lower()
