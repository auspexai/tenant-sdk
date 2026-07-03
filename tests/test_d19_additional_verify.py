"""D19 SDK side: the v2 schema gate + the anchor-or-fail check for
`additional_results` (unit-level; the full custody chain is covered by the
platform's route tests and the existing verify fixtures)."""

from __future__ import annotations

import pytest

from auspexai_tenant.evidence import verify_additional_results


class _Att:
    def __init__(self, units=None, diverged=None):
        self.units = units or []
        self.diverged_units = diverged or []


def test_no_section_is_none():
    assert verify_additional_results({}, _Att()) is None


def test_diverged_row_anchors_to_predicate_hashes():
    att = _Att(diverged=[{"unit_id": "u2", "result_hashes": ["h-good"]}])
    row = {
        "unit_id": "u2",
        "integrity_basis": "diverged",
        "semantic_hash": "h-good",
        "aged_off": True,  # hash-only stub: anchor still checked, signature skipped
    }
    assert verify_additional_results({"additional_results": [row]}, att) is True
    row_bad = dict(row, semantic_hash="h-forged")
    assert verify_additional_results({"additional_results": [row_bad]}, att) is False


def test_outlier_row_anchors_to_tolerance_block():
    att = _Att(
        units=[
            {
                "unit_id": "u1",
                "consensus_result_hash": "rep",
                "tolerance": {"outlier_result_hashes": ["h-out"]},
            }
        ]
    )
    row = {
        "unit_id": "u1",
        "integrity_basis": "outlier",
        "semantic_hash": "h-out",
        "aged_off": True,
    }
    assert verify_additional_results({"additional_results": [row]}, att) is True
    # Pre-forward-fix predicate (no hashes) → the row can never anchor.
    att_prefix = _Att(units=[{"unit_id": "u1", "consensus_result_hash": "rep", "tolerance": {}}])
    assert verify_additional_results({"additional_results": [row]}, att_prefix) is False


def test_unknown_basis_and_missing_attestation_fail():
    row = {"unit_id": "u1", "integrity_basis": "mystery", "aged_off": True}
    assert verify_additional_results({"additional_results": [row]}, _Att()) is False
    assert verify_additional_results({"additional_results": [row]}, None) is False


def test_observation_row_requires_its_receipt_in_bundle():
    att = _Att()
    row = {
        "unit_id": "u1",
        "integrity_basis": "observation",
        "receipt_id": "rcpt-missing",
        "aged_off": True,
    }
    assert verify_additional_results({"additional_results": [row]}, att) is False


def test_v2_schema_accepted_v3_refused():
    from auspexai_tenant.evidence import verify_bundle

    with pytest.raises(ValueError, match="unknown evidence-bundle schema"):
        verify_bundle({"schema": "auspexai-evidence-bundle/v3", "transfer": {}})
