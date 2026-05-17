"""Tests for manifest validation: Pydantic round-trip, conditional rules, drift detector."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from auspexai_tenant.cli import main
from auspexai_tenant.manifest import Manifest
from auspexai_tenant.schemas import load_schema

FIXTURES = Path(__file__).parent / "fixtures"


# --- Pydantic model validation ------------------------------------------------


def test_valid_minimal_loads() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    m = Manifest.model_validate(raw)
    assert m.tenant_id == "sentinel"
    assert m.schema_version == "0.1"
    assert m.replication_factor == 3
    assert m.sensitive_content_flags == []
    assert m.approver_attestations is None


def test_valid_sensitive_loads() -> None:
    raw = json.loads((FIXTURES / "valid_sensitive.json").read_text())
    m = Manifest.model_validate(raw)
    assert "dual_use" in m.sensitive_content_flags
    assert "red_team" in m.sensitive_content_flags
    assert m.approver_attestations is not None
    assert len(m.approver_attestations) == 1


def test_sensitive_without_attestation_fails() -> None:
    raw = json.loads((FIXTURES / "valid_sensitive.json").read_text())
    del raw["approver_attestations"]
    with pytest.raises(ValidationError, match="approver_attestations is required"):
        Manifest.model_validate(raw)


def test_invalid_missing_required_field_rejected() -> None:
    raw = json.loads((FIXTURES / "invalid_missing_field.json").read_text())
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_extra_field_rejected() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    raw["unknown_field"] = "x"
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_bad_tenant_id_pattern_rejected() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    raw["tenant_id"] = "BadCase"
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_replication_factor_zero_rejected() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    raw["replication_factor"] = 0
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_unknown_work_unit_kind_rejected() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    raw["work_unit_source"] = {"kind": "plugin", "command": ["./x"]}
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_unknown_reducer_kind_rejected() -> None:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    raw["reducer"] = {"kind": "magic"}
    with pytest.raises(ValidationError):
        Manifest.model_validate(raw)


def test_pydantic_roundtrip_preserves_data() -> None:
    raw = json.loads((FIXTURES / "valid_sensitive.json").read_text())
    m = Manifest.model_validate(raw)
    serialized = json.loads(m.model_dump_json())
    # Round-trip-validate the serialized form
    m2 = Manifest.model_validate(serialized)
    assert m2.tenant_id == m.tenant_id
    assert m2.sensitive_content_flags == m.sensitive_content_flags


# --- JSON Schema vs Pydantic drift detector -----------------------------------


def test_json_schema_accepts_valid_minimal() -> None:
    """The published JSON Schema must accept what Pydantic accepts."""
    schema = load_schema("manifest_v0_1.json")
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    jsonschema.validate(raw, schema)
    Manifest.model_validate(raw)


def test_json_schema_accepts_valid_sensitive() -> None:
    schema = load_schema("manifest_v0_1.json")
    raw = json.loads((FIXTURES / "valid_sensitive.json").read_text())
    jsonschema.validate(raw, schema)
    Manifest.model_validate(raw)


def test_json_schema_rejects_invalid_missing_field() -> None:
    """The published JSON Schema must reject what Pydantic rejects."""
    schema = load_schema("manifest_v0_1.json")
    raw = json.loads((FIXTURES / "invalid_missing_field.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)


def test_json_schema_sensitive_without_attestation_rejected() -> None:
    """Conditional rule (sensitive flags → attestations required) holds in both validators."""
    schema = load_schema("manifest_v0_1.json")
    raw = json.loads((FIXTURES / "valid_sensitive.json").read_text())
    del raw["approver_attestations"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, schema)


# --- CLI ----------------------------------------------------------------------


def test_cli_validates_minimal() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["manifest", "validate", str(FIXTURES / "valid_minimal.json")],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_cli_validates_sensitive() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["manifest", "validate", str(FIXTURES / "valid_sensitive.json")],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_cli_rejects_invalid() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["manifest", "validate", str(FIXTURES / "invalid_missing_field.json")],
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_cli_rejects_nonexistent_path() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["manifest", "validate", "/nonexistent/path.json"])
    assert result.exit_code != 0


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
