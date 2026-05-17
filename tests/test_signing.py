"""Tests for the manifest signing toolchain.

Covers: MaintainerKey generate/save/load round-trip, signature creation and
verification, tamper detection, schema cross-check, and the
`key generate / key pubkey / manifest sign` CLI commands.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import jsonschema
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from auspexai_tenant.cli import main
from auspexai_tenant.manifest import Manifest
from auspexai_tenant.schemas import load_schema
from auspexai_tenant.signing import (
    MaintainerKey,
    ManifestSignature,
    sign_manifest,
    verify_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_minimal() -> Manifest:
    raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    return Manifest.model_validate(raw)


# ---- MaintainerKey -----------------------------------------------------------


def test_generate_produces_valid_keypair() -> None:
    k = MaintainerKey.generate()
    assert len(k.pubkey_hex) == 64
    assert all(c in "0123456789abcdef" for c in k.pubkey_hex)


def test_save_load_round_trip(tmp_path: Path) -> None:
    original = MaintainerKey.generate()
    path = tmp_path / "key.pem"
    original.save(path)
    loaded = MaintainerKey.load(path)
    assert loaded.pubkey_hex == original.pubkey_hex


def test_save_uses_restrictive_permissions(tmp_path: Path) -> None:
    k = MaintainerKey.generate()
    path = tmp_path / "key.pem"
    k.save(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600 permissions, got {oct(mode)}"


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    k = MaintainerKey.generate()
    path = tmp_path / "nested" / "deep" / "key.pem"
    k.save(path)
    assert path.is_file()


def test_load_rejects_non_ed25519_key(tmp_path: Path) -> None:
    # Write a non-PEM file
    path = tmp_path / "bogus.pem"
    path.write_text("not actually a PEM key")
    with pytest.raises(ValueError, match=r"(?i)pem|encod"):
        MaintainerKey.load(path)


def test_load_rejects_wrong_key_type(tmp_path: Path) -> None:
    # Write an RSA key (different algorithm)
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
    )

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    path = tmp_path / "rsa.pem"
    path.write_bytes(pem)
    with pytest.raises(ValueError, match="Ed25519"):
        MaintainerKey.load(path)


# ---- sign_manifest / verify_manifest -----------------------------------------


def test_sign_and_verify_round_trip() -> None:
    m = _load_minimal()
    k = MaintainerKey.generate()
    sig = sign_manifest(m, k)
    assert verify_manifest(m, sig) is True


def test_verify_rejects_tampered_manifest() -> None:
    m = _load_minimal()
    k = MaintainerKey.generate()
    sig = sign_manifest(m, k)

    tampered_raw = json.loads((FIXTURES / "valid_minimal.json").read_text())
    tampered_raw["replication_factor"] = 5
    tampered = Manifest.model_validate(tampered_raw)

    assert verify_manifest(tampered, sig) is False


def test_verify_rejects_wrong_pubkey() -> None:
    m = _load_minimal()
    signer = MaintainerKey.generate()
    attacker = MaintainerKey.generate()
    sig = sign_manifest(m, signer)

    forged = ManifestSignature(
        sig_v="0.1",
        maintainer_pubkey=attacker.pubkey_hex,
        signature=sig.signature,
    )
    assert verify_manifest(m, forged) is False


def test_verify_rejects_malformed_base64_signature() -> None:
    m = _load_minimal()
    k = MaintainerKey.generate()
    sig = ManifestSignature(
        sig_v="0.1",
        maintainer_pubkey=k.pubkey_hex,
        signature="AAA",
    )
    assert verify_manifest(m, sig) is False


def test_signature_validates_against_published_schema() -> None:
    m = _load_minimal()
    k = MaintainerKey.generate()
    sig = sign_manifest(m, k)
    schema = load_schema("manifest_signature_v0_1.json")
    jsonschema.validate(json.loads(sig.model_dump_json()), schema)


def test_signature_pydantic_rejects_bad_pubkey() -> None:
    with pytest.raises(ValidationError):
        ManifestSignature(
            sig_v="0.1",
            maintainer_pubkey="ZZZ",  # not lowercase hex 64 chars
            signature="AAAA",
        )


# ---- CLI: key generate ------------------------------------------------------


def test_cli_key_generate_creates_keypair(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    result = runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    assert result.exit_code == 0, result.output
    assert "Public key:" in result.output
    assert key_path.exists()
    # Load it to verify it's actually a valid keypair
    MaintainerKey.load(key_path)


def test_cli_key_generate_refuses_to_overwrite(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    result = runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_cli_key_generate_overwrites_with_force(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    r1 = runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    assert r1.exit_code == 0
    r2 = runner.invoke(main, ["key", "generate", "--output", str(key_path), "--force"])
    assert r2.exit_code == 0


def test_cli_key_pubkey(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    result = runner.invoke(main, ["key", "pubkey", "--key", str(key_path)])
    assert result.exit_code == 0, result.output
    # Output is the hex pubkey (64 chars + newline)
    assert len(result.output.strip()) == 64


# ---- CLI: manifest sign -----------------------------------------------------


def test_cli_manifest_sign_produces_sig_file(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    result = runner.invoke(main, ["manifest", "sign", str(manifest_path), "--key", str(key_path)])
    assert result.exit_code == 0, result.output

    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    assert sig_path.is_file()
    sig_raw = json.loads(sig_path.read_text())
    sig = ManifestSignature.model_validate(sig_raw)
    assert sig.sig_v == "0.1"


def test_cli_manifest_sign_custom_output(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())
    sig_path = tmp_path / "custom.sig"

    result = runner.invoke(
        main,
        [
            "manifest",
            "sign",
            str(manifest_path),
            "--key",
            str(key_path),
            "--output",
            str(sig_path),
        ],
    )
    assert result.exit_code == 0
    assert sig_path.is_file()


def test_cli_manifest_sign_rejects_invalid_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps({"not_a_manifest": True}))

    result = runner.invoke(main, ["manifest", "sign", str(bad_manifest), "--key", str(key_path)])
    assert result.exit_code == 1
    assert "validation" in result.output.lower()


def test_cli_manifest_sign_end_to_end_with_verify(tmp_path: Path) -> None:
    """Sign via CLI, then verify via the library API end-to-end."""
    runner = CliRunner()
    key_path = tmp_path / "key.pem"
    runner.invoke(main, ["key", "generate", "--output", str(key_path)])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text((FIXTURES / "valid_minimal.json").read_text())

    runner.invoke(main, ["manifest", "sign", str(manifest_path), "--key", str(key_path)])
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")

    m = Manifest.model_validate(json.loads(manifest_path.read_text()))
    sig = ManifestSignature.model_validate(json.loads(sig_path.read_text()))
    assert verify_manifest(m, sig) is True
