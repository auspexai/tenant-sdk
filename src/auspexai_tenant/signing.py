"""Tenant signing key + manifest signing for the AuspexAI Tenant SDK.

A tenant (researcher) holds an Ed25519 keypair used to sign manifests before
upload to the coordinator. The keypair lives at `~/.config/auspexai-tenant/
tenant_key` (PKCS8 PEM, mode 0600) by default; tenants can override the
path via `--key` on every CLI command that consumes it.

This is the only key the Tenant SDK uses — it *is* your researcher credential
(the same key `auspexai-tenant apply` registers and every command signs with).
It is unrelated to the operator's maintainer *bearer token*, which lives in a
different tool and never touches this SDK.

v0.1 limitations (revisit in Phase 2):

- **No password encryption** on the private key. Phase 2 will integrate with
  the OS keystore per the §5.10 / Q6 keystore design used on the worker side.
- **No cross-language canonical JSON** for signing. Pydantic v2's
  `model_dump_json()` is deterministic in Python 3.11+ (insertion-order-
  preserving dict serialization) but is not RFC 8785 canonical JSON. If
  non-Python tenants emerge, switch to RFC 8785 or in-toto's canonical
  encoding here.
- **No transparency-log recording.** The platform-side §5.16 Sigstore +
  Rekor flow signs *releases and receipts*, not tenant manifests. Phase 2+
  may extend Rekor recording to manifests if tenant-impersonation becomes
  a real threat. Until then, the coordinator verifies signatures against
  a published roster of tenant pubkeys.

The `ManifestSignature` envelope structure is the published wire format —
see `schemas/manifest_signature_v0_1.json` and `cddl/manifest_signature_v0_1.cddl`.
Non-Python tenants targeting these schemas can produce verifiable signature
files without using this module.
"""

from __future__ import annotations

import base64
import stat
from pathlib import Path
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
)
from pydantic import BaseModel, ConfigDict, Field

from auspexai_tenant.manifest import Manifest

DEFAULT_KEY_PATH = Path.home() / ".config" / "auspexai-tenant" / "tenant_key"


class ManifestSignature(BaseModel):
    """The .sig envelope file accompanying a signed manifest.

    Wire format published as `schemas/manifest_signature_v0_1.json` and
    `cddl/manifest_signature_v0_1.cddl`. Immutable once published.
    """

    model_config = ConfigDict(extra="forbid")

    sig_v: Literal["0.1"]
    maintainer_pubkey: Annotated[
        str,
        Field(
            pattern=r"^[a-f0-9]{64}$",
            description="Lowercase hex of the 32-byte Ed25519 public key.",
        ),
    ]
    signature: Annotated[
        str,
        Field(
            pattern=r"^[A-Za-z0-9+/=]+$",
            description=(
                "Base64-encoded Ed25519 signature over the canonical JSON encoding "
                "of the manifest (Pydantic model_dump_json output)."
            ),
        ),
    ]


class TenantKey:
    """An Ed25519 keypair held by a tenant (researcher) for signing manifests."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private = private_key

    @classmethod
    def generate(cls) -> TenantKey:
        """Generate a fresh Ed25519 keypair using the system CSPRNG."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: Path) -> TenantKey:
        """Load a tenant keypair from a PKCS8 PEM file (no password).

        Raises FileNotFoundError if the file is missing, ValueError if the
        contents are not a valid Ed25519 private key.
        """
        pem_bytes = path.read_bytes()
        loaded = load_pem_private_key(pem_bytes, password=None)
        if not isinstance(loaded, Ed25519PrivateKey):
            raise ValueError(
                f"{path}: expected an Ed25519 private key, got {type(loaded).__name__}"
            )
        return cls(loaded)

    def save(self, path: Path) -> None:
        """Save the keypair to a PKCS8 PEM file with restrictive permissions (0600).

        Creates the parent directory if missing. Overwrites the destination if
        it exists (callers are expected to gate this with their own checks).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        pem_bytes = self._private.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
        path.write_bytes(pem_bytes)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @property
    def pubkey_hex(self) -> str:
        """Lowercase hex of the 32-byte Ed25519 public key."""
        raw = self._public_bytes()
        return raw.hex()

    def _public_bytes(self) -> bytes:
        return self._private.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )

    def sign(self, data: bytes) -> bytes:
        """Sign arbitrary bytes; return the raw 64-byte Ed25519 signature."""
        return self._private.sign(data)


def _canonical_manifest_bytes(manifest: Manifest) -> bytes:
    """Return the canonical-JSON bytes of a manifest, used as the signing input.

    v0.1: Pydantic v2 `model_dump_json()` with no indent. Deterministic within
    Python 3.11+ but not RFC 8785 canonical. See module docstring.
    """
    return manifest.model_dump_json().encode("utf-8")


def sign_manifest(manifest: Manifest, key: TenantKey) -> ManifestSignature:
    """Sign a manifest with the tenant's keypair.

    Returns a ManifestSignature envelope ready to write to disk alongside the
    manifest. The signing input is the canonical JSON bytes of the manifest
    (see `_canonical_manifest_bytes`).
    """
    sig_bytes = key.sign(_canonical_manifest_bytes(manifest))
    return ManifestSignature(
        sig_v="0.1",
        maintainer_pubkey=key.pubkey_hex,
        signature=base64.b64encode(sig_bytes).decode("ascii"),
    )


def verify_manifest(manifest: Manifest, signature: ManifestSignature) -> bool:
    """Verify that `signature` is a valid signature of `manifest`.

    Returns True if the signature verifies under the embedded tenant pubkey
    (the `maintainer_pubkey` wire field — a frozen schema name, not a claim
    about who holds the key), False if the signature is invalid. Callers must
    additionally check that the embedded pubkey is one they trust (e.g., on the
    coordinator's published roster).
    """
    pubkey_bytes = bytes.fromhex(signature.maintainer_pubkey)
    pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
    try:
        sig_bytes = base64.b64decode(signature.signature, validate=True)
    except (ValueError, base64.binascii.Error):  # type: ignore[attr-defined]
        return False
    try:
        pubkey.verify(sig_bytes, _canonical_manifest_bytes(manifest))
        return True
    except InvalidSignature:
        return False


def canonical_deviation_bytes(manifest_hash: str, what_changed: str, why: str) -> bytes:
    """D16.2-D (§5): the canonical deviation declaration the tenant signs —
    the ORIGINAL design's manifest hash + what changed + why. KEEP IN LOCKSTEP
    with the coordinator's `pre_registration.canonical_deviation_bytes` (the
    manifest-signing convention applied to the deviation: sorted keys, compact
    separators, utf-8)."""
    import json as _json

    return _json.dumps(
        {"manifest_hash": manifest_hash, "what_changed": what_changed, "why": why},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
