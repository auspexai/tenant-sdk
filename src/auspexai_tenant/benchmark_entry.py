"""Signed Drift-Benchmark registry entries (G5, drift_benchmark_design.md §6).

Publishing is a DELIBERATE act: `benchmark publish` wraps a scored report +
both experiments' attestation anchors into a tenant-key-signed entry — the
researcher's own claim, verifiable by anyone. The public board renders a
curated registry of these entries; every cell traces to signed evidence
(experiment ids, attestation Merkle roots, Rekor inclusion), which is the
board's differentiator: don't trust the chart — verify the cell.

Naming (G2 prior-art pass, 2026-07-03): "Drift Benchmark" — always two words,
brand-qualified in public copy; never the compressed "DriftBench" (a crowded
name: ≥4 unrelated projects).
"""

from __future__ import annotations

import json
from base64 import b64decode, b64encode
from datetime import UTC, datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ENTRY_SCHEMA = "auspexai-benchmark-entry/v0"


def _canonical_entry_bytes(entry: dict[str, Any]) -> bytes:
    """The signed body: the entry WITHOUT the signature itself (the publisher
    pubkey IS signed — the key identity is part of the claim), canonical
    JSON (sorted keys, tight separators) — the same convention every other
    signed surface in the platform uses."""
    body = {k: v for k, v in entry.items() if k != "signature_b64"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _attestation_anchor(bundle: dict[str, Any]) -> dict[str, Any]:
    att = bundle.get("attestation") or {}
    return {
        "merkle_root": att.get("merkle_root"),
        "algorithm": att.get("algorithm"),
        "rekor_log_index": att.get("rekor_log_index"),
        "rekor_entry_uuid": att.get("rekor_entry_uuid"),
    }


def build_entry(
    *,
    record: dict[str, Any],
    observation_bundle: dict[str, Any],
    reference_bundle: dict[str, Any],
    tenant_id: str | None,
    key,  # TenantKey
) -> dict[str, Any]:
    """A signed registry entry from a saved benchmark record + the two
    custody-verified bundles it was scored over. The caller has ALREADY
    verified both bundles (the CLI does) — this function only assembles
    and signs the claim."""
    rep = record.get("report") or {}
    entry: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "published_at": datetime.now(UTC).isoformat(),
        "tenant_id": tenant_id,
        "observation": {
            **(record.get("observation") or {}),
            "manifest_hash": observation_bundle.get("manifest_hash"),
            "attestation": _attestation_anchor(observation_bundle),
        },
        "reference": {
            **(record.get("reference") or {}),
            "manifest_hash": reference_bundle.get("manifest_hash"),
            "attestation": _attestation_anchor(reference_bundle),
        },
        "report": {
            "peak_eu": rep.get("peak_eu"),
            "breadth": rep.get("breadth"),
            "byte_divergence_rate": rep.get("byte_divergence_rate"),
            "diverged_units_total": rep.get("diverged_units_total"),
            "key_feature": rep.get("key_feature"),
            "computed_at": record.get("computed_at"),
            "probes": [
                {
                    "key": p.get("key"),
                    "peak_eu": p.get("peak_eu"),
                    "beyond_envelope": p.get("beyond_envelope"),
                    "byte_divergence_rate": p.get("byte_divergence_rate"),
                    "observations": p.get("observations"),
                    "reference_observations": p.get("reference_observations"),
                }
                for p in rep.get("probes") or []
            ],
        },
    }
    entry["publisher_pubkey_hex"] = key.pubkey_hex
    entry["signature_b64"] = b64encode(key.sign(_canonical_entry_bytes(entry))).decode()
    return entry


def verify_entry(entry: dict[str, Any]) -> bool:
    """True iff the entry's Ed25519 signature verifies over its canonical body
    with the embedded publisher pubkey. (Grounding that pubkey to a real
    tenant is the CURATOR's job at registry-inclusion time.)"""
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(entry["publisher_pubkey_hex"]))
        pub.verify(b64decode(entry["signature_b64"]), _canonical_entry_bytes(entry))
        return True
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return False
