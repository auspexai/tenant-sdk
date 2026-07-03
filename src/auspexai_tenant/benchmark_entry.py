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


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    """Canonical JSON (sorted keys, tight separators). The signature covers
    these exact BYTES, and every verifier — SDK and the board page's
    in-browser check — verifies and renders from them: the entry envelope
    carries the bytes base64'd, which sidesteps every cross-language
    canonical-JSON trap (float formatting, unicode escaping, key order)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _attestation_anchor(bundle: dict[str, Any]) -> dict[str, Any]:
    att = bundle.get("attestation") or {}
    return {
        "merkle_root": att.get("merkle_root"),
        "algorithm": att.get("algorithm"),
        "rekor_log_index": att.get("rekor_log_index"),
        "rekor_entry_uuid": att.get("rekor_entry_uuid"),
    }


def derive_config_delta(obs: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """The declared differences between two signed manifests, on the dimensions
    the platform defines. Structured, never prose: the board renders chips from
    this, so the explanation of a score is a machine fact a verifier can
    recompute from the two manifests the entry anchors."""

    def _model(m: dict[str, Any]) -> str | None:
        models = m.get("models") or []
        return f"{models[0].get('id')}" if models else None

    def _generation(m: dict[str, Any]) -> str:
        d = m.get("inference_determinism")
        if not d:
            return "greedy"
        t = d.get("temperature")
        if not t:
            return "greedy"
        knobs = {k: d.get(k) for k in ("top_p", "top_k", "min_p") if d.get(k) is not None}
        knob_s = "".join(f",{k}={v}" for k, v in sorted(knobs.items()))
        return f"sampling(temp={t}{knob_s},seeded)"

    delta: dict[str, Any] = {"changed": {}, "unchanged": []}
    dims = {
        "model": _model,
        "generation": _generation,
        "reducer": lambda m: (m.get("reducer") or {}).get("kind"),
        "feature_schema": lambda m: "declared" if m.get("feature_schema") else None,
    }
    for name, fn in dims.items():
        a, b = fn(obs), fn(ref)
        if name == "feature_schema":
            same = (obs.get("feature_schema") or {}) == (ref.get("feature_schema") or {})
            if same:
                delta["unchanged"].append(name)
            else:
                delta["changed"][name] = {"from": "reference's", "to": "observation's"}
            continue
        if a == b:
            delta["unchanged"].append(name)
        else:
            delta["changed"][name] = {"from": b, "to": a}
    return delta


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
    payload: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "published_at": datetime.now(UTC).isoformat(),
        "publisher_pubkey_hex": key.pubkey_hex,
        "tenant_id": tenant_id,
        # The MACHINE-DERIVED config delta (no free text anywhere on the board):
        # what the observation's signed manifest declares differently from the
        # reference's. This IS the meaning of the score — the benchmark measures
        # whatever the declared delta varied; "no declared change" = a temporal
        # control. Derived, signed, tamper-evident.
        "config_delta": derive_config_delta(
            (observation_bundle.get("manifest") or {}),
            (reference_bundle.get("manifest") or {}),
        ),
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
    body = _canonical_payload_bytes(payload)
    return {
        "schema": ENTRY_SCHEMA,
        "payload_b64": b64encode(body).decode(),
        "publisher_pubkey_hex": key.pubkey_hex,
        "signature_b64": b64encode(key.sign(body)).decode(),
    }


def verify_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The verified PAYLOAD (parsed from the exact signed bytes), or None.
    The envelope pubkey must equal the pubkey INSIDE the signed payload (the
    key identity is part of the claim — no swap possible). Grounding that
    pubkey to a real tenant is the CURATOR's job at registry-inclusion time.
    Callers render/consume ONLY the returned payload, never envelope mirrors."""
    try:
        body = b64decode(entry["payload_b64"])
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(entry["publisher_pubkey_hex"]))
        pub.verify(b64decode(entry["signature_b64"]), body)
        payload = json.loads(body)
        if payload.get("publisher_pubkey_hex") != entry["publisher_pubkey_hex"]:
            return None
        if payload.get("schema") != ENTRY_SCHEMA:
            return None
        return payload
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return None
