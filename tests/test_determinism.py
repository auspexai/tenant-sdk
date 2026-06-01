"""Tests for the §9 #33 determinism-contract helpers.

The load-bearing test is `test_sub_quantum_noise_agrees`: it simulates two honest
workers whose raw floats differ by hardware noise below the quantum, and proves
their canonicalized payloads serialize to byte-identical canonical JSON — i.e.
they would pass the coordinator's exact-hash `hash_agreement` without any
coordinator change. That is the whole claim of the determinism contract.
"""

from __future__ import annotations

import json

import pytest

from auspexai_tenant.determinism import canonical_quantize, canonicalize_floats


def _canonical_json(payload: dict) -> str:
    """Mirror the coordinator's canonical serialization (semantic_hash input)."""
    return json.dumps(
        {"exit_code": 0, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def test_quantizes_to_places() -> None:
    assert canonical_quantize(0.1234567, places=6) == 0.123457
    assert canonical_quantize(2.0, places=6) == 2.0
    assert canonical_quantize(1.0 / 3.0, places=4) == 0.3333


def test_sub_quantum_noise_agrees() -> None:
    # Two honest workers compute the "same" metric; results differ at ~1e-9,
    # far below a 1e-6 quantum (the realistic GPU/BLAS-noise regime).
    worker_a = 0.305128_000_001
    worker_b = 0.305127_999_998
    assert worker_a != worker_b  # raw floats differ -> would break exact hash

    qa = canonical_quantize(worker_a, places=6)
    qb = canonical_quantize(worker_b, places=6)
    assert qa == qb  # collapse to the same canonical value

    # ...and the canonical JSON the coordinator would hash is byte-identical.
    pa = canonicalize_floats({"metric": worker_a, "n_samples": 2000, "seed": 7})
    pb = canonicalize_floats({"metric": worker_b, "n_samples": 2000, "seed": 7})
    assert _canonical_json(pa) == _canonical_json(pb)


def test_negative_zero_normalized() -> None:
    assert canonical_quantize(-1e-9, places=6) == 0.0
    # JSON must not carry a "-0.0" that would disagree with "0.0".
    assert json.dumps(canonical_quantize(-1e-9, places=6)) == "0.0"


def test_non_finite_rejected() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_quantize(bad)


def test_bool_not_quantized() -> None:
    # bool is an int subclass but is not a measurement.
    out = canonicalize_floats({"ok": True, "n": 3, "x": 0.123456789})
    assert out["ok"] is True
    assert out["n"] == 3
    assert out["x"] == 0.123457


def test_nested_structure_quantized() -> None:
    raw = {"a": [0.1111119, 0.2222221], "b": {"c": 0.3333339}, "label": "geom"}
    out = canonicalize_floats(raw, places=5)
    assert out == {"a": [0.11111, 0.22222], "b": {"c": 0.33333}, "label": "geom"}


def test_rejects_non_number_scalar() -> None:
    with pytest.raises(TypeError):
        canonical_quantize("0.3")  # type: ignore[arg-type]
