"""Determinism contract helpers (§9 #33).

The platform's per-unit trust model is **exact-hash agreement**: N replicas of a
work unit agree iff their ``{exit_code, payload}`` serialize to byte-identical
canonical JSON (see the coordinator's ``semantic_hash``). That works for integer
/ exact outputs, but two *honest, correct* workers computing a floating-point
metric on different hardware will differ in the last bits (GPU/BLAS/reduction
order / fp accumulation) and spuriously "disagree".

The Phase-1 bridge (ratified 2026-06-01, ``genericity_pass_ratification.md``) is a
**determinism contract**, not a change to what "agreement" means:

1. The tenant pins a PRNG **seed** in the opaque ``work_unit.payload`` — all N
   replicas of a unit get the identical payload, hence the identical seed, hence
   reproducible draws.
2. The executor obeys "**same seed -> same bytes**": its output is a deterministic
   function of the payload.
3. Float outputs are **quantized to a fixed precision** before they go in the
   result payload, so honest hardware noise below the quantum collapses to an
   identical canonical value — and the existing exact-hash reducer just works,
   with no coordinator change.

This module is step 3. ``canonical_quantize`` maps a float to a bit-stable value;
``canonicalize_floats`` applies it across a result payload.

**Quantum-vs-noise rule (load-bearing):** choose ``places`` so the quantum
(``10**-places``) is *much larger* than the honest float noise of the
computation. The residual risk is **quantization-boundary straddle** (two honest
values landing on opposite sides of a rounding boundary). It cannot be eliminated
by quantization alone — a tenant whose scientific signal lives *below* any noise
floor it can tolerate is the genuine trigger for tolerance/statistical consensus
(§9 #31, Phase 2). For robust, bounded geometric/score metrics, a quantum a few
orders of magnitude above the noise makes straddle vanishingly rare.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["DEFAULT_PLACES", "canonical_quantize", "canonicalize_floats"]

DEFAULT_PLACES = 6


def canonical_quantize(x: float, *, places: int = DEFAULT_PLACES) -> float:
    """Quantize ``x`` to ``places`` decimal places, bit-stable across machines.

    Returns a ``float`` whose canonical JSON serialization is identical for any
    two inputs that agree to within the quantum (``10**-places``). Normalizes
    ``-0.0`` to ``0.0`` so sign-of-zero noise doesn't break agreement.

    Raises ``ValueError`` on non-finite input (NaN / +-inf cannot be canonically
    serialized to JSON and must never enter a hashed payload).

    The returned value relies on IEEE-754 double semantics + Python's
    platform-independent shortest-``repr`` (the same one ``json.dumps`` uses), so
    ``json.dumps(canonical_quantize(a)) == json.dumps(canonical_quantize(b))``
    whenever ``a`` and ``b`` round to the same quantum.
    """
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError(f"canonical_quantize expects a real number, got {type(x).__name__}")
    if not math.isfinite(x):
        raise ValueError(f"cannot canonically quantize non-finite value {x!r}")
    if places < 0:
        raise ValueError(f"places must be >= 0, got {places}")
    q = round(float(x), places)
    # Collapse -0.0 (and any negative zero produced by rounding) to 0.0 so two
    # honest workers straddling zero don't disagree on the sign bit alone.
    if q == 0.0:
        return 0.0
    return q


def canonicalize_floats(obj: Any, *, places: int = DEFAULT_PLACES) -> Any:
    """Recursively quantize every ``float`` in a JSON-shaped structure.

    Walks dicts and lists, quantizing each ``float`` via ``canonical_quantize``
    and leaving ``int`` / ``bool`` / ``str`` / ``None`` untouched. Use this on an
    executor's output payload so the whole result is bit-stable before
    submission::

        return canonicalize_floats({"metric": raw_metric, "n_samples": n})

    ``bool`` is deliberately preserved (Python ``bool`` is an ``int`` subclass,
    but it is not a measurement and must not be quantized to ``1.0``/``0.0``).
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return canonical_quantize(obj, places=places)
    if isinstance(obj, dict):
        return {k: canonicalize_floats(v, places=places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [canonicalize_floats(v, places=places) for v in obj]
    return obj
