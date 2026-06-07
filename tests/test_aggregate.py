"""Tests for the incremental result aggregators (M8 §3.3).

Covers the `RunningAggregate` contract for each reference aggregator: folding,
checkpoint/restore round-trips (journal durability), commutativity (order-
independence), and finalize() output — plus edge handling for Histogram.
"""

from __future__ import annotations

import pytest

from auspexai_tenant.aggregate import Counter, Histogram, Mean, RunningAggregate


def _r(value: float, **extra: object) -> dict:
    """A consensus result item carrying `value` in its payload."""
    return {"unit_id": f"u{value}", "payload": {"value": value, **extra}}


VAL = lambda r: r["payload"]["value"]  # noqa: E731 - terse extractor for tests


# ---- protocol conformance --------------------------------------------------


@pytest.mark.parametrize(
    "agg",
    [Counter(), Mean(VAL), Histogram([0.0, 1.0, 2.0], VAL)],
)
def test_reference_aggregators_satisfy_protocol(agg: object) -> None:
    assert isinstance(agg, RunningAggregate)


# ---- Counter ---------------------------------------------------------------


def test_counter_total() -> None:
    c = Counter()
    for v in (1, 2, 3):
        c.fold(_r(v))
    assert c.total == 3
    assert c.finalize() == {"total": 3}


def test_counter_bucketed() -> None:
    c = Counter(bucket=lambda r: "even" if int(r["payload"]["value"]) % 2 == 0 else "odd")
    for v in (1, 2, 3, 4):
        c.fold(_r(v))
    assert c.finalize() == {"total": 4, "counts": {"odd": 2, "even": 2}}


def test_counter_checkpoint_restore() -> None:
    c = Counter(bucket=lambda r: r["payload"]["value"])
    c.fold(_r(1))
    c.fold(_r(1))
    snap = c.checkpoint()

    restored = Counter(bucket=lambda r: r["payload"]["value"])
    restored.restore(snap)
    assert restored.finalize() == c.finalize()
    restored.fold(_r(1))
    assert restored.total == 3  # continues from the checkpoint


# ---- Mean ------------------------------------------------------------------


def test_mean_and_variance() -> None:
    m = Mean(VAL)
    for v in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        m.fold(_r(v))
    out = m.finalize()
    assert out["count"] == 8
    assert out["mean"] == pytest.approx(5.0)
    assert out["variance"] == pytest.approx(4.0)  # population variance
    assert out["std"] == pytest.approx(2.0)


def test_mean_empty_is_zero() -> None:
    assert Mean(VAL).finalize() == {"count": 0, "mean": 0.0, "variance": 0.0, "std": 0.0}


def test_mean_commutative() -> None:
    forward, backward = Mean(VAL), Mean(VAL)
    vals = [1.0, 2.0, 3.0, 10.0]
    for v in vals:
        forward.fold(_r(v))
    for v in reversed(vals):
        backward.fold(_r(v))
    assert forward.finalize() == pytest.approx(backward.finalize())


def test_mean_checkpoint_restore_resumes() -> None:
    m = Mean(VAL)
    m.fold(_r(2.0))
    m.fold(_r(4.0))
    resumed = Mean(VAL)
    resumed.restore(m.checkpoint())
    resumed.fold(_r(6.0))  # one more after resume
    assert resumed.mean == pytest.approx(4.0)  # mean of 2,4,6
    assert resumed.count == 3


# ---- Histogram -------------------------------------------------------------


def test_histogram_bins_half_open() -> None:
    h = Histogram([0.0, 1.0, 2.0, 3.0], VAL)
    for v in (0.0, 0.5, 1.0, 2.999):  # bin0, bin0, bin1, bin2
        h.fold(_r(v))
    out = h.finalize()
    assert out["counts"] == [2, 1, 1]
    assert out["underflow"] == 0
    assert out["overflow"] == 0
    assert out["total"] == 4


def test_histogram_underflow_overflow() -> None:
    h = Histogram([0.0, 1.0, 2.0], VAL)
    h.fold(_r(-0.1))  # underflow
    h.fold(_r(2.0))  # overflow (>= last edge)
    h.fold(_r(5.0))  # overflow
    out = h.finalize()
    assert out["underflow"] == 1
    assert out["overflow"] == 2
    assert out["counts"] == [0, 0]
    assert out["total"] == 3


def test_histogram_commutative() -> None:
    edges = [0.0, 1.0, 2.0, 3.0]
    a, b = Histogram(edges, VAL), Histogram(edges, VAL)
    vals = [0.2, 1.5, 2.5, 0.9, -1.0, 4.0]
    for v in vals:
        a.fold(_r(v))
    for v in reversed(vals):
        b.fold(_r(v))
    assert a.finalize() == b.finalize()


def test_histogram_checkpoint_restore() -> None:
    edges = [0.0, 1.0, 2.0]
    h = Histogram(edges, VAL)
    h.fold(_r(0.5))
    h.fold(_r(1.5))
    resumed = Histogram(edges, VAL)
    resumed.restore(h.checkpoint())
    assert resumed.finalize() == h.finalize()


def test_histogram_restore_bin_mismatch_raises() -> None:
    h = Histogram([0.0, 1.0, 2.0], VAL)  # 2 bins
    with pytest.raises(ValueError, match="bins"):
        h.restore({"counts": [0, 0, 0], "underflow": 0, "overflow": 0})


def test_histogram_rejects_bad_edges() -> None:
    with pytest.raises(ValueError, match="ascending"):
        Histogram([0.0, 2.0, 1.0], VAL)
    with pytest.raises(ValueError, match="at least 2 edges"):
        Histogram([0.0], VAL)
