"""Unit tests for `parse_duration` — the `--duration-cap` value parser (D5).

Covers each accepted format (unit strings, compound units, HH:MM:SS / MM:SS clock
form, bare seconds) and the click-friendly error on malformed input.
"""

from __future__ import annotations

import click
import pytest

from auspexai_tenant.cli import parse_duration


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4h", 4 * 3600.0),
        ("90m", 90 * 60.0),
        ("30s", 30.0),
        ("1h30m", 3600.0 + 30 * 60.0),
        ("2h15m30s", 2 * 3600.0 + 15 * 60.0 + 30.0),
        ("01:30:00", 3600.0 + 30 * 60.0),  # HH:MM:SS
        ("1:30:00", 3600.0 + 30 * 60.0),
        ("90:00", 90 * 60.0),  # MM:SS
        ("3600", 3600.0),  # bare seconds
        ("1.5", 1.5),  # fractional seconds
        ("0", 0.0),
        ("  4h  ", 4 * 3600.0),  # whitespace tolerated
        ("4H", 4 * 3600.0),  # case-insensitive units
    ],
)
def test_parse_duration_formats(text: str, expected: float) -> None:
    assert parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "  ",
        "abc",
        "4x",  # unknown unit
        "4h30",  # trailing junk after a unit
        "30m4h",  # units out of order
        "4h4h",  # duplicate unit
        "-5m",  # negative unit
        "-30",  # negative bare seconds
        "1:2:3:4",  # too many clock parts
        "1::3",  # empty clock part
        "h",  # unit with no number
    ],
)
def test_parse_duration_rejects_bad_input(bad: str) -> None:
    with pytest.raises(click.BadParameter):
        parse_duration(bad)
