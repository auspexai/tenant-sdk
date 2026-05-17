"""Loader for the published JSON Schema files.

The SDK ships its JSON Schemas as data files alongside the Python code so that
non-Python tenant authors (or drift-detection tests) can validate against the
canonical schemas directly. The hatch build configuration force-includes the
top-level schemas/ and cddl/ directories into the installed package; this
loader prefers the installed-package path and falls back to walking up to the
repo-root path for editable installs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PKG_PATH = Path(__file__).resolve().parent


def _schemas_dir() -> Path:
    """Locate the schemas/ directory. Works for both editable and wheel installs."""
    # Wheel install: schemas/ lives inside the package (hatch force-include)
    candidate = _PKG_PATH / "schemas"
    if (candidate / "manifest_v0_1.json").is_file():
        return candidate
    # Editable install: walk up to find repo-root schemas/
    for parent in [_PKG_PATH, *_PKG_PATH.parents]:
        candidate = parent / "schemas"
        if (candidate / "manifest_v0_1.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate auspexai_tenant schemas/. "
        "Expected either an installed wheel or an editable install with schemas/ at the repo root."
    )


def _cddl_dir() -> Path:
    """Locate the cddl/ directory. Same dual-path strategy as schemas."""
    candidate = _PKG_PATH / "cddl"
    if (candidate / "manifest_v0_1.cddl").is_file():
        return candidate
    for parent in [_PKG_PATH, *_PKG_PATH.parents]:
        candidate = parent / "cddl"
        if (candidate / "manifest_v0_1.cddl").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate auspexai_tenant cddl/")


@lru_cache(maxsize=8)
def load_schema(name: str) -> dict[str, Any]:
    """Load a published JSON Schema by filename (e.g., 'manifest_v0_1.json')."""
    return json.loads((_schemas_dir() / name).read_text())


@lru_cache(maxsize=8)
def load_cddl(name: str) -> str:
    """Load a published CDDL definition by filename (e.g., 'manifest_v0_1.cddl')."""
    return (_cddl_dir() / name).read_text()
