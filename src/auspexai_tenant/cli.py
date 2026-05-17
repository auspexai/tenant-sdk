"""auspexai-tenant CLI entrypoint.

v0.1: ships `manifest validate` only. Future commands (next sessions):
- manifest sign / upload
- workunits tar-writer helper
- receipts show
- executor harness scaffolding
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from pydantic import ValidationError

from auspexai_tenant import __version__
from auspexai_tenant.manifest import Manifest


@click.group()
@click.version_option(version=__version__, prog_name="auspexai-tenant")
def main() -> None:
    """AuspexAI Tenant SDK CLI."""


@main.group()
def manifest() -> None:
    """Manifest commands."""


@manifest.command("validate")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def manifest_validate(path: Path) -> None:
    """Validate a manifest JSON file against the published v0.1 schema."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        click.echo(f"ERROR: {path} is not valid JSON: {e}", err=True)
        sys.exit(2)

    try:
        Manifest.model_validate(raw)
    except ValidationError as e:
        click.echo(f"ERROR: {path} failed manifest v0.1 validation:\n{e}", err=True)
        sys.exit(1)

    click.echo(f"OK: {path} is a valid v0.1 manifest")


if __name__ == "__main__":
    main()
