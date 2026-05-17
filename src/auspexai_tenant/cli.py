"""auspexai-tenant CLI entrypoint.

v0.1 commands:
    auspexai-tenant key generate            # generate maintainer Ed25519 keypair
    auspexai-tenant key pubkey              # print maintainer public key
    auspexai-tenant manifest validate       # validate a manifest against the schema
    auspexai-tenant manifest sign           # sign a manifest with the maintainer key
    auspexai-tenant manifest upload         # POST a (signed) manifest to a coordinator

Future commands (next sessions):
- workunits tar-writer helper
- receipts show
- executor harness scaffolding (the harness itself ships as a library entry; no
  CLI wrapper yet — tenants embed it directly).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import httpx
from pydantic import ValidationError

from auspexai_tenant import __version__
from auspexai_tenant.manifest import Manifest
from auspexai_tenant.signing import (
    DEFAULT_KEY_PATH,
    MaintainerKey,
    ManifestSignature,
    sign_manifest,
)
from auspexai_tenant.upload import upload_manifest


@click.group()
@click.version_option(version=__version__, prog_name="auspexai-tenant")
def main() -> None:
    """AuspexAI Tenant SDK CLI."""


# ----------------------------------------------------------------------------
# key commands
# ----------------------------------------------------------------------------


@main.group()
def key() -> None:
    """Maintainer key commands."""


@key.command("generate")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Where to write the keypair (PKCS8 PEM, mode 0600).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing keypair at the output path.",
)
def key_generate(output: Path, force: bool) -> None:
    """Generate a fresh Ed25519 maintainer keypair."""
    if output.exists() and not force:
        click.echo(
            f"ERROR: {output} already exists. Re-run with --force to overwrite.",
            err=True,
        )
        sys.exit(1)
    new_key = MaintainerKey.generate()
    new_key.save(output)
    click.echo(f"Generated maintainer key at {output}")
    click.echo(f"Public key: {new_key.pubkey_hex}")
    click.echo("Register this public key with the AuspexAI coordinator to enable manifest uploads.")


@key.command("pubkey")
@click.option(
    "--key",
    "key_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Path to the maintainer keypair PEM file.",
)
def key_pubkey(key_path: Path) -> None:
    """Print the public key from a maintainer keypair."""
    try:
        k = MaintainerKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)
    click.echo(k.pubkey_hex)


# ----------------------------------------------------------------------------
# manifest commands
# ----------------------------------------------------------------------------


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


@manifest.command("sign")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--key",
    "key_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Path to the maintainer keypair PEM file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the signature file (default: <manifest>.sig).",
)
def manifest_sign(path: Path, key_path: Path, output: Path | None) -> None:
    """Sign a manifest with the maintainer keypair. Produces <manifest>.sig."""
    try:
        raw = json.loads(path.read_text())
        m = Manifest.model_validate(raw)
    except json.JSONDecodeError as e:
        click.echo(f"ERROR: {path} is not valid JSON: {e}", err=True)
        sys.exit(2)
    except ValidationError as e:
        click.echo(f"ERROR: {path} failed manifest v0.1 validation:\n{e}", err=True)
        sys.exit(1)

    try:
        k = MaintainerKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)

    sig = sign_manifest(m, k)
    sig_path = output if output is not None else path.with_suffix(path.suffix + ".sig")
    sig_path.write_text(sig.model_dump_json(indent=2) + "\n")
    click.echo(f"OK: signed {path}")
    click.echo(f"Signature: {sig_path}")
    click.echo(f"Maintainer pubkey: {k.pubkey_hex}")


@manifest.command("upload")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--coordinator",
    required=True,
    help="Coordinator base URL (e.g., https://coordinator.example.com).",
)
@click.option(
    "--sig",
    "sig_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Signature file path (default: <manifest>.sig if it exists).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the upload request without sending it.",
)
def manifest_upload(path: Path, coordinator: str, sig_path: Path | None, dry_run: bool) -> None:
    """Upload a manifest (and signature, if present) to a coordinator."""
    if sig_path is None:
        default_sig = path.with_suffix(path.suffix + ".sig")
        sig_path = default_sig if default_sig.exists() else None
    elif not sig_path.exists():
        click.echo(f"ERROR: --sig path does not exist: {sig_path}", err=True)
        sys.exit(1)

    if dry_run:
        endpoint = f"{coordinator.rstrip('/')}/api/v0/manifests"
        click.echo(f"[dry-run] POST {endpoint}")
        click.echo(f"[dry-run] manifest: {path} ({path.stat().st_size} bytes)")
        if sig_path:
            click.echo(f"[dry-run] signature: {sig_path} ({sig_path.stat().st_size} bytes)")
        else:
            click.echo("[dry-run] signature: (none)")
        return

    try:
        result = upload_manifest(path, coordinator, sig_path)
    except httpx.RequestError as e:
        click.echo(f"ERROR: network failure: {e}", err=True)
        sys.exit(2)

    if result.ok:
        click.echo(f"OK: uploaded ({result.status_code})")
        if result.body:
            click.echo(result.body)
    else:
        click.echo(f"ERROR: upload failed ({result.status_code})", err=True)
        if result.body:
            click.echo(result.body, err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


# Re-export for module-level imports
__all__ = ["ManifestSignature", "main"]
