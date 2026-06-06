"""auspexai-tenant CLI entrypoint.

v0.1 commands:
    auspexai-tenant key generate            # generate maintainer Ed25519 keypair
    auspexai-tenant key pubkey              # print maintainer public key
    auspexai-tenant manifest validate       # validate a manifest against the schema
    auspexai-tenant manifest sign           # sign a manifest with the maintainer key
    auspexai-tenant manifest upload         # POST a (signed) manifest to a coordinator
    auspexai-tenant receipts show           # pretty-print a CBOR-encoded receipt

The ExecutorHarness and ReducerHarness ship as library entries (tenants embed
them in their executor / reducer scripts directly); they have no CLI wrapper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import httpx
from pydantic import ValidationError

from auspexai_tenant import __version__
from auspexai_tenant.client import CoordinatorError, TenantClient, verify_transfer
from auspexai_tenant.manifest import Manifest
from auspexai_tenant.receipts import decode_cbor
from auspexai_tenant.signing import (
    DEFAULT_KEY_PATH,
    MaintainerKey,
    ManifestSignature,
    sign_manifest,
)
from auspexai_tenant.upload import submit_experiment_from_files


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
    help="Coordinator base URL (e.g., https://coord.auspexai.network).",
)
@click.option(
    "--sig",
    "sig_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Signature file path (default: <manifest>.sig).",
)
@click.option(
    "--key",
    "key_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Maintainer key used to sign the request (RFC 9421).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the request that would be sent without sending it.",
)
def manifest_upload(
    path: Path, coordinator: str, sig_path: Path | None, key_path: Path, dry_run: bool
) -> None:
    """Submit a manifest + signature to a coordinator's `POST /experiments`.

    The request is authenticated with an RFC 9421 HTTP Message Signature using
    the maintainer key; the coordinator resolves the signing key to your tenant.
    A signature file (from `manifest sign`) is required.
    """
    if sig_path is None:
        sig_path = path.with_suffix(path.suffix + ".sig")
    if not sig_path.exists():
        click.echo(
            f"ERROR: signature file not found: {sig_path}\n"
            "Run `auspexai-tenant manifest sign` first, or pass --sig.",
            err=True,
        )
        sys.exit(1)

    endpoint = f"{coordinator.rstrip('/')}/api/v0/experiments"

    if dry_run:
        click.echo(f"[dry-run] POST {endpoint}  (RFC 9421 signed)")
        click.echo(f"[dry-run] manifest: {path} ({path.stat().st_size} bytes)")
        click.echo(f"[dry-run] signature: {sig_path} ({sig_path.stat().st_size} bytes)")
        click.echo(f"[dry-run] signing key: {key_path}")
        return

    try:
        k = MaintainerKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)

    try:
        result = submit_experiment_from_files(path, sig_path, coordinator, k)
    except httpx.RequestError as e:
        click.echo(f"ERROR: network failure: {e}", err=True)
        sys.exit(2)

    if result.ok:
        click.echo(f"OK: submitted ({result.status_code})")
        if result.body:
            click.echo(result.body)
    else:
        click.echo(f"ERROR: submission failed ({result.status_code})", err=True)
        if result.body:
            click.echo(result.body, err=True)
        sys.exit(1)


# ----------------------------------------------------------------------------
# receipts commands
# ----------------------------------------------------------------------------


@main.group()
def receipts() -> None:
    """Receipt commands."""


@receipts.command("show")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def receipts_show(path: Path) -> None:
    """Decode a CBOR-encoded receipt and pretty-print it as JSON.

    v0.1 reads raw CBOR receipts (the predicate body of an in-toto Statement
    before COSE wrapping). Full COSE + in-toto + Rekor verification lands in
    a later milestone alongside the platform-side signing infrastructure.
    """
    try:
        receipt = decode_cbor(path.read_bytes())
    except ValidationError as e:
        click.echo(f"ERROR: {path} failed receipt v0.1 validation:\n{e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"ERROR: failed to decode {path}: {e}", err=True)
        sys.exit(2)

    click.echo(receipt.model_dump_json(indent=2))


# ----------------------------------------------------------------------------
# experiment commands (M-Results retrieval)
# ----------------------------------------------------------------------------

_coord_opt = click.option(
    "--coordinator",
    required=True,
    help="Coordinator base URL (e.g., https://coord.auspexai.network).",
)
_key_opt = click.option(
    "--key",
    "key_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Tenant key used to sign requests (RFC 9421).",
)


def _make_client(coordinator: str, key_path: Path) -> TenantClient:
    try:
        k = MaintainerKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)
    return TenantClient(coordinator, k)


def _run(fn):
    """Call `fn`, mapping coordinator/network failures to CLI exit codes."""
    try:
        return fn()
    except CoordinatorError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except httpx.RequestError as e:
        click.echo(f"ERROR: network failure: {e}", err=True)
        sys.exit(2)


@main.group()
def experiment() -> None:
    """Retrieve your experiments, results, receipts, and the offload bundle."""


@experiment.command("list")
@_coord_opt
@_key_opt
def experiment_list(coordinator: str, key_path: Path) -> None:
    """List my experiments."""
    client = _make_client(coordinator, key_path)
    exps = _run(client.list_experiments)
    for e in exps:
        label = e.get("tenant_experiment_label", "")
        click.echo(f"{e.get('experiment_id')}  {e.get('status')}  {label}")
    if not exps:
        click.echo("(no experiments)")


@experiment.command("attestation")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
@click.option(
    "--verify-against-results",
    is_flag=True,
    help="Also re-pull the consensus result set and recompute the root (the strong "
    "reproducibility check — proves the set you'd reduce is the attested one).",
)
@click.option(
    "--checkpoint",
    is_flag=True,
    help="Fetch a partial consensus-so-far attestation over a not-yet-COMPLETED "
    "experiment (M9 leg 2) — the integrity anchor for a partial collection.",
)
def experiment_attestation(
    experiment_id: str,
    coordinator: str,
    key_path: Path,
    verify_against_results: bool,
    checkpoint: bool,
) -> None:
    """Fetch + independently verify the result-set attestation (#34).

    Verifies the recomputed Merkle root, the COSE signature, and (with
    --verify-against-results) that a freshly-pulled result set reproduces the
    attested root. Available once the experiment is COMPLETED — or pass
    --checkpoint for a partial (consensus-so-far) attestation while it's still
    running."""
    from auspexai_tenant.attestation import verify_against_results as _check_results
    from auspexai_tenant.attestation import verify_attestation as _verify

    client = _make_client(coordinator, key_path)
    att = _run(lambda: client.get_attestation(experiment_id, checkpoint=checkpoint))
    v = _verify(att)
    click.echo(f"attestation: {att.attestation_id}")
    if att.partial:
        click.echo("kind:        PARTIAL (checkpoint — consensus-so-far, not the final set)")
    click.echo(f"merkle_root: {att.merkle_root}")
    click.echo(f"units:       {att.unit_count}")
    click.echo(f"signer:      {v.signer_pubkey_hex}")
    click.echo(f"signature:   {'valid' if v.signature_valid else 'INVALID'}")
    root_ok = v.root_matches_units and v.signed_root_matches
    click.echo(f"root match:  {'ok' if root_ok else 'MISMATCH'}")
    if att.rekor_log_index:
        click.echo(f"rekor:       logIndex {att.rekor_log_index}")
    if verify_against_results:
        results = list(_run(lambda: list(client.iter_results(experiment_id))))
        click.echo(
            f"vs results:  {'ok — re-pulled set reproduces the attested root' if _check_results(att, results) else 'MISMATCH'}"
        )
    if not v.ok:
        click.echo("VERIFICATION FAILED", err=True)
        sys.exit(1)
    click.echo("verified ✓")


@experiment.command("status")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
def experiment_status(experiment_id: str, coordinator: str, key_path: Path) -> None:
    """Show one experiment's detail (status, retention, collection)."""
    client = _make_client(coordinator, key_path)
    click.echo(json.dumps(_run(lambda: client.get_experiment(experiment_id)), indent=2))


@experiment.command("results")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
@click.option(
    "--raw", is_flag=True, help="Include all replicas (T-X), not just the consensus copy."
)
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write results JSON to a file instead of stdout.",
)
def experiment_results(
    experiment_id: str, coordinator: str, key_path: Path, raw: bool, out_path: Path | None
) -> None:
    """Fetch result payloads — consensus (one per unit) by default, --raw for all
    replicas. Pages through the full result set."""
    client = _make_client(coordinator, key_path)
    include = "raw" if raw else "consensus"
    items = _run(lambda: list(client.iter_results(experiment_id, include=include)))
    text = json.dumps(items, indent=2)
    if out_path is not None:
        out_path.write_text(text)
        click.echo(f"Wrote {len(items)} result(s) to {out_path}")
    else:
        click.echo(text)


@experiment.command("receipts")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
def experiment_receipts(experiment_id: str, coordinator: str, key_path: Path) -> None:
    """List the receipts issued for my experiment (the permanent proof layer)."""
    client = _make_client(coordinator, key_path)
    receipts_ = _run(lambda: client.get_receipts(experiment_id))
    for r in receipts_:
        click.echo(f"{r.get('receipt_id')}  issued {r.get('issued_at', '')}")
    if not receipts_:
        click.echo("(no receipts yet)")


@experiment.command("export")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Bundle output path (default: <experiment_id>-bundle.json).",
)
def experiment_export(
    experiment_id: str, coordinator: str, key_path: Path, out_path: Path | None
) -> None:
    """Collect the offload bundle (consensus results + receipts + manifest + a
    signed custody record) and save it. Collecting transfers data custody to you."""
    client = _make_client(coordinator, key_path)
    bundle = _run(lambda: client.export(experiment_id))
    v = verify_transfer(bundle)
    out = out_path or Path(f"{experiment_id}-bundle.json")
    out.write_text(json.dumps(bundle, indent=2))
    n_results = len(bundle.get("consensus_results") or [])
    n_receipts = len(bundle.get("receipts") or [])
    click.echo(f"Saved offload bundle to {out}: {n_results} result(s), {n_receipts} receipt(s).")
    if v.valid:
        click.echo(
            f"Custody transfer VERIFIED — transfer {v.transfer_id}, "
            f"signed by coordinator {v.coordinator_pubkey_hex[:16]}…"
        )
        click.echo(
            "Per the Terms of Participation, data custody + legal responsibility are now yours."
        )
    else:
        click.echo(
            "WARNING: the custody-record signature did NOT verify against the "
            "coordinator key. Do not trust this bundle.",
            err=True,
        )
        sys.exit(1)


# ----------------------------------------------------------------------------


@main.group()
def model() -> None:
    """Browse the network model catalog + request models (demand-board, §9 #39)."""


@model.command("catalog")
@_coord_opt
@_key_opt
def model_catalog(coordinator: str, key_path: Path) -> None:
    """Show the network's bottom-up model catalog (what active workers can run)."""
    client = _make_client(coordinator, key_path)
    cat = _run(client.get_catalog)
    models = cat.get("models") or []
    click.echo(f"network: {cat.get('total_active_workers', 0)} active worker(s)")
    for m in models:
        click.echo(f"  {m['model_id']:48} {m['worker_count']} worker(s)")
    if not models:
        click.echo("(no models on the network yet)")


@model.command("request")
@click.argument("model_id")
@click.option("--reason", required=True, help="Why you need this model (one line).")
@click.option("--hf-repo", "hf_repo", default=None, help="Optional HuggingFace repo hint.")
@_coord_opt
@_key_opt
def model_request(
    model_id: str, reason: str, hf_repo: str | None, coordinator: str, key_path: Path
) -> None:
    """Request a model (BYOM). MODEL_ID is the worker store id (<repo-slug>-<quant>)."""
    client = _make_client(coordinator, key_path)
    req = _run(lambda: client.request_model(model_id, reason=reason, hf_repo=hf_repo))
    click.echo(f"request {req['request_id']}: {req['status']}")
    if req["status"] == "available":
        click.echo("  the network already has a worker that can run this model.")
    elif req["status"] == "pending":
        click.echo("  no active worker holds it yet — queued for maintainer review.")


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


# Re-export for module-level imports
__all__ = ["ManifestSignature", "main"]
