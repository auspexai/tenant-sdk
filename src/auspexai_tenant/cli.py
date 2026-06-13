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

import importlib
import json
import os
import sys
from pathlib import Path

import click
import httpx
from pydantic import ValidationError

from auspexai_tenant import __version__
from auspexai_tenant.client import CoordinatorError, TenantClient
from auspexai_tenant.github_device_flow import (
    DeviceCode,
    DeviceFlowError,
    default_client_id,
    run_device_flow,
)
from auspexai_tenant.manifest import Manifest, compute_package_digest
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
    default="https://coord.auspexai.network",
    show_default=True,
    envvar="AUSPEXAI_COORDINATOR_URL",
    help="Coordinator base URL (the public network by default, like `apply`). "
    "Override with --coordinator or AUSPEXAI_COORDINATOR_URL.",
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
@click.option(
    "--check-rekor",
    is_flag=True,
    help="Also perform an ONLINE Rekor transparency-log inclusion check — confirm "
    "the attestation's COSE artifact is publicly logged at its cited index.",
)
def experiment_attestation(
    experiment_id: str,
    coordinator: str,
    key_path: Path,
    verify_against_results: bool,
    checkpoint: bool,
    check_rekor: bool,
) -> None:
    """Fetch + independently verify the result-set attestation (#34).

    Verifies the recomputed Merkle root, the COSE signature, and (with
    --verify-against-results) that a freshly-pulled result set reproduces the
    attested root. All offline by default; pass --check-rekor for an online
    transparency-log inclusion check. Available once the experiment is COMPLETED
    — or pass --checkpoint for a partial (consensus-so-far) attestation while
    it's still running."""
    from auspexai_tenant.attestation import verify_against_results as _check_results
    from auspexai_tenant.attestation import verify_attestation as _verify

    client = _make_client(coordinator, key_path)
    att = _run(lambda: client.get_attestation(experiment_id, checkpoint=checkpoint))
    v = _verify(att, check_rekor=check_rekor)
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
    if v.rekor_inclusion is not None:
        ri = v.rekor_inclusion
        if not ri.checked:
            click.echo(f"rekor incl.: not anchored ({ri.error})")
        elif ri.included:
            click.echo(f"rekor incl.: ok — included in tree (size {ri.tree_size})")
        else:
            click.echo(f"rekor incl.: NOT VERIFIED ({ri.error or 'inclusion checks failed'})")
    if verify_against_results:
        results = list(_run(lambda: list(client.iter_results(experiment_id))))
        # v1 leaves bind the input hash; thread the attested values through so
        # the check stays a RESULTS-reproduction check (input binding itself is
        # the evidence bundle's `inputs_bound_ok`).
        hashes = {u["unit_id"]: u.get("unit_payload_sha256") or "" for u in att.units}
        ok_results = _check_results(att, results, unit_payload_hashes=hashes)
        click.echo(
            f"vs results:  {'ok — re-pulled set reproduces the attested root' if ok_results else 'MISMATCH'}"
        )
    if not v.ok:
        click.echo("VERIFICATION FAILED", err=True)
        sys.exit(1)
    click.echo("verified ✓")


@experiment.command("export")
@click.argument("experiment_id")
@_coord_opt
@_key_opt
@click.option(
    "-o",
    "--out",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Bundle output path (default: <experiment_id>-bundle.json).",
)
@click.option(
    "--verify/--no-verify",
    "do_verify",
    default=True,
    help="Run the full evidence-bundle verification chain after download (default on).",
)
@click.option(
    "--check-rekor",
    is_flag=True,
    help="Also perform the ONLINE Rekor transparency-log inclusion check on the "
    "bundled attestation (verification is otherwise fully offline).",
)
@click.option(
    "--signer",
    "signers",
    multiple=True,
    help="Hard-pin the custody + attestation signer to this pubkey hex "
    "(repeatable). Default: a public-network bundle is grounded against the "
    "SDK's embedded published keys automatically.",
)
@click.option(
    "--no-pin",
    is_flag=True,
    help="Skip signer grounding — accept self-consistency only.",
)
def experiment_export(
    experiment_id: str,
    coordinator: str,
    key_path: Path,
    output: Path | None,
    do_verify: bool,
    check_rekor: bool,
    signers: tuple[str, ...],
    no_pin: bool,
) -> None:
    """Take custody of the evidence bundle (EB-1, §9 #47).

    Downloads the self-contained bundle (manifest + work-unit inputs +
    consensus results + receipts + the Rekor-anchored attestation + a signed
    proof-of-transfer) and verifies the full chain: custody signature,
    attestation, root unification, completeness, input binding, and per-result
    WORKER signatures. Collecting transfers data custody to YOU and arms
    collection-anchored age-off coordinator-side — after age-off the network
    can re-verify your bundle forever but can never re-deliver the payloads.
    Your verified copy is the durable copy."""
    from auspexai_tenant.evidence import verify_bundle

    client = _make_client(coordinator, key_path)
    bundle = _run(lambda: client.export(experiment_id))
    out = output or Path(f"{experiment_id}-bundle.json")
    # The coordinator already delivered (and armed collection age-off) by the
    # time we write, so a missing -o parent dir must NOT lose the bundle: make
    # the parent before writing. Re-export re-delivers idempotently until
    # age-off, but the researcher shouldn't have to know that to recover.
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    t = bundle.get("transfer") or {}
    click.echo(f"bundle:      {out} ({out.stat().st_size} bytes)")
    click.echo(f"transfer:    {t.get('transfer_id')}  root_kind={t.get('root_kind')}")
    click.echo(
        f"contents:    {len(bundle.get('consensus_results') or [])} results, "
        f"{len(bundle.get('work_units') or [])} work units, "
        f"{len(bundle.get('receipts') or [])} receipts, "
        f"attestation {'present' if bundle.get('attestation') else 'ABSENT (pre-completion export)'}"
    )
    if not do_verify:
        return
    if signers and no_pin:
        click.echo("ERROR: use --signer OR --no-pin, not both.", err=True)
        sys.exit(1)
    v = verify_bundle(
        bundle, authorized_signers=list(signers) or None, no_pin=no_pin, check_rekor=check_rekor
    )

    def _fmt(value: bool | None) -> str:
        if value is None:
            return "n/a"
        return "ok" if value else "FAIL"

    click.echo(f"custody sig: {_fmt(v.transfer_signature_valid)}")
    click.echo(f"signer pin:  {_signer_pin_line(v)}")
    if v.attestation is not None:
        click.echo(f"attestation: {_fmt(v.attestation.ok)}")
    click.echo(f"root unify:  {_fmt(v.root_unified)}")
    click.echo(f"complete:    {_fmt(v.completeness_ok)}")
    click.echo(f"inputs:      {_fmt(v.inputs_bound_ok)}")
    ws = v.worker_signatures
    skipped = ws.skipped_aged_off + ws.skipped_missing_fields
    click.echo(
        f"worker sigs: {ws.verified} verified"
        + (f", {len(ws.failed)} FAILED ({', '.join(ws.failed)})" if ws.failed else "")
        + (f", {skipped} skipped" if skipped else "")
    )
    if not v.ok:
        click.echo("VERIFICATION FAILED — do not trust this bundle.", err=True)
        sys.exit(1)
    click.echo("verified ✓ — you now hold the durable copy")
    click.echo(
        "Per the Terms of Participation, data custody + legal responsibility are now "
        "yours; after age-off the network re-verifies but never re-delivers."
    )


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


def _load_attr(spec: str):
    """Load a tenant factory named `module:attr` (e.g. `mypkg.run:build`)."""
    if ":" not in spec:
        click.echo(f"ERROR: expected 'module:attr', got {spec!r}", err=True)
        sys.exit(1)
    module_name, _, attr = spec.partition(":")
    # Installed console scripts do NOT have the cwd on sys.path (unlike
    # `python script.py`), so `--driver my_driver:build` next to the file
    # failed unless the researcher knew to set PYTHONPATH (researcher-#0
    # finding). The driver-by-your-side case is the documented norm — add cwd.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as e:
        click.echo(f"ERROR: could not load {spec!r}: {e}", err=True)
        sys.exit(1)


def _load_key(key_path: Path):
    try:
        return MaintainerKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)


@experiment.command("submit")
@click.argument("pkg_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--manifest",
    "manifest_name",
    default="manifest.json",
    show_default=True,
    help="Manifest filename within PKG_DIR.",
)
@_coord_opt
@_key_opt
def experiment_submit(pkg_dir: Path, manifest_name: str, coordinator: str, key_path: Path) -> None:
    """Submit a BUILT executor package in one step: sign the manifest, upload
    the package (content-addressed), and create the experiment — collapsing
    `manifest sign` + `package upload` + `manifest upload`.

    A pure courier: it signs and ships exactly what your build produced. The
    label is authored upstream (experiment.toml → your build.py, which stamps a
    unique suffix via `make_unique_label`), so re-running your build then `submit`
    never collides. Prints the label and the coordinator experiment id."""
    manifest_path = pkg_dir / manifest_name
    if not manifest_path.exists():
        click.echo(f"ERROR: no manifest at {manifest_path}", err=True)
        sys.exit(1)
    try:
        m = Manifest.model_validate(json.loads(manifest_path.read_text()))
    except json.JSONDecodeError as e:
        click.echo(f"ERROR: {manifest_path} is not valid JSON: {e}", err=True)
        sys.exit(2)
    except ValidationError as e:
        click.echo(f"ERROR: {manifest_path} failed manifest validation:\n{e}", err=True)
        sys.exit(1)
    label = m.experiment_id  # the manifest's experiment_id IS the tenant label

    key = _load_key(key_path)
    client = _make_client(coordinator, key_path)

    # 1. sign the manifest (next to it, like `manifest sign`)
    sig = sign_manifest(m, key)
    sig_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    sig_path.write_text(sig.model_dump_json(indent=2) + "\n")
    click.echo(f"signed:     {sig_path}")

    # 2. upload the package (idempotent on the content-addressed tree)
    pkg_out = _run(lambda: client.upload_package(pkg_dir))
    digest = pkg_out.get("package_digest", "")
    click.echo(f"package:    {digest[:16]}… ({pkg_out.get('status', 'stored')})")

    # 3. create the experiment
    result = submit_experiment_from_files(manifest_path, sig_path, coordinator, key)
    if not result.ok:
        # A duplicate label is the most likely failure — point at the fix.
        hint = ""
        if result.status_code == 409:
            hint = (
                "\n  the label is already in use (labels are unique forever). Re-run "
                "your build so it stamps a fresh make_unique_label suffix, then submit again."
            )
        click.echo(f"ERROR: experiment creation failed ({result.status_code}){hint}", err=True)
        if result.body:
            click.echo(result.body, err=True)
        sys.exit(1)
    exp_id = None
    try:
        exp_id = json.loads(result.body).get("experiment_id")
    except (json.JSONDecodeError, AttributeError):
        pass
    click.echo(f"label:      {label}")
    click.echo(f"experiment: {exp_id or '(see response below)'}")
    if exp_id is None and result.body:
        click.echo(result.body)
    else:
        click.echo(f"next:       auspexai-tenant experiment run {label}   # or 'latest'")


def _resolve_experiment(client, target: str) -> tuple[str, str]:
    """Resolve a run TARGET — a coordinator experiment id, a tenant label, or
    the literal 'latest' — to (experiment_id, label). 'latest' = the most
    recently submitted experiment for this tenant; a label matches
    tenant_experiment_label (most recent wins if reused)."""

    def _submitted_at(e: dict) -> str:
        return e.get("submitted_at") or ""

    if target.startswith("exp-"):
        exp = _run(lambda: client.get_experiment(target))
        return target, exp.get("tenant_experiment_label", target)

    exps = _run(client.list_experiments)
    if not exps:
        click.echo("ERROR: no experiments found for this tenant.", err=True)
        sys.exit(1)
    if target == "latest":
        chosen = max(exps, key=_submitted_at)
    else:
        matches = [e for e in exps if e.get("tenant_experiment_label") == target]
        if not matches:
            click.echo(
                f"ERROR: no experiment with label {target!r} (and it isn't an exp- id "
                "or 'latest').",
                err=True,
            )
            sys.exit(1)
        chosen = max(matches, key=_submitted_at)
    return chosen["experiment_id"], chosen.get("tenant_experiment_label", target)


@experiment.command("run")
@click.argument("target")
@click.option(
    "--driver",
    "driver_spec",
    default=None,
    help="Tenant driver factory 'module:attr' returning a DriverSpec "
    "(condition / next_batch / reduce). Defaults to [driver].entrypoint in "
    "experiment.toml.",
)
@click.option(
    "--journal",
    "journal_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Run-journal path for durable crash-resume. Defaults to "
    "[driver].journal in experiment.toml, else <label>.journal.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="experiment.toml path (default: ./experiment.toml if present).",
)
@click.option(
    "--doorbell",
    is_flag=True,
    help="Also subscribe to the SSE event stream and poll immediately on a relevant "
    "event (the timer floor still guarantees liveness).",
)
@_coord_opt
@_key_opt
def experiment_run(
    target: str,
    driver_spec: str | None,
    journal_path: Path | None,
    config_path: Path | None,
    doorbell: bool,
    coordinator: str,
    key_path: Path,
) -> None:
    """Drive an autonomic (adaptive / run-until-convergence) experiment, headless.

    TARGET is a coordinator experiment id (exp-…), a tenant label, or the literal
    'latest' (the most recently submitted experiment for this tenant). The driver
    and journal default from experiment.toml's [driver] table, so a typical run is
    just `experiment run latest`.

    Loads your DriverSpec factory and runs the control loop — submit → poll agreed
    results → fold → test condition → next batch or finalize — until convergence,
    max_rounds, exhaustion, or a stall policy. Resumable via --journal."""
    from auspexai_tenant.driver import DriverSpec, run_until
    from auspexai_tenant.experiment import Experiment
    from auspexai_tenant.experiment_config import load_experiment_config
    from auspexai_tenant.wake import SseWake, sse_line_source

    cfg = load_experiment_config(config_path)
    driver_spec = driver_spec or cfg.driver_entrypoint
    if not driver_spec:
        click.echo(
            "ERROR: no driver — pass --driver module:attr or set [driver].entrypoint "
            "in experiment.toml.",
            err=True,
        )
        sys.exit(1)
    spec = _load_attr(driver_spec)()
    if not isinstance(spec, DriverSpec):
        click.echo(
            f"ERROR: {driver_spec} must return a DriverSpec, got {type(spec).__name__}", err=True
        )
        sys.exit(1)
    key = _load_key(key_path)
    client = _make_client(coordinator, key_path)
    experiment_id, label = _resolve_experiment(client, target)
    if journal_path is None:
        journal_path = cfg.journal_path(label)
    click.echo(f"experiment: {experiment_id}  (label {label})")
    click.echo(f"journal:    {journal_path}")
    exp = Experiment(coordinator, key, experiment_id)
    wake = spec.wake
    if wake is None and doorbell:
        wake = SseWake(sse_line_source(coordinator, key, experiment_id))
    result = _run(
        lambda: run_until(
            exp,
            condition=spec.condition,
            next_batch=spec.next_batch,
            reduce=spec.reduce,
            journal=journal_path,
            wake=wake,
            stall=spec.stall,
            max_rounds=spec.max_rounds,
        )
    )
    click.echo(f"outcome:  {result.outcome}")
    click.echo(f"rounds:   {result.rounds}")
    click.echo("aggregate:")
    click.echo(json.dumps(result.aggregate, indent=2, default=str))
    if result.attestation is not None:
        click.echo(f"attestation merkle_root: {result.attestation.merkle_root}")


@experiment.command("reduce")
@click.argument("experiment_id")
@click.option(
    "--reducer",
    "reducer_spec",
    required=True,
    help="Factory 'module:attr' returning a RunningAggregate (fold/finalize).",
)
@_coord_opt
@_key_opt
def experiment_reduce(
    experiment_id: str, reducer_spec: str, coordinator: str, key_path: Path
) -> None:
    """Batch-reduce a completed experiment's consensus result set with your
    aggregator — the post-completion experiment-level reduce (#34)."""
    agg = _load_attr(reducer_spec)()
    if not (hasattr(agg, "fold") and hasattr(agg, "finalize")):
        click.echo(f"ERROR: {reducer_spec} must return a RunningAggregate", err=True)
        sys.exit(1)
    client = _make_client(coordinator, key_path)
    results = _run(lambda: list(client.iter_results(experiment_id)))
    for result in results:
        agg.fold(result)
    click.echo(json.dumps(agg.finalize(), indent=2, default=str))


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


@main.group()
def software() -> None:
    """Request worker-baseline capabilities (code-plane demand, §9 #46)."""


@software.command("request")
@click.argument("title")
@click.option("--description", required=True, help="What capability is needed.")
@click.option("--reason", required=True, help="Why your experiments need it (one line).")
@_coord_opt
@_key_opt
def software_request(
    title: str, description: str, reason: str, coordinator: str, key_path: Path
) -> None:
    """Request a software capability the worker baseline doesn't provide.

    Enters the maintainer review queue: a dependencies/security/alternatives
    assessment is attached before approve/decline, and a recorded worker
    release later fulfils approved requests."""
    client = _make_client(coordinator, key_path)
    req = _run(lambda: client.request_software(title, description=description, reason=reason))
    click.echo(f"request {req['request_id']}: {req['status']}")
    click.echo("  queued for maintainer review (assessment → approve/decline → release).")


@software.command("list")
@click.option("--status", default=None, help="Filter: pending|assessed|approved|declined|released")
@_coord_opt
@_key_opt
def software_list(status: str | None, coordinator: str, key_path: Path) -> None:
    """List your tenant's software requests with assessment + resolution state."""
    client = _make_client(coordinator, key_path)
    reqs = _run(lambda: client.list_my_software_requests(status=status))
    for r in reqs:
        line = f"{r['request_id']}  {r['status']:9} {r['title']}"
        if r.get("release_version"):
            line += f"  (released in v{r['release_version']})"
        click.echo(line)
        if r.get("assessment"):
            draft = " [AUTO-DRAFT — unratified]" if r.get("assessment_draft") else ""
            summary = r["assessment"].get("summary") or r["assessment"]["security"]
            click.echo(f"    assessment{draft}: {summary}")
        if r.get("resolution_reason"):
            click.echo(f"    resolution ({r.get('resolved_by', '?')}): {r['resolution_reason']}")
    if not reqs:
        click.echo("(no software requests yet)")


# ----------------------------------------------------------------------------
# executor-package upload (coordinator-served provisioning, §9 #40a)
# ----------------------------------------------------------------------------


@main.group()
def package() -> None:
    """Upload executor packages for coordinator-served provisioning (§9 #40)."""


@package.command("upload")
@click.argument("pkg_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--coordinator",
    default="https://coord.auspexai.network",
    show_default=True,
    envvar="AUSPEXAI_COORDINATOR_URL",
    help="Coordinator base URL (the public network by default, like `apply`).",
)
@_key_opt
def package_upload(pkg_dir: Path, coordinator: str, key_path: Path) -> None:
    """Upload the executor package staged in PKG_DIR to the coordinator.

    Builds a deterministic tar.gz of the package tree (sorted entries, fixed
    metadata — the same tree always produces identical bytes) excluding what
    the package digest excludes (manifest.json[.sig], __pycache__/, *.pyc),
    then POSTs it RFC 9421-signed with the tree digest in X-Package-Digest.
    The digest printed is the value to pin as `executor.package_sha256` in
    your manifest; the coordinator re-derives it and refuses a mismatch."""
    client = _make_client(coordinator, key_path)
    out = _run(lambda: client.upload_package(pkg_dir))
    digest = out.get("package_digest") or compute_package_digest(pkg_dir)
    status = out.get("status", "stored")
    click.echo(f"package_digest: {digest}")
    click.echo(f"status:         {status}")
    if status == "already_exists":
        click.echo("  the coordinator already holds this exact package tree.")
    else:
        click.echo("  pin this digest as executor.package_sha256 in your manifest before signing.")


# ----------------------------------------------------------------------------
# tenant application (apply-from-CLI onboarding, Option D)
# ----------------------------------------------------------------------------


def _ensure_key(key_path: Path) -> tuple[MaintainerKey, bool]:
    """Load the tenant key at `key_path`, generating one if missing.

    Returns (key, created). A corrupt existing file is an error (we never
    silently overwrite key material)."""
    if key_path.exists():
        return _load_key(key_path), False
    new_key = MaintainerKey.generate()
    new_key.save(key_path)
    return new_key, True


def _stdin_isatty() -> bool:
    """Is stdin interactive? A separate seam (not a bare sys.stdin.isatty()
    call) so tests can simulate a tty under CliRunner's replaced stdin."""
    return sys.stdin.isatty()


def _required_field(value: str | None, flag: str, prompt_text: str) -> str:
    """Return `value`, prompting interactively when absent on a tty; error out
    (before any device-flow work) when absent non-interactively."""
    if value:
        return value
    if _stdin_isatty():
        return click.prompt(prompt_text)
    click.echo(f"ERROR: {flag} is required (or run interactively to be prompted).", err=True)
    sys.exit(1)


# Research-class taxonomy (pinned contract with the coordinator): id → human
# label. Order matters — it is the numbering shown in the interactive picker.
RESEARCH_CLASSES: dict[str, str] = {
    "behavioral_drift": "Longitudinal behavioral drift",
    "eval_sweeps": "Deterministic eval sweeps",
    "refusal_boundary_mapping": "Refusal/jailbreak-boundary mapping",
    "cross_model_comparison": "Cross-model comparison",
    "quantization_effects": "Quantization-effect studies",
    "prompt_sensitivity": "Prompt-sensitivity analysis",
    "other": "Other",
}


def _parse_research_class_selection(raw: str) -> list[str]:
    """Map a '1,5'-style picker answer to taxonomy ids (order kept, deduped).

    Empty input is a valid skip → []. Raises ValueError on anything that is
    not a 1-based number within the taxonomy."""
    ids = list(RESEARCH_CLASSES)
    chosen: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or not 1 <= int(part) <= len(ids):
            raise ValueError(f"'{part}' is not a number between 1 and {len(ids)}")
        cid = ids[int(part) - 1]
        if cid not in chosen:
            chosen.append(cid)
    return chosen


def _prompt_research_classes() -> list[str]:
    """Interactive numbered multi-select over the research-class taxonomy.
    Enter on an empty line skips (the free-text summary can then carry the
    at-least-one requirement); invalid input re-prompts."""
    click.echo("Research areas:")
    for i, label in enumerate(RESEARCH_CLASSES.values(), start=1):
        click.echo(f"  {i}. {label}")
    while True:
        raw = click.prompt(
            "Select research areas (comma-separated numbers, e.g. 1,5)",
            default="",
            show_default=False,
        )
        try:
            return _parse_research_class_selection(raw)
        except ValueError as e:
            click.echo(f"  {e} — try again (Enter to skip).")


@main.command("apply")
@click.option(
    "--coordinator",
    default="https://coord.auspexai.network",
    show_default=True,
    envvar="AUSPEXAI_COORDINATOR_URL",
    help="Coordinator base URL. The public network is the default — apply is "
    "the front-door command and must not require operator knowledge.",
)
@click.option(
    "--key",
    "key_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Tenant key that signs the application (generated here if missing).",
)
@click.option(
    "--status",
    "show_status",
    is_flag=True,
    help="Show the status of your existing applications instead of applying.",
)
@click.option(
    "--tenant-id",
    default=None,
    help="Your PERMANENT research-tenant identifier — all your experiments "
    "will live under it (think lab/org slug, e.g. 'my-lab'). You apply once; "
    "running more experiments never needs another application.",
)
@click.option("--name", default=None, help="Contact name.")
@click.option("--affiliation", default=None, help="Affiliation (lab / institution / independent).")
@click.option(
    "--research-class",
    "research_classes",
    multiple=True,
    type=click.Choice(list(RESEARCH_CLASSES)),
    help="Research area from the network taxonomy (repeatable). At least one "
    "of --research-class / --summary is required.",
)
@click.option(
    "--summary",
    default=None,
    help="Free-text research summary (optional when --research-class is given).",
)
def apply_cmd(
    coordinator: str,
    key_path: Path,
    show_status: bool,
    tenant_id: str | None,
    name: str | None,
    affiliation: str | None,
    research_classes: tuple[str, ...],
    summary: str | None,
) -> None:
    """Apply for a tenant account, entirely from the CLI.

    Generates (or reuses) your tenant key, verifies your GitHub identity via
    Device Flow, and submits an application RFC 9421-signed by that key — the
    coordinator learns your public key from the signature itself, so no pubkey
    is ever hand-passed and proof-of-possession is built in. Track the outcome
    with `auspexai-tenant apply --status`; once approved, the same key is your
    tenant credential for every other command."""
    if show_status:
        client = _make_client(coordinator, key_path)
        apps = _run(client.my_tenant_applications)
        for a in apps:
            click.echo(f"{a['application_id']}  {a['status']}")
            if a.get("created_tenant_id"):
                click.echo(f"    tenant: {a['created_tenant_id']}")
            if a.get("resolution_reason"):
                click.echo(f"    resolution: {a['resolution_reason']}")
        if not apps:
            click.echo("(no applications)")
        return

    # Collect the application fields BEFORE any key/device-flow work so a
    # missing flag never burns a device code.
    tenant_id = _required_field(
        tenant_id,
        "--tenant-id",
        "Tenant id (permanent — all your experiments live under it; e.g. 'my-lab')",
    )
    name = _required_field(name, "--name", "Contact name")
    affiliation = _required_field(
        affiliation, "--affiliation", "Affiliation (organization, or 'independent')"
    )
    classes = list(research_classes)
    if _stdin_isatty():
        if not classes:
            classes = _prompt_research_classes()
        if not summary:
            summary = (
                click.prompt(
                    "Anything else about your research? (Enter to skip)",
                    default="",
                    show_default=False,
                ).strip()
                or None
            )
    if not classes and not summary:
        click.echo(
            "ERROR: describe your research — give at least one --research-class "
            f"({', '.join(RESEARCH_CLASSES)}) and/or a --summary.",
            err=True,
        )
        sys.exit(1)

    k, created = _ensure_key(key_path)
    if created:
        click.echo(f"Generated a new tenant key at {key_path}")
    click.echo(f"Signing key (becomes your tenant credential on approval): {k.pubkey_hex}")

    def _show_code(code: DeviceCode) -> None:
        click.echo("GitHub identity check — in any browser:")
        click.echo(f"  1. open  {code.verification_uri}")
        click.echo(f"  2. enter {code.user_code}")
        click.echo("Waiting for authorization (Ctrl-C to abort)...")

    try:
        token = run_device_flow(on_code=_show_code, client_id=default_client_id())
    except DeviceFlowError as e:
        click.echo(f"ERROR: GitHub device flow failed: {e}", err=True)
        sys.exit(1)

    client = TenantClient(coordinator, k)
    out = _run(
        lambda: client.apply_for_tenant(
            github_access_token=token,
            requested_tenant_id=tenant_id,
            contact_name=name,
            affiliation=affiliation,
            research_summary=summary,
            research_classes=classes or None,
        )
    )
    click.echo(f"application {out['application_id']}: {out['status']}")
    click.echo(f"  track with: auspexai-tenant apply --status --coordinator {coordinator}")


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


# Re-export for module-level imports
__all__ = ["ManifestSignature", "main"]


@main.group()
def bundle() -> None:
    """Work with saved evidence bundles — re-verify forever, offline."""


def _signer_pin_line(v) -> str:
    """One-line custody-signer grounding verdict from a BundleVerification.
    Shared by `experiment export` and `bundle verify` so the trust claim reads
    identically wherever the chain is checked."""
    mode = v.signer_pin_mode
    if mode == "known":
        return "trusted ✓ — AuspexAI published key"
    if mode == "explicit":
        return (
            "ok — pinned to your --signer"
            if v.transfer_signer_authorized
            else "FAIL — signer is NOT in your --signer set"
        )
    if mode == "skipped":
        return "skipped (--no-pin) — self-consistency only"
    return (
        "unpinned — signer is not a known AuspexAI key; pass --signer <hex> "
        "(see AUTHORIZED_SIGNERS.md) or --no-pin"
    )


@bundle.command("verify")
@click.argument("bundle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--signer",
    "signers",
    multiple=True,
    help="Authorized signer pubkey hex (repeatable) — HARD-pins the custody AND "
    "attestation keys externally (e.g. a private coordinator, or a specific "
    "AUTHORIZED_SIGNERS.md key). By default a public-network bundle is already "
    "grounded against the SDK's embedded published keys.",
)
@click.option(
    "--no-pin",
    is_flag=True,
    help="Skip signer grounding entirely — accept self-consistency only "
    "(no claim that the signer is a known AuspexAI key).",
)
@click.option(
    "--check-rekor",
    is_flag=True,
    help="Also perform the ONLINE Rekor inclusion check (otherwise fully offline).",
)
def bundle_verify(
    bundle_file: Path, signers: tuple[str, ...], no_pin: bool, check_rekor: bool
) -> None:
    """Re-verify a saved evidence bundle — works forever, with no coordinator.

    The network's custody doctrine is re-verify-forever/never-re-deliver:
    hashes, receipts, and attestations are retained indefinitely (and the
    public Rekor log holds the anchor even without us), so the bundle you
    downloaded can be re-checked at any time, by anyone you hand it to."""
    from auspexai_tenant.evidence import verify_bundle

    if signers and no_pin:
        click.echo("ERROR: use --signer OR --no-pin, not both.", err=True)
        sys.exit(1)
    data = json.loads(bundle_file.read_text(encoding="utf-8"))
    v = verify_bundle(
        data,
        authorized_signers=list(signers) or None,
        no_pin=no_pin,
        check_rekor=check_rekor,
    )

    def _fmt(value: bool | None) -> str:
        return "n/a" if value is None else ("ok" if value else "FAIL")

    click.echo(f"custody sig: {_fmt(v.transfer_signature_valid)}")
    click.echo(f"signer pin:  {_signer_pin_line(v)}")
    if v.attestation is not None:
        click.echo(f"attestation: {_fmt(v.attestation.ok)}")
    click.echo(f"root unify:  {_fmt(v.root_unified)}")
    click.echo(f"complete:    {_fmt(v.completeness_ok)}")
    click.echo(f"inputs:      {_fmt(v.inputs_bound_ok)}")
    ws = v.worker_signatures
    skipped = ws.skipped_aged_off + ws.skipped_missing_fields
    click.echo(
        f"worker sigs: {ws.verified} verified"
        + (f", {len(ws.failed)} FAILED ({', '.join(ws.failed)})" if ws.failed else "")
        + (f", {skipped} skipped" if skipped else "")
    )
    if not v.ok:
        click.echo("VERIFICATION FAILED — do not trust this bundle.", err=True)
        sys.exit(1)
    click.echo("verified ✓")


@bundle.command("table")
@click.argument("bundle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output table; format by extension: .csv or .parquet.",
)
@click.option(
    "--signer", "signers", multiple=True, help="Authorized signer pubkey hex (repeatable)."
)
@click.option("--check-rekor", is_flag=True, help="Also check Rekor inclusion online first.")
def bundle_table(
    bundle_file: Path, out_path: Path, signers: tuple[str, ...], check_rekor: bool
) -> None:
    """VERIFY the bundle, then write its results as a flat table.

    One row per consensus result; work-unit inputs flatten to input.* columns
    and result payloads to output.* — ready for pandas, Excel, Tableau, or R.
    Refuses to write anything from a bundle that fails verification.
    Needs the analysis extra: pip install 'auspexai-tenant[analysis]'."""
    from auspexai_tenant.evidence import BundleVerificationError, load_verified

    try:
        df = load_verified(
            bundle_file,
            authorized_signers=list(signers) or None,
            check_rekor=check_rekor,
        )
    except BundleVerificationError as e:
        click.echo(f"REFUSED: {e}", err=True)
        sys.exit(1)
    # Residual non-scalar cells (lists, e.g. lexical.top_tokens) JSON-encode so
    # the table is typed-column clean for Parquet and round-trips through CSV.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        click.echo(f"ERROR: unsupported table format {suffix!r} (use .csv or .parquet)", err=True)
        sys.exit(1)
    click.echo(f"verified ✓ → {out_path} ({len(df)} rows, {len(df.columns)} columns)")
