"""auspexai-tenant CLI entrypoint.

v0.1 commands:
    auspexai-tenant key generate            # generate your tenant Ed25519 keypair
    auspexai-tenant key pubkey              # print your tenant public key
    auspexai-tenant manifest validate       # validate a manifest against the schema
    auspexai-tenant manifest sign           # sign a manifest with your tenant key
    auspexai-tenant manifest upload         # POST a (signed) manifest to a coordinator
    auspexai-tenant receipts show           # pretty-print a CBOR-encoded receipt

The ExecutorHarness and ReducerHarness ship as library entries (tenants embed
them in their executor / reducer scripts directly); they have no CLI wrapper.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
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
from auspexai_tenant.runs import RunLayout
from auspexai_tenant.signing import (
    DEFAULT_KEY_PATH,
    ManifestSignature,
    TenantKey,
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
    """Tenant key commands."""


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
    """Generate a fresh Ed25519 tenant keypair."""
    if output.exists() and not force:
        click.echo(
            f"ERROR: {output} already exists. Re-run with --force to overwrite.",
            err=True,
        )
        sys.exit(1)
    new_key = TenantKey.generate()
    new_key.save(output)
    click.echo(f"Generated tenant key at {output}")
    click.echo(f"Public key: {new_key.pubkey_hex}")
    click.echo("Register this public key with the AuspexAI coordinator to enable manifest uploads.")


@key.command("pubkey")
@click.option(
    "--key",
    "key_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_KEY_PATH,
    show_default=True,
    help="Path to your tenant keypair PEM file.",
)
def key_pubkey(key_path: Path) -> None:
    """Print the public key from a tenant keypair."""
    try:
        k = TenantKey.load(key_path)
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
    help="Path to your tenant keypair PEM file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the signature file (default: <manifest>.sig).",
)
def manifest_sign(path: Path, key_path: Path, output: Path | None) -> None:
    """Sign a manifest with your tenant keypair. Produces <manifest>.sig."""
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
        k = TenantKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)

    sig = sign_manifest(m, k)
    sig_path = output if output is not None else path.with_suffix(path.suffix + ".sig")
    sig_path.write_text(sig.model_dump_json(indent=2) + "\n")
    click.echo(f"OK: signed {path}")
    click.echo(f"Signature: {sig_path}")
    click.echo(f"Tenant pubkey: {k.pubkey_hex}")


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
    help="Tenant key used to sign the request (RFC 9421).",
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
    your tenant key; the coordinator resolves the signing key to your tenant.
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
        k = TenantKey.load(key_path)
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
_profile_opt = click.option(
    "--profile",
    "profile",
    default=None,
    help="Select a [profiles.<name>] override-set from experiment.toml "
    "(e.g. starter / research) — its sub-tables deep-merge onto the base config. "
    "Default: the base config (or [default_profile] if set).",
)


def _make_client(coordinator: str, key_path: Path) -> TenantClient:
    try:
        k = TenantKey.load(key_path)
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
@click.argument("target")
@_coord_opt
@_key_opt
@click.option(
    "-o",
    "--out",
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Bundle output path (default: runs/<label>/bundle.json).",
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
    target: str,
    coordinator: str,
    key_path: Path,
    output: Path | None,
    do_verify: bool,
    check_rekor: bool,
    signers: tuple[str, ...],
    no_pin: bool,
) -> None:
    """Take custody of the evidence bundle (EB-1, §9 #47).

    TARGET is a coordinator experiment id (`exp-…`), a tenant label, or the
    literal `latest` — the label keys the local run directory so the bundle
    lands beside its journal under runs/<label>/.

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
    # Accept an exp- id, a tenant label, or 'latest' (like `run`); the resolved
    # label keys the local run directory so the bundle co-locates with the
    # journal under runs/<label>/.
    exp_id, label = _resolve_experiment(client, target)
    bundle = _run(lambda: client.export(exp_id))
    out = output or RunLayout(label).bundle_path()
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
    if v.tolerance_evidence_ok is not None:
        # C7 Inc 4: shown only when the bundle carries tolerance-consensus units
        # (their attested representative hashes recompute from the bundled data).
        click.echo(f"tolerance:   {_fmt(v.tolerance_evidence_ok)}")
    if v.prereg_bound_ok is not None:
        # D16.2: shown only for a pre-registered run.
        click.echo(f"pre-reg:     {_fmt(v.prereg_bound_ok)}")
        ordering = (
            "n/a (anchor pending — the hourly sweep)"
            if v.design_precedes_data_ok is None
            else ("ok" if v.design_precedes_data_ok else "FAIL")
        )
        click.echo(f"design<data: {ordering}")
    if v.deviations_disclosed:
        # D16.2-D: honest disclosure — deviations never fail a bundle; a record
        # that fails to VERIFY does.
        click.echo(
            f"deviations:  {v.deviations_disclosed} declared "
            f"({'records verify' if v.deviations_ok else 'RECORD FAILS TO VERIFY'})"
        )
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
        return TenantKey.load(key_path)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: failed to load key from {key_path}: {e}", err=True)
        sys.exit(1)


@experiment.command("deviate")
@click.argument("target")
@_coord_opt
@_key_opt
@click.option(
    "--what",
    "what_changed",
    required=True,
    help="What changed vs the pre-registered design (10-2000 chars).",
)
@click.option("--why", required=True, help="Why the analysis changed (10-2000 chars).")
def experiment_deviate(
    target: str, coordinator: str, key_path: Path, what_changed: str, why: str
) -> None:
    """Declare a DEVIATION from the pre-registered design (D16.2 §5).

    An append-only, tenant-SIGNED record — never an edit of the original. The
    coordinator anchors it in the public transparency log, so WHEN the analysis
    changed is provable. Exploratory analysis is allowed and valuable; declaring
    the deviation keeps it from masquerading as confirmatory. TARGET is an
    exp- id, a tenant label, or 'latest'."""
    from auspexai_tenant.experiment import Experiment

    key = _load_key(key_path)
    client = _make_client(coordinator, key_path)
    exp_id, label = _resolve_experiment(client, target)
    exp = Experiment(coordinator, key, exp_id)
    try:
        rec = _run(lambda: exp.declare_deviation(what_changed=what_changed, why=why))
    except CoordinatorError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    click.echo(f"deviation recorded for {label} ({exp_id}): {rec['deviation_id']}")
    click.echo(f"  declared_at: {rec['declared_at']}")
    click.echo(
        "  Rekor anchor: pending — lands on the hourly sweep; the record is already "
        "signed + immutable."
    )


@experiment.command("build")
@click.argument("pkg_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="experiment.toml path (default: next to PKG_DIR).",
)
@_profile_opt
@click.option(
    "--exact-label",
    is_flag=True,
    help="Use [experiment].label verbatim (no unique timestamp suffix).",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Manifest output path (default: PKG_DIR/manifest.json).",
)
def experiment_build(
    pkg_dir: Path,
    config_path: Path | None,
    profile: str | None,
    exact_label: bool,
    out_path: Path | None,
) -> None:
    """Assemble PKG_DIR/manifest.json from experiment.toml — the GENERIC build,
    no per-tenant build.py.

    Reads [experiment]/[executor]/[reducer]/[work_unit_source], computes the
    package digest over PKG_DIR, stamps a unique label (unless --exact-label),
    validates the manifest, and writes it. Then `experiment submit PKG_DIR`."""
    from auspexai_tenant.experiment_config import (
        load_experiment_config,
        make_unique_label,
        manifest_dict_from_config,
    )

    cfg = load_experiment_config(config_path or pkg_dir.parent, profile=profile)
    base = cfg.label
    if not base:
        click.echo(
            "ERROR: experiment.toml [experiment].label is required "
            "(looked next to PKG_DIR; pass --config to point elsewhere).",
            err=True,
        )
        sys.exit(1)
    label = base if exact_label else make_unique_label(base)
    digest = compute_package_digest(pkg_dir)
    try:
        m = manifest_dict_from_config(cfg, package_sha256=digest, label=label)
    except ValueError as e:  # a missing/empty required field, named by table+key
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    try:
        Manifest.model_validate(m)
    except ValidationError as e:  # ValidationError is-a ValueError → check it first
        click.echo(f"ERROR: the assembled manifest failed v0.1 validation:\n{e}", err=True)
        sys.exit(1)
    out = out_path or (pkg_dir / "manifest.json")
    out.write_text(json.dumps(m, indent=2) + "\n")
    click.echo(f"manifest:   {out}")
    click.echo(f"label:      {label}")
    if cfg.active_profile:
        click.echo(f"profile:    {cfg.active_profile}")
    click.echo(f"package:    {digest[:16]}…")
    click.echo(f"next:       auspexai-tenant experiment submit {pkg_dir} --key <key>")


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


def _record_benchmark_declaration(cfg, experiment_id: str, label: str) -> None:
    """Write the run's benchmark declaration beside it (runs/<label>/
    benchmark_reference.json) when the config declares one. The declaration is
    made at LAUNCH — before any data exists (the pre-registration posture) — and
    is what lets the dashboard materialize the score automatically instead of
    asking the researcher to choose a comparison after the fact. Best-effort:
    a write failure never fails the launch."""
    try:
        ref = cfg.benchmark_reference
    except ValueError as e:
        click.echo(f"WARNING: [benchmark] declaration ignored: {e}", err=True)
        return
    if not ref:
        return
    if ref == experiment_id:
        click.echo("WARNING: [benchmark].reference is this run itself — ignored.", err=True)
        return
    try:
        from datetime import UTC, datetime

        from auspexai_tenant.runs import runs_base

        layout = RunLayout(label, base=runs_base(cfg.runs_dir))
        layout.dir.mkdir(parents=True, exist_ok=True)
        (layout.dir / "benchmark_reference.json").write_text(
            json.dumps(
                {
                    "schema": "auspexai-benchmark-declaration/v0",
                    "mode": "fixed_reference",
                    "experiment_id": experiment_id,
                    "label": label,
                    "reference_experiment_id": ref,
                    "declared_at": datetime.now(UTC).isoformat(),
                    "source": "experiment_config"
                    + (f":profiles.{cfg.active_profile}" if cfg.active_profile else ""),
                },
                indent=2,
            )
        )
        click.echo(f"benchmark declared: will score against {ref} (recorded beside the run)")
    except (OSError, ValueError) as e:
        click.echo(f"WARNING: could not record the benchmark declaration: {e}", err=True)


def _auto_benchmark(client, label: str, runs_dir: str | None = None) -> None:
    """Score the finished run against its DECLARED reference, automatically —
    the tail of `launch`/`run`: with `[benchmark] reference` in the config, no
    flag or extra command establishes the benchmark; it exists when the run
    does. Reads the declaration recorded at launch beside the run, exports +
    verifies both bundles, scores, persists the same report the dashboard
    reads (runs/<label>/benchmark_vs_<ref>.json). Best-effort: a benchmark
    failure never fails the run — it says why and moves on (the dashboard's
    first-view materialization is the retry)."""
    from auspexai_tenant.runs import runs_base

    try:
        layout = RunLayout(label, base=runs_base(runs_dir))
    except ValueError:
        return
    decl_path = layout.dir / "benchmark_reference.json"
    if not decl_path.exists():
        return
    try:
        decl = json.loads(decl_path.read_text())
        ref = str(decl["reference_experiment_id"])
        exp_id = str(decl.get("experiment_id") or "")
    except (OSError, ValueError, KeyError):
        click.echo("WARNING: unreadable benchmark declaration — skipping the auto-score.", err=True)
        return
    out_path = layout.dir / f"benchmark_vs_{ref}.json"
    if out_path.exists():
        return  # a resumed run was already scored
    click.echo(f"benchmark: scoring against {ref} …")
    try:
        from datetime import UTC, datetime

        from auspexai_tenant.benchmark import drift_benchmark_bundles
        from auspexai_tenant.evidence import verify_bundle

        bundle = client.export(exp_id)
        ref_bundle = client.export(ref)
        for side, blob in (("observation", bundle), ("reference", ref_bundle)):
            if not verify_bundle(blob).ok:
                click.echo(
                    f"WARNING: the {side} bundle failed verification — benchmark skipped "
                    "(only custody-verified evidence is scored).",
                    err=True,
                )
                return
        report = drift_benchmark_bundles(bundle, ref_bundle)
        record = {
            "schema": "auspexai-drift-benchmark-report/v0",
            "computed_at": datetime.now(UTC).isoformat(),
            "observation": {"experiment_id": exp_id, "label": label},
            "reference": {"experiment_id": ref, "label": decl.get("reference_label") or ref},
            "report": report.to_dict(),
        }
        layout.dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        peak = f"{report.peak_eu:.2f} EU" if report.peak_eu is not None else "n/a"
        breadth = f"{report.breadth:.0%}" if report.breadth is not None else "n/a"
        click.echo(f"benchmark: peak {peak} · breadth {breadth} · saved {out_path}")
    except (CoordinatorError, httpx.RequestError, OSError, ValueError) as e:
        click.echo(
            f"WARNING: could not auto-score ({e}); the dashboard scores it on first view, "
            f"or: auspexai-tenant benchmark drift",
            err=True,
        )


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


_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0}
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)([hms])", re.IGNORECASE)


def parse_duration(s: str) -> float:
    """Parse a wall-clock duration into seconds.

    Accepts:
      * unit strings — "4h", "90m", "30s", or compound "1h30m" / "2h15m30s"
        (each unit at most once, descending h > m > s);
      * a clock string "HH:MM:SS" (or "MM:SS");
      * a bare number ("3600", "1.5") = seconds.

    Raises `click.BadParameter` on anything malformed (units out of order, an
    unknown suffix, a negative value, trailing junk, etc.)."""
    raw = s.strip()
    if not raw:
        raise click.BadParameter("empty duration")
    lowered = raw.lower()

    # bare number = seconds
    try:
        seconds = float(lowered)
    except ValueError:
        pass
    else:
        if seconds < 0:
            raise click.BadParameter(f"duration cannot be negative: {s!r}")
        return seconds

    # HH:MM:SS / MM:SS clock form
    if ":" in lowered:
        parts = lowered.split(":")
        if len(parts) not in (2, 3) or any(p == "" for p in parts):
            raise click.BadParameter(f"invalid clock duration: {s!r} (use HH:MM:SS or MM:SS)")
        try:
            nums = [float(p) for p in parts]
        except ValueError as e:
            raise click.BadParameter(f"invalid clock duration: {s!r}") from e
        if any(n < 0 for n in nums):
            raise click.BadParameter(f"duration cannot be negative: {s!r}")
        total = 0.0
        for n in nums:
            total = total * 60 + n
        return total

    # unit form: 4h / 90m / 30s / 1h30m — each unit once, descending order
    matches = list(_DURATION_TOKEN.finditer(lowered))
    consumed = "".join(m.group(0) for m in matches)
    if not matches or consumed != lowered:
        raise click.BadParameter(
            f"invalid duration: {s!r} (use e.g. 4h, 90m, 1h30m, HH:MM:SS, or a number of seconds)"
        )
    order = "hms"
    last_idx = -1
    total = 0.0
    for m in matches:
        unit = m.group(2).lower()
        idx = order.index(unit)
        if idx <= last_idx:
            raise click.BadParameter(
                f"invalid duration: {s!r} (units must be h>m>s, each at most once)"
            )
        last_idx = idx
        total += float(m.group(1)) * _DURATION_UNITS[unit]
    return total


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
@_profile_opt
@click.option(
    "--doorbell",
    is_flag=True,
    help="Also subscribe to the SSE event stream and poll immediately on a relevant "
    "event (the timer floor still guarantees liveness).",
)
@click.option(
    "--duration-cap",
    "duration_cap",
    default=None,
    help="Wall-clock cap (e.g. 4h, 90m, 1h30m, HH:MM:SS); the run stops + finalizes "
    "at the cap, like max_rounds.",
)
@click.option(
    "--resumable",
    is_flag=True,
    help="On Ctrl-C, leave the experiment running server-side (resume later with "
    "`experiment run <label>`). Default: Ctrl-C aborts the run cleanly.",
)
@_coord_opt
@_key_opt
def experiment_run(
    target: str,
    driver_spec: str | None,
    journal_path: Path | None,
    config_path: Path | None,
    profile: str | None,
    doorbell: bool,
    duration_cap: str | None,
    resumable: bool,
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
    from auspexai_tenant.experiment import Experiment, LifecycleConflictError
    from auspexai_tenant.experiment_config import load_experiment_config
    from auspexai_tenant.wake import SseWake, sse_line_source

    cfg = load_experiment_config(config_path, profile=profile)
    if profile is None and cfg.available_profiles:
        # D21c: the 2026-07-04 footgun — a no-profile resume silently took the
        # DEFAULT driver cadence + duration (4x rounds, short clock).
        click.echo(
            "WARNING: no --profile given — driver settings (cadence, duration, unit "
            "prefix) come from the TOP-LEVEL config. If this run was launched with a "
            f"profile, resume with it (available: {', '.join(cfg.available_profiles)}).",
            err=True,
        )
    # [driver].path → on sys.path so the entrypoint module (e.g. a `driver/` subdir)
    # imports regardless of cwd; lets `run`/`launch` work from the repo root with no
    # `cd`. Resolved relative to the experiment.toml the walk-up found.
    if cfg.driver_path and cfg.source_path is not None:
        driver_dir = str((cfg.source_path.parent / cfg.driver_path).resolve())
        if driver_dir not in sys.path:
            sys.path.insert(0, driver_dir)
    driver_spec = driver_spec or cfg.driver_entrypoint
    if not driver_spec:
        click.echo(
            "ERROR: no driver — pass --driver module:attr or set [driver].entrypoint "
            "in experiment.toml.",
            err=True,
        )
        sys.exit(1)
    spec = _load_attr(driver_spec)(cfg)
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
    if cfg.active_profile:
        click.echo(f"profile:    {cfg.active_profile}")
    exp = Experiment(coordinator, key, experiment_id)
    wake = spec.wake
    if wake is None and doorbell:
        wake = SseWake(sse_line_source(coordinator, key, experiment_id))
    # D22-A: `run` (a resume, or the `launch`→`run` delegation) can land on a
    # still-`submitted` experiment — a maintainer approves minutes-to-hours after
    # launch. Submitting into `submitted` 409s `experiment_not_open_for_submissions`
    # and, pre-fix, killed the driver (the weekend-campaign resume incident). Wait
    # for approval FIRST; an already-approved experiment returns immediately.
    # --doorbell (an SSE source whose events include `experiment.status`) reacts to
    # approval instantly; otherwise a tight poll. Ctrl-C here leaves it submitted
    # (resume semantics — never abort an experiment we're only resuming).
    approval_wake = SseWake(sse_line_source(coordinator, key, experiment_id)) if doorbell else None
    try:
        _wait_for_approval(
            client,
            experiment_id,
            _APPROVAL_POLL_SECONDS,
            resumable=True,
            wake=approval_wake,
        )
    except KeyboardInterrupt:
        click.echo(
            f"\ninterrupted while waiting for approval — {experiment_id} left submitted; "
            f"resume with `auspexai-tenant experiment run {label}`.",
            err=True,
        )
        sys.exit(130)
    # CLI flag wins over a spec-level cap; either may be unset.
    parsed_cap = parse_duration(duration_cap) if duration_cap is not None else None
    duration_cap_seconds = (
        parsed_cap if parsed_cap is not None else getattr(spec, "duration_cap_seconds", None)
    )
    try:
        result = _run(
            lambda: run_until(
                exp,
                condition=spec.condition,
                next_batch=spec.next_batch,
                reduce=spec.reduce,
                journal=journal_path,
                wake=wake,
                stall=spec.stall,
                include=spec.include,
                max_rounds=spec.max_rounds,
                duration_cap_seconds=duration_cap_seconds,
            )
        )
    except KeyboardInterrupt:
        # D14 (client→server coupling): Ctrl-C kills the run as everyone expects.
        # By default that aborts the experiment server-side too, so a killed launch
        # never leaves a silent orphan that auto-approves and sits idle. --resumable
        # opts OUT (leave it running; resume with `experiment run <label>`).
        if resumable:
            click.echo(
                f"\ninterrupted — {experiment_id} left running server-side; "
                f"resume with `auspexai-tenant experiment run {label}`.",
                err=True,
            )
            sys.exit(130)
        click.echo(f"\ninterrupted — aborting {experiment_id} …", err=True)
        try:
            exp.abort()
            click.echo(f"aborted {experiment_id}.", err=True)
        except LifecycleConflictError:
            click.echo(f"{experiment_id} was already terminal.", err=True)
        except (CoordinatorError, httpx.RequestError) as e:
            click.echo(
                f"WARNING: couldn't abort {experiment_id} ({e}); abort it from the dashboard.",
                err=True,
            )
        sys.exit(130)
    except (httpx.RequestError, CoordinatorError) as e:
        # D18 (network→client resilience): a transient tunnel failure that OUTLASTED
        # the retry budget (retry.call_with_retry), or another coordinator error,
        # killed the driver loop. Never a silent orphan — the run is still `approved`
        # server-side and resumable from the journal's last round checkpoint. Surface
        # the explicit choice LOUDLY + non-zero (symmetric with the D14 Ctrl-C path);
        # C16's E14 detector is the backstop that also flags the abandoned run.
        click.echo(
            f"\ndriver stopped — coordinator unreachable or erroring ({e}).\n"
            f"{experiment_id} is still running server-side (NOT aborted). Once it is "
            f"reachable again:\n"
            f"  resume:  auspexai-tenant experiment run {label}\n"
            f"  abort:   auspexai-tenant experiment abort {label}   (or via the dashboard)",
            err=True,
        )
        sys.exit(1)
    click.echo(f"outcome:  {result.outcome}")
    click.echo(f"rounds:   {result.rounds}")
    if result.outcome == "time_capped":
        click.echo(
            f"stopped: duration cap reached after {result.rounds} rounds; experiment finalized."
        )
    click.echo("aggregate:")
    click.echo(json.dumps(result.aggregate, indent=2, default=str))
    if result.attestation is not None:
        click.echo(f"attestation merkle_root: {result.attestation.merkle_root}")
    _auto_benchmark(client, label, runs_dir=cfg.runs_dir)


# D22-A: how often `_wait_for_approval` re-checks status when there is no
# doorbell. A maintainer approves minutes-to-hours after launch, so a tight poll
# is cheap and the wait itself is unbounded (the run is resumable meanwhile).
_APPROVAL_POLL_SECONDS = 5.0


def _wait_for_approval(
    client, experiment_id: str, poll_interval: float, *, resumable: bool = False, wake=None
) -> str:
    """Block until the experiment leaves the pending (`submitted`) state — the
    driver can only submit work units once it is approved/paused, so BOTH `launch`
    and `run` wait here rather than 409-ing the drive into an orphaned experiment
    (D22-A: a resume can land on a still-`submitted` experiment when the maintainer
    approves minutes-to-hours after launch). Returns the go-status; exits on a
    terminal status. A Ctrl-C here propagates to the caller, whose handler decides
    abort (launch's D14 semantics) vs leave-submitted (run's resume semantics);
    `resumable` only selects the announced hint. When `wake` (a `--doorbell` SSE
    source, whose events include `experiment.status`) is given, react to the
    approval event instantly, with `poll_interval` as the liveness floor."""
    waiting = {"submitted", "pending", "draft"}
    terminal = {"completed", "aborted", "failed", "expired", "rejected", "declined"}
    announced = False
    while True:
        try:
            status = (client.get_experiment(experiment_id) or {}).get("status")
        except (CoordinatorError, httpx.RequestError) as e:
            click.echo(f"ERROR: could not poll experiment status: {e}", err=True)
            sys.exit(1)
        if status in terminal:
            click.echo(f"ERROR: experiment is {status} — nothing to drive.", err=True)
            sys.exit(1)
        if status not in waiting:
            return status or "approved"
        if not announced:
            hint = (
                "Ctrl-C leaves it submitted; drive it after approval with `experiment run latest`"
                if resumable
                else "Ctrl-C aborts it"
            )
            click.echo(
                "Waiting for a maintainer to approve it — approve at "
                f"https://ops.auspexai.network ({hint})."
            )
            announced = True
        if wake is not None:
            wake.wait()  # SSE approval event or the floor, whichever first
        else:
            time.sleep(poll_interval)


@experiment.command("launch")
@click.argument(
    "pkg_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="experiment.toml path (default: walk up from cwd).",
)
@_profile_opt
@click.option(
    "--exact-label", is_flag=True, help="Use [experiment].label verbatim (no timestamp suffix)."
)
@click.option(
    "--driver",
    "driver_spec",
    default=None,
    help="Driver 'module:attr' (default: [driver].entrypoint).",
)
@click.option(
    "--journal",
    "journal_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Run-journal path (default: [driver].journal).",
)
@click.option(
    "--no-drive",
    is_flag=True,
    help="Build + submit only; skip driving (approve, then `experiment run latest`).",
)
@click.option(
    "--poll-interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Seconds between status polls while waiting for the approval gate.",
)
@click.option(
    "--duration-cap",
    "duration_cap",
    default=None,
    help="Wall-clock cap (e.g. 4h, 90m, 1h30m, HH:MM:SS); the run stops + finalizes "
    "at the cap, like max_rounds.",
)
@click.option(
    "--resumable",
    is_flag=True,
    help="On Ctrl-C, leave the experiment running server-side (resume later with "
    "`experiment run latest`). Default: Ctrl-C aborts the run cleanly.",
)
@_coord_opt
@_key_opt
@click.pass_context
def experiment_launch(
    ctx: click.Context,
    pkg_dir: Path | None,
    config_path: Path | None,
    profile: str | None,
    exact_label: bool,
    driver_spec: str | None,
    journal_path: Path | None,
    no_drive: bool,
    poll_interval: float,
    duration_cap: str | None,
    coordinator: str,
    key_path: Path,
    resumable: bool,
) -> None:
    """Build + submit + drive in ONE command — the whole experiment lifecycle.

    Equivalent to `experiment build PKG_DIR` → `experiment submit PKG_DIR` →
    `experiment run latest`, but with no ceremony: experiment.toml is found by
    walking up from the cwd, the driver resolves via [driver].path (no `cd`, no
    `--config`), and the signing key comes from [experiment].key_path when --key
    is unset (so no repeated --key). After submit, a maintainer approves the
    experiment in the operator console; driving begins immediately and proceeds
    automatically on approval (Ctrl-C aborts the run cleanly — including the
    server-side experiment; pass --resumable to instead leave it running and
    resume later with `experiment run latest`).
    PKG_DIR defaults to the `pkg/` next to experiment.toml — so plain `launch`
    works from anywhere in the repo."""
    from auspexai_tenant.experiment_config import load_experiment_config

    # Validate the cap up front so a typo fails before the build/submit work.
    if duration_cap is not None:
        parse_duration(duration_cap)
    cfg = load_experiment_config(config_path, profile=profile)
    # Resolve the package dir from the experiment.toml the walk-up found, so plain
    # `launch` works from anywhere in the repo — not only the dir that holds pkg/.
    if pkg_dir is None:
        if cfg.source_path is None:
            click.echo(
                "ERROR: no experiment.toml found (walked up from the current dir). "
                "cd into your tenant repo, or pass PKG_DIR.",
                err=True,
            )
            sys.exit(1)
        pkg_dir = cfg.source_path.parent / "pkg"
    if not pkg_dir.is_dir():
        if str(pkg_dir) in (cfg.available_profiles or []):
            click.echo(
                f"ERROR: '{pkg_dir}' is a PROFILE, not a package dir — you want:\n"
                f"  auspexai-tenant experiment launch --profile {pkg_dir}",
                err=True,
            )
            sys.exit(1)
        click.echo(f"ERROR: package dir not found: {pkg_dir} — pass PKG_DIR explicitly.", err=True)
        sys.exit(1)
    # Key precedence: explicit --key > [experiment].key_path > the default key path.
    if key_path == DEFAULT_KEY_PATH and cfg.key_path:
        key_path = Path(cfg.key_path).expanduser()

    # Per-invocation manifest scratch (D21b): concurrent launches interleaved
    # writes into the shared pkg/manifest.json (the 2026-07-03 corrupted-JSON
    # race). Each launch builds+submits its OWN file, then cleans it up.
    scratch_name = f"manifest.launch-{os.getpid()}.json"
    ctx.invoke(
        experiment_build,
        pkg_dir=pkg_dir,
        config_path=config_path,
        profile=profile,
        exact_label=exact_label,
        out_path=pkg_dir / scratch_name,
    )
    try:
        ctx.invoke(
            experiment_submit,
            pkg_dir=pkg_dir,
            manifest_name=scratch_name,
            coordinator=coordinator,
            key_path=key_path,
        )
    finally:
        for leftover in (pkg_dir / scratch_name, pkg_dir / (scratch_name + ".sig")):
            leftover.unlink(missing_ok=True)
    client = _make_client(coordinator, key_path)
    experiment_id, label = _resolve_experiment(client, "latest")
    _record_benchmark_declaration(cfg, experiment_id, label)
    if no_drive:
        click.echo("\nbuilt + submitted. Approve it, then: auspexai-tenant experiment run latest")
        return

    click.echo("")
    click.echo(f"submitted {label}  ({experiment_id})")
    # Wait for the maintainer-approval gate before driving: the driver can only
    # submit work units once the experiment is approved/paused (a 409 otherwise).
    try:
        status = _wait_for_approval(client, experiment_id, poll_interval, resumable=resumable)
    except KeyboardInterrupt:
        # D14 tail: a Ctrl-C in the submit→approval window previously exited with
        # the experiment still submitted server-side, which then auto-approved as
        # a driverless orphan (exp-oK4PrkRP). Same semantics as the drive-loop
        # handler below: abort by default, --resumable opts out. SUBMITTED →
        # ABORTED is a valid researcher transition, so this never 409s in the
        # normal case.
        from auspexai_tenant.experiment import Experiment, LifecycleConflictError

        if resumable:
            click.echo(
                f"\ninterrupted — {experiment_id} left submitted server-side; once "
                f"approved, drive it with `auspexai-tenant experiment run {label}`.",
                err=True,
            )
            sys.exit(130)
        click.echo(f"\ninterrupted — aborting {experiment_id} …", err=True)
        try:
            Experiment(coordinator, _load_key(key_path), experiment_id).abort()
            click.echo(f"aborted {experiment_id}.", err=True)
        except LifecycleConflictError:
            click.echo(f"{experiment_id} was already terminal.", err=True)
        except (CoordinatorError, httpx.RequestError) as e:
            click.echo(
                f"WARNING: couldn't abort {experiment_id} ({e}); abort it from the dashboard.",
                err=True,
            )
        sys.exit(130)
    click.echo(f"approved ({status}) — driving.")
    ctx.invoke(
        experiment_run,
        target="latest",
        driver_spec=driver_spec,
        journal_path=journal_path,
        config_path=config_path,
        profile=profile,
        doorbell=True,
        duration_cap=duration_cap,
        resumable=resumable,
        coordinator=coordinator,
        key_path=key_path,
    )


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
    """Browse the network model catalog (demand-board, §9 #39)."""


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


# ----------------------------------------------------------------------------
# the Drift Benchmark (D16.4) — one comparable measurement of behavioral drift
# ----------------------------------------------------------------------------


@main.group()
def benchmark() -> None:
    """The Drift Benchmark — envelope-normalized drift, comparable across runs,
    models, and configurations (see docs/analyzing_your_results.md)."""


@benchmark.command("publish")
@click.argument("target")
@click.option(
    "--reference",
    "reference_id",
    default=None,
    help="The baseline to score/publish against (default: the run's declared "
    "reference, or its only saved report).",
)
@click.option(
    "--no-submit", is_flag=True, help="Sign + write the entry only; skip board submission."
)
@_coord_opt
@_key_opt
def benchmark_publish(
    target: str, reference_id: str | None, no_submit: bool, coordinator: str, key_path: Path
) -> None:
    """Publish a Drift-Benchmark result as a SIGNED registry entry.

    Publishing is a deliberate act (G5, drift_benchmark_design.md §6): this
    exports + custody-verifies BOTH bundles, scores the run if no saved report
    exists yet (historical runs publish in one step), then signs a claim
    binding the score to the two experiments' attestation anchors with YOUR
    tenant key, and SUBMITS it to the public board automatically (a courier
    Worker opens the website PR; CI's grounded admission rule verifies and
    auto-merges — machines admit, no curator). --no-submit writes the signed
    entry file only."""
    if "/" in target:
        click.echo(
            f"ERROR: '{target}' looks like a PATH — pass the experiment LABEL or "
            "exp- id instead (see: auspexai-tenant experiment list).",
            err=True,
        )
        sys.exit(1)
    from auspexai_tenant.benchmark import drift_benchmark_bundles
    from auspexai_tenant.benchmark_entry import build_entry, verify_entry
    from auspexai_tenant.evidence import verify_bundle
    from auspexai_tenant.runs import RunLayout, runs_base

    key = _load_key(key_path)
    client = _make_client(coordinator, key_path)
    exp_id, label = _resolve_experiment(client, target)
    # The same base every other surface resolves: [runs].dir (config walk-up)
    # > $AUSPEXAI_RUNS_DIR > ./runs-if-exists > the stable per-user base.
    try:
        from auspexai_tenant.experiment_config import load_experiment_config

        cfg_runs_dir = load_experiment_config(None).runs_dir
    except (ValueError, OSError):
        cfg_runs_dir = None
    layout = RunLayout(label, base=runs_base(cfg_runs_dir))

    # Resolve the reference: --reference > a single saved report > the
    # declaration recorded at launch.
    record = None
    reports = sorted(layout.dir.glob("benchmark_vs_*.json"))
    if reference_id:
        matching = [p for p in reports if p.stem == f"benchmark_vs_{reference_id}"]
        record = json.loads(matching[0].read_text()) if matching else None
    elif len(reports) == 1:
        record = json.loads(reports[0].read_text())
        reference_id = record["reference"]["experiment_id"]
    elif len(reports) > 1:
        click.echo(
            "ERROR: multiple saved reports — pick one with --reference "
            f"({', '.join(p.stem.removeprefix('benchmark_vs_') for p in reports)})",
            err=True,
        )
        sys.exit(1)
    if reference_id is None:
        decl_path = layout.dir / "benchmark_reference.json"
        if decl_path.exists():
            try:
                reference_id = str(json.loads(decl_path.read_text())["reference_experiment_id"])
            except (ValueError, KeyError, OSError):
                pass
    if reference_id is None:
        click.echo(
            "ERROR: no saved report and no declared reference for this run — "
            "pass --reference <experiment-id>.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"collecting + verifying both bundles ({exp_id}, {reference_id}) …")
    obs_bundle = _run(lambda: client.export(exp_id))
    ref_bundle = _run(lambda: client.export(reference_id))
    for side, blob in (("observation", obs_bundle), ("reference", ref_bundle)):
        if not verify_bundle(blob).ok:
            click.echo(
                f"ERROR: the {side} bundle failed verification — refusing to sign.", err=True
            )
            sys.exit(1)

    if record is None:
        # Historical / never-scored run: score it right here (the bundles are
        # already in hand and verified) and persist the same report every other
        # surface reads.
        from datetime import UTC, datetime

        report = drift_benchmark_bundles(obs_bundle, ref_bundle)
        record = {
            "schema": "auspexai-drift-benchmark-report/v0",
            "computed_at": datetime.now(UTC).isoformat(),
            "observation": {"experiment_id": exp_id, "label": label},
            "reference": {"experiment_id": reference_id, "label": reference_id},
            "report": report.to_dict(),
        }
        layout.dir.mkdir(parents=True, exist_ok=True)
        (layout.dir / f"benchmark_vs_{reference_id}.json").write_text(json.dumps(record, indent=2))
        click.echo(f"scored: peak {report.peak_eu} EU (report saved beside the run)")

    # G6 (ratified): request the coordinator's publication authorization —
    # the R1+ standing gate, the audit row, and the signed block the board CI
    # verifies. An older coordinator (endpoint absent) publishes ungoverned
    # until the flag-day; a standing refusal is terminal and explains itself.
    authorization = None
    rep = record.get("report") or {}
    try:
        from auspexai_tenant.experiment import Experiment

        resp = Experiment(coordinator, key, exp_id)._post(
            f"/api/v0/experiments/{exp_id}/actions/authorize-benchmark-publication",
            json={
                "reference_experiment_id": reference_id,
                "peak_eu": rep.get("peak_eu"),
                "breadth": rep.get("breadth"),
                "byte_divergence_rate": rep.get("byte_divergence_rate"),
            },
        )
        if resp.status_code == 200:
            authorization = resp.json()["authorization"]
            click.echo(
                f"publication authorized (standing R{authorization.get('standing_at_issue')}, "
                "audited coordinator-side)"
            )
        elif resp.status_code == 403:
            detail = resp.json().get("detail", {}).get("error", {})
            click.echo(f"ERROR: {detail.get('message', 'publication refused')}", err=True)
            sys.exit(1)
        elif resp.status_code in (404, 405):
            click.echo(
                "WARNING: coordinator predates publication governance — publishing "
                "without an authorization block.",
                err=True,
            )
        else:
            click.echo(
                f"WARNING: authorization request failed (HTTP {resp.status_code}) — "
                "publishing without an authorization block.",
                err=True,
            )
    except Exception as e:
        click.echo(f"WARNING: authorization request failed ({e}) — continuing.", err=True)

    entry = build_entry(
        record=record,
        observation_bundle=obs_bundle,
        reference_bundle=ref_bundle,
        tenant_id=(obs_bundle.get("manifest") or {}).get("tenant_id"),
        key=key,
        authorization=authorization,
    )
    payload = verify_entry(entry)
    assert payload is not None
    out = layout.dir / f"benchmark_entry_{reference_id}.json"
    out.write_text(json.dumps(entry, indent=2))
    peak = payload["report"]["peak_eu"]
    click.echo(f"signed entry: peak {peak} EU vs {reference_id}")
    click.echo(f"written: {out}")
    if no_submit:
        return
    # One-command publish: the entry is self-grounding, so submission is a
    # dumb courier (a Worker opens the website PR; CI runs the grounded
    # admission rule and merges only on green — machines admit, no curator).
    submit_url = os.environ.get(
        "AUSPEXAI_BOARD_SUBMIT_URL", "https://board.auspexai.network/submit"
    )
    click.echo(f"submitting to the board ({submit_url}) …")
    try:
        resp = httpx.post(submit_url, json=entry, timeout=30)
        body = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if resp.status_code == 200 and body.get("status") == "submitted":
            click.echo(f"submitted — admission PR: {body.get('pr')}")
            click.echo(
                "CI verifies the entry's grounding and merges on green; the board updates itself."
            )
        elif body.get("status") == "already-submitted":
            click.echo("already submitted (identical entry) — nothing to do.")
        else:
            raise httpx.HTTPError(f"HTTP {resp.status_code}")
    except (httpx.HTTPError, ValueError) as e:
        click.echo(
            f"WARNING: submission failed ({e}). The signed entry is safe at {out} — "
            "submit it as a PR adding the file under entries/ in auspexai/website.",
            err=True,
        )


@benchmark.command("verify-entry")
@click.argument("entry_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def benchmark_verify_entry(entry_file: Path) -> None:
    """Re-verify any published entry's signature + shape (anyone can run this)."""
    from auspexai_tenant.benchmark_entry import verify_entry

    entry = json.loads(entry_file.read_text())
    payload = verify_entry(entry)
    if payload is not None:
        click.echo(
            f"OK: signature valid (publisher {payload.get('publisher_pubkey_hex', '')[:16]}…, "
            f"peak {payload.get('report', {}).get('peak_eu')} EU vs "
            f"{payload.get('reference', {}).get('experiment_id')})"
        )
    else:
        click.echo("FAILED: signature invalid or entry malformed", err=True)
        sys.exit(1)


@benchmark.command("drift")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--key",
    "key_feature",
    default="probe_id",
    show_default=True,
    help="The dotted feature path observations are joined on (a role=key feature).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
@click.option(
    "--include-diverged",
    is_flag=True,
    help="Also score diverged/outlier payloads (forensics — they failed the run's "
    "own integrity predicate and stay out of the scalar by default).",
)
@click.option(
    "--plot",
    "plot_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also render the ladder PNG here (needs the [analysis] extra).",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip verify_bundle on the two inputs (offline scoring of unverified data).",
)
def benchmark_drift(
    bundle: Path,
    reference: Path,
    key_feature: str,
    include_diverged: bool,
    as_json: bool,
    plot_path: Path | None,
    no_verify: bool,
) -> None:
    """Score BUNDLE's behavior against REFERENCE in envelope units (EU).

    Both arguments are exported evidence bundles (`experiment export` /
    the dashboard's Download bundle). EU normalizes every declared scalar
    comparison by its own calibrated envelope, so orthogonal features —
    and different models/configurations — land on ONE comparable scale:
    <1 EU is within calibrated noise; >=1 EU is drift. Byte-level
    divergence is reported as a separate overlay, never folded into the
    scalar. The envelope comes from the REFERENCE bundle's signed manifest."""
    from auspexai_tenant.benchmark import (
        _load_bundle_file,
        drift_benchmark_bundles,
        format_report,
    )
    from auspexai_tenant.evidence import verify_bundle

    data = _load_bundle_file(str(bundle))
    ref = _load_bundle_file(str(reference))
    if not no_verify:
        for label, blob in (("bundle", data), ("reference", ref)):
            verification = verify_bundle(blob)
            if not verification.ok:
                click.echo(
                    f"ERROR: {label} failed verification — score it anyway with "
                    "--no-verify if you understand what that means.",
                    err=True,
                )
                sys.exit(1)
    report = drift_benchmark_bundles(
        data, ref, key_feature=key_feature, include_diverged=include_diverged
    )
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        click.echo(format_report(report))
    if plot_path is not None:
        from auspexai_tenant.benchmark_plot import plot_report

        obs_id = (data.get("manifest") or {}).get("experiment_id") or bundle.stem
        ref_id = (ref.get("manifest") or {}).get("experiment_id") or reference.stem
        plot_report(
            report,
            str(plot_path),
            title=f"Drift Benchmark — {obs_id}",
            subtitle=None,
        )
        click.echo(f"plot: {plot_path}  (reference: {ref_id})")


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


def _ensure_key(key_path: Path) -> tuple[TenantKey, bool]:
    """Load the tenant key at `key_path`, generating one if missing.

    Returns (key, created). A corrupt existing file is an error (we never
    silently overwrite key material)."""
    if key_path.exists():
        return _load_key(key_path), False
    new_key = TenantKey.generate()
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
    existing = out.get("account_existing_tenants") or []
    if existing:
        # #6 multiplicity heads-up: multi-tenancy is allowed (review is the
        # gate), so this is a warning, not a failure — but a second tenant for
        # the same account should be deliberate, not an accidental re-apply.
        click.echo(
            f"  ⚠ your account already operates {len(existing)} tenant(s): {', '.join(existing)}",
            err=True,
        )
        click.echo(
            "    this application requests an ADDITIONAL tenant (a maintainer "
            "reviews it). If that was unintended, contact the maintainer before "
            "it is approved.",
            err=True,
        )
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
    if v.tolerance_evidence_ok is not None:
        # C7 Inc 4: shown only when the bundle carries tolerance-consensus units
        # (their attested representative hashes recompute from the bundled data).
        click.echo(f"tolerance:   {_fmt(v.tolerance_evidence_ok)}")
    if v.prereg_bound_ok is not None:
        # D16.2: shown only for a pre-registered run.
        click.echo(f"pre-reg:     {_fmt(v.prereg_bound_ok)}")
        ordering = (
            "n/a (anchor pending — the hourly sweep)"
            if v.design_precedes_data_ok is None
            else ("ok" if v.design_precedes_data_ok else "FAIL")
        )
        click.echo(f"design<data: {ordering}")
    if v.deviations_disclosed:
        # D16.2-D: honest disclosure — deviations never fail a bundle; a record
        # that fails to VERIFY does.
        click.echo(
            f"deviations:  {v.deviations_disclosed} declared "
            f"({'records verify' if v.deviations_ok else 'RECORD FAILS TO VERIFY'})"
        )
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


def _emit_data_dictionary(data: dict) -> None:
    """Print the bundle's declared feature_schema as a data dictionary (D16.1).
    `data` is an already-VERIFIED bundle dict (the manifest is bound to the signed
    hash), so this is the attested documentation for the output.* columns."""
    from auspexai_tenant.evidence import data_dictionary_rows

    fs = (data.get("manifest") or {}).get("feature_schema")
    if not fs:
        click.echo("(no feature_schema declared — pre-D16.1 bundle / not a v0.3 manifest)")
        return
    click.echo(f"Data dictionary — {len(fs)} declared feature(s), from the signed manifest:\n")
    for r in data_dictionary_rows(fs):
        tags = "/".join(t for t in (r["kind"], r["role"]) if t)
        extra = " · ".join(t for t in (r["unit"], r["bounds"]) if t)
        head = f"  {r['column']}  [{tags}]"
        click.echo(head + (f"  ({extra})" if extra else ""))
        if r["meaning"]:
            click.echo(f"      {r['meaning']}")
        if r["change_means"]:
            click.echo(f"      Δ a change means: {r['change_means']}")
    click.echo("")


@bundle.command("table")
@click.argument("bundle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output table; format by extension: .csv or .parquet. "
    "Optional if --data-dictionary is given.",
)
@click.option(
    "--data-dictionary",
    "data_dictionary",
    is_flag=True,
    help="Print the declared, signed feature_schema (column · kind · role · unit · "
    "bounds · meaning · what a change means) — the data dictionary for output.* columns (D16.1).",
)
@click.option(
    "--signer", "signers", multiple=True, help="Authorized signer pubkey hex (repeatable)."
)
@click.option("--check-rekor", is_flag=True, help="Also check Rekor inclusion online first.")
def bundle_table(
    bundle_file: Path,
    out_path: Path | None,
    data_dictionary: bool,
    signers: tuple[str, ...],
    check_rekor: bool,
) -> None:
    """VERIFY the bundle, then write its results as a flat table and/or print its
    data dictionary.

    One row per consensus result; work-unit inputs flatten to input.* columns
    and result payloads to output.* — ready for pandas, Excel, Tableau, or R.
    `--data-dictionary` prints the declared, signed feature_schema next to the
    data (and works without the analysis extra). Refuses anything from a bundle
    that fails verification (incl. the D16.1 manifest binding). The table needs
    the analysis extra: pip install 'auspexai-tenant[analysis]'."""
    from auspexai_tenant.evidence import (
        BundleVerificationError,
        _read_bundle,
        load_verified,
        verify_bundle,
    )

    if out_path is None and not data_dictionary:
        click.echo(
            "ERROR: pass --out to write the table and/or --data-dictionary to print the schema",
            err=True,
        )
        sys.exit(1)

    # Data-dictionary-only: verify (no pandas needed) + print the signed schema.
    if data_dictionary and out_path is None:
        data = _read_bundle(bundle_file)
        v = verify_bundle(data, authorized_signers=list(signers) or None, check_rekor=check_rekor)
        if not v.ok:
            click.echo(f"REFUSED: {BundleVerificationError(v)}", err=True)
            sys.exit(1)
        _emit_data_dictionary(data)
        return

    try:
        df = load_verified(
            bundle_file,
            authorized_signers=list(signers) or None,
            check_rekor=check_rekor,
        )
    except BundleVerificationError as e:
        click.echo(f"REFUSED: {e}", err=True)
        sys.exit(1)
    if data_dictionary:
        _emit_data_dictionary(_read_bundle(bundle_file))
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
