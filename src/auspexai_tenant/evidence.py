"""Evidence-bundle verification (EB-1, principles §9 #47).

The coordinator's `GET /experiments/{id}/results/export` returns the
**evidence bundle** (`schema: auspexai-evidence-bundle/v1`): the signed
manifest + work-unit INPUTS + consensus results + COSE receipts + the COSE
result-set attestation (with its Rekor anchor and, when captured, the
inclusion proof) + a signed proof-of-transfer. `verify_bundle` runs the full
chain a researcher needs for custody confidence, offline by default:

  1. proof-of-transfer signature — and (with `authorized_signers`) that the
     signing key is EXTERNALLY pinned, not merely the one the bundle claims;
  2. the attestation itself (COSE signature, root recompute over its units,
     signed-root match, optional online Rekor inclusion);
  3. root unification — the custody record signs the attestation's merkle
     root (`transfer.root_kind` != flat-v0), binding data ↔ custody ↔ Rekor;
  4. completeness — the delivered consensus set is EXACTLY the attested leaf
     set: nothing missing, nothing extra;
  5. input binding (v1 attestations) — every leaf's `unit_payload_sha256`
     recomputes from the bundled work-unit payloads: "result R was produced
     from parameters P" verified from bytes in hand;
  6. worker signatures — the only signatures in the chain NOT made by the
     coordinator's key (the worker daemon's `signing/result.py` convention),
     removing the coordinator from the trusted base for data authenticity.

Aged-off results (payload withheld) can't have their worker signature or
payload recomputed — they are counted as skipped, never as failures: the
receipt + semantic_hash still prove the unit ran.
"""

from __future__ import annotations

import hashlib
import json
import operator as _operator
from base64 import b64decode
from dataclasses import dataclass
from typing import Any

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from auspexai_tenant.attestation import (
    DEFAULT_REKOR_URL,
    RESULT_SET_ALGORITHM_V1,
    AttestationVerification,
    ResultSetAttestation,
    _cose_verify,
    unit_payload_sha256,
    verify_attestation,
)

EVIDENCE_BUNDLE_SCHEMA = "auspexai-evidence-bundle/v1"
FLAT_ROOT_KIND = "flat-v0"
BUNDLE_SCHEMA_V1 = "auspexai-evidence-bundle/v1"

# Published AuspexAI coordinator signing keys (the public network's). Each is
# Fulcio-attested in Rekor and listed in AUTHORIZED_SIGNERS.md. The SDK embeds
# them so verifying a bundle from the public network GROUNDS trust by default —
# no hex-hunt — while a bundle signed by a key not in this set reports
# "unpinned" (never a false pass; pass --signer for a private coordinator, or
# --no-pin to accept self-consistency only). Annual rotation per §5.16: retired
# keys STAY here (they verify historical bundles forever) and new ones are added.
KNOWN_PUBLIC_SIGNERS: tuple[str, ...] = (
    # 2026 — active. Rekor logIndex 1615064195; coord.auspexai.network.
    "13c3b143c995764663e1016668cb7d8d24f4497fdc18d3f24b54a9a7529df453",
)


@dataclass(frozen=True)
class WorkerSignatureReport:
    """Per-result worker-signature verification over the bundle's consensus
    rows. `failed` lists result_ids whose signature did NOT verify — any entry
    is a hard failure. Aged-off skips are never failures (no payload to
    recompute over). Missing-field skips are LENIENT only for pre-EB-1 bundles
    (no `schema` member — those coordinators didn't ship
    `worker_pubkey_hex`/`exit_code`); on a v1 bundle the members are
    guaranteed, so a missing field is indistinguishable from a strip-the-
    signature tamper and `strict` mode fails it — otherwise stripping the
    fields would pass verification vacuously (verified=0, failed=[])."""

    verified: int
    failed: list[str]
    skipped_aged_off: int
    skipped_missing_fields: int
    strict: bool = False

    @property
    def ok(self) -> bool:
        if self.failed:
            return False
        return not (self.strict and self.skipped_missing_fields)


@dataclass(frozen=True)
class BundleVerification:
    """The named checks of `verify_bundle`. Tri-state fields are None when the
    check was not applicable (no attestation in the bundle / flat-root custody
    / v0 attestation without input binding) — None never fails `ok`; an
    explicit False always does."""

    transfer_signature_valid: bool
    transfer_signer_authorized: bool
    attestation: AttestationVerification | None
    root_unified: bool | None
    completeness_ok: bool | None
    inputs_bound_ok: bool | None
    worker_signatures: WorkerSignatureReport
    # Firewall #2: the footprint's recomputable half (integrity_basis counts) must
    # match a fresh recount from the signed predicate's per-unit basis. None when
    # the attestation carries no footprint (pre-firewall); False = the asserted
    # counts diverge from the signed unit list (tamper or coordinator bug).
    footprint_ok: bool | None = None
    # D16.1: the embedded manifest body recomputes to the SIGNED manifest_hash
    # (the hash the transfer record covers). None when the bundle carries no
    # manifest or no manifest_hash (pre-D16.1 / older coordinators); False = the
    # embedded manifest was swapped (so its feature_schema / models / etc. are
    # NOT the attested ones). Lets load_verified surface the feature_schema as the
    # bound, signed data dictionary rather than an out-of-band copy.
    manifest_bound_ok: bool | None = None
    # The coordinator-asserted governance footprint, surfaced for the researcher
    # to correct for apparatus influence (None on pre-firewall attestations).
    governance_footprint: dict[str, Any] | None = None
    # D16.2: the pre-registration binding — the bundle's submit-time anchor
    # blob COSE-verifies, its signed predicate binds THIS bundle's manifest hash,
    # and the design it anchors equals the embedded manifest's pre_registration
    # (with manifest_bound_ok, the reported analysis is provably the declared
    # one). None when the bundle carries no pre_registration (exploratory runs /
    # pre-D16.2); False = tamper or a swapped design.
    prereg_bound_ok: bool | None = None
    # D16.2: `design ≺ data` — the pre-registration's Rekor anchor precedes the
    # result attestation's. None until BOTH carry real anchors (the hourly
    # backfill may not have run yet — pending is never a failure); False = the
    # design was anchored AFTER the results (retroactive pre-registration).
    design_precedes_data_ok: bool | None = None
    # D16.2-D: deviation disclosure (§6 check 3). `deviations_disclosed` is the
    # COUNT of declared deviations (0 = the pre-registered analysis stands;
    # None = the bundle predates the member). `deviations_ok` verifies each
    # record: the DECLARER's signature over the canonical declaration, the
    # coordinator's anchor statement, and the binding to THIS bundle's manifest.
    # Having deviations never fails a bundle — honest disclosure is the point;
    # only a record that fails to VERIFY (tamper) does.
    deviations_disclosed: int | None = None
    deviations_ok: bool | None = None
    # C7 Inc 4: a within_cell_tolerance unit is ATTESTED by its deterministic
    # representative's hash (not a replica's), with the representative itself
    # shipped in the bundle's `unit_consensus` block. This check RECOMPUTES each
    # representative's hash from the bundled representative and matches it — the
    # tolerance leaf is recomputable from bytes in hand, not merely internally
    # consistent. None when the bundle carries no unit_consensus entries (exact /
    # pre-Inc-4); False = a representative fails to recompute (tamper or strip).
    tolerance_evidence_ok: bool | None = None
    # How the custody signer was grounded. "explicit" = pinned to a caller
    # --signer set; "known" = matched an embedded KNOWN_PUBLIC_SIGNERS key (the
    # default for public-network bundles); "unpinned" = no pin matched (signer
    # is the bundle's own self-attested key, e.g. a private coordinator);
    # "skipped" = --no-pin. Only "explicit"/"known" mean externally grounded.
    # `transfer_signer_authorized` (the ok-gate) stays True for unpinned/skipped
    # — a soft default-pin never turns into a false FAIL.
    signer_pin_mode: str = "unpinned"

    @property
    def signer_grounded(self) -> bool:
        """True iff the custody signer was actually matched to a key external to
        the bundle — an embedded published key ("known"), or an explicit
        --signer set that the signer is IN. A failed explicit pin is not
        grounded; unpinned/skipped never are."""
        if self.signer_pin_mode == "known":
            return True
        if self.signer_pin_mode == "explicit":
            return self.transfer_signer_authorized
        return False

    @property
    def ok(self) -> bool:
        return (
            self.transfer_signature_valid
            and self.transfer_signer_authorized
            and (self.attestation is None or self.attestation.ok)
            and self.root_unified is not False
            and self.completeness_ok is not False
            and self.inputs_bound_ok is not False
            and self.footprint_ok is not False
            and self.manifest_bound_ok is not False
            and self.tolerance_evidence_ok is not False
            and self.prereg_bound_ok is not False
            and self.design_precedes_data_ok is not False
            and self.deviations_ok is not False
            and self.worker_signatures.ok
        )


def _attestation_from_bundle(block: dict[str, Any]) -> ResultSetAttestation:
    """Reconstruct a `ResultSetAttestation` from the bundle's attestation
    block. The block carries the COSE artifact but not the convenience `units`
    list — recover units (and the predicate-only fields) from the SIGNED
    statement itself, which is the stronger source anyway."""
    cose = cbor2.loads(b64decode(block["cose_b64"]))
    statement = cbor2.loads(cose[2])  # [protected, unprotected, payload, sig]
    predicate = cbor2.loads(statement["predicate"])
    return ResultSetAttestation(
        attestation_id=block["attestation_id"],
        experiment_id=predicate.get("experiment_id", ""),
        tenant_id=predicate.get("tenant_id", ""),
        merkle_root=block["merkle_root"],
        algorithm=block["algorithm"],
        unit_count=predicate.get("unit_count", 0),
        units=predicate.get("units") or [],
        cose_b64=block["cose_b64"],
        signing_key_pubkey_hex=block["signing_key_pubkey_hex"],
        rekor_log_index=block.get("rekor_log_index", 0),
        rekor_entry_uuid=block.get("rekor_entry_uuid", ""),
        partial=bool(predicate.get("partial", False)),
        rekor_inclusion_proof=block.get("rekor_inclusion_proof"),
        governance_footprint=predicate.get("governance_footprint"),
        diverged_units=predicate.get("diverged_units") or [],
    )


_INTEGRITY_BASES = ("within_cell_exact", "within_cell_tolerance", "process_only", "diverged")


def recompute_integrity_basis_counts(
    units: list[dict[str, Any]], diverged_units: list[dict[str, Any]] | None
) -> dict[str, int]:
    """Firewall #2 recomputable half: re-derive the integrity_basis distribution
    from the SIGNED predicate's own per-unit basis + diverged_units, independently
    of the footprint's aggregate claim. The consumer-side twin of the
    coordinator's `assert_footprint_recomputable` sign-time guard."""
    counts = {b: 0 for b in _INTEGRITY_BASES}
    for u in units or []:
        b = u.get("integrity_basis")
        if b in counts:
            counts[b] += 1
    counts["diverged"] += len(diverged_units or [])
    return counts


def _canonical_result_bytes(r: dict[str, Any]) -> bytes:
    """The worker daemon's canonical signing input (signing/result.py) —
    reproduced byte-for-byte from bundle fields. A v1 result (§9 #13a)
    additionally binds `schema_version` + `served_weights`, and a v2 result
    (A2 #32) additionally binds `ran_under` (the sandbox policy), so verifying the
    worker signature also verifies those (a tampered digest or containment claim
    fails the signature). v0/v1 reconstruction stays byte-identical."""
    body = {
        "unit_id": r["unit_id"],
        "worker_pubkey": r["worker_pubkey_hex"].lower(),
        "completed_at": r["completed_at"],
        "exit_code": int(r["exit_code"]),
        "payload": r["payload"],
    }
    schema_version = r.get("schema_version")
    if schema_version and int(schema_version) >= 1:
        body["schema_version"] = int(schema_version)
        body["served_weights"] = {
            str(k): str(v).lower() for k, v in (r.get("served_weights") or {}).items()
        }
    if schema_version and int(schema_version) >= 2:
        body["ran_under"] = str(r.get("ran_under") or "").lower()
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_worker_signatures(
    consensus_results: list[dict[str, Any]], *, strict: bool = False
) -> WorkerSignatureReport:
    """Verify each consensus result's worker signature — the only signature in
    the chain not made by the coordinator (defense in depth against an at-rest
    or coordinator-level tamper). `strict` makes missing signature fields a
    failure (v1 bundles guarantee the members, so absence = tamper)."""
    verified = 0
    failed: list[str] = []
    skipped_aged = 0
    skipped_missing = 0
    for r in consensus_results:
        if r.get("aged_off") or r.get("payload") is None:
            skipped_aged += 1
            continue
        if not r.get("worker_pubkey_hex") or r.get("exit_code") is None:
            skipped_missing += 1
            continue
        try:
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(r["worker_pubkey_hex"]))
            pub.verify(b64decode(r["worker_signature"]), _canonical_result_bytes(r))
            verified += 1
        except (InvalidSignature, KeyError, ValueError):
            failed.append(r.get("result_id") or r.get("unit_id") or "<unknown>")
    return WorkerSignatureReport(
        verified=verified,
        failed=failed,
        skipped_aged_off=skipped_aged,
        skipped_missing_fields=skipped_missing,
        strict=strict,
    )


_NOT_ANCHORED_UUID = "lab-mode-no-rekor"


def _real_anchor(log_index: Any, entry_uuid: Any) -> bool:
    """A genuine Rekor anchor (not the pending/lab placeholder)."""
    return bool(log_index) and bool(entry_uuid) and entry_uuid != _NOT_ANCHORED_UUID


def _verify_deviation(d: dict[str, Any], signed_manifest_hash: str | None) -> bool:
    """One deviation record verifies iff: the DECLARER's Ed25519 signature holds
    over the canonical declaration; the coordinator's COSE anchor statement
    decodes and its predicate binds the declaration digest + the same manifest
    hash; and the record references THIS bundle's custody-signed manifest."""
    import hashlib as _hashlib

    from auspexai_tenant.signing import canonical_deviation_bytes

    try:
        declaration = canonical_deviation_bytes(d["manifest_hash"], d["what_changed"], d["why"])
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(d["tenant_pubkey_hex"])).verify(
            b64decode(d["tenant_signature_b64"]), declaration
        )
        valid, _signer, payload = _cose_verify(b64decode(d["cose_b64"]))
        if not valid:
            return False
        predicate = cbor2.loads(cbor2.loads(payload)["predicate"])
    except Exception:
        return False
    if predicate.get("declaration_sha256") != _hashlib.sha256(declaration).hexdigest():
        return False
    if predicate.get("manifest_hash") != d["manifest_hash"]:
        return False
    return not (signed_manifest_hash and d["manifest_hash"] != signed_manifest_hash)


def _verify_prereg_binding(
    pr_block: dict[str, Any],
    bundle: dict[str, Any],
    transfer: dict[str, Any],
    authorized_signers: list[str] | None,
) -> bool:
    """D16.2: the pre-registration blob COSE-verifies; its signed predicate binds
    THIS bundle's manifest hash; and the design it anchors equals the embedded
    manifest's pre_registration block (analysis-matches-declared: with
    manifest_bound_ok, the analysis the manifest declares is provably the one
    that was anchored before data existed). Any decode failure is False —
    a malformed anchor never passes."""
    try:
        valid, signer_hex, payload = _cose_verify(b64decode(pr_block["cose_b64"]))
        if not valid:
            return False
        if authorized_signers is not None and (
            signer_hex is None or signer_hex.lower() not in {s.lower() for s in authorized_signers}
        ):
            return False
        statement = cbor2.loads(payload)
        predicate = cbor2.loads(statement["predicate"])
    except Exception:
        return False
    signed_mh = transfer.get("manifest_hash")
    if not signed_mh or predicate.get("manifest_hash") != signed_mh:
        return False
    if pr_block.get("manifest_hash") != signed_mh:
        return False
    # The anchored design == the manifest's declared design (when embedded).
    manifest = bundle.get("manifest")
    if isinstance(manifest, dict):
        if predicate.get("pre_registration") != manifest.get("pre_registration"):
            return False
    # The result attestation's binding (maximal tier), when it carries one,
    # must reference the same design.
    att_block = bundle.get("attestation")
    if att_block:
        try:
            att_cose = cbor2.loads(b64decode(att_block["cose_b64"]))
            att_pred = cbor2.loads(cbor2.loads(att_cose[2])["predicate"])
            att_ref = att_pred.get("pre_registration")
            if att_ref is not None and att_ref.get("manifest_hash") != signed_mh:
                return False
        except Exception:
            return False
    return True


def _semantic_hash(exit_code: int, payload: dict[str, Any]) -> str:
    """Mirror of the coordinator's semantic-hash convention
    (`auspexai_platform.hashing.semantic_hash`): SHA-256 over canonical
    ``{exit_code, payload}``. Used to RECOMPUTE a tolerance unit's attested
    representative hash from the representative shipped in the bundle."""
    canonical = json.dumps(
        {"exit_code": exit_code, "payload": payload}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    """The content-address of a manifest dict. MIRRORS the coordinator's
    ManifestRepository.hash_manifest (canonical JSON: sorted keys, compact
    separators, UTF-8, sha256) — the hash carried in the signed transfer record.
    Kept in lockstep so an embedded manifest can be bound to that signed hash."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_bundle(
    bundle: dict[str, Any],
    *,
    authorized_signers: list[str] | None = None,
    no_pin: bool = False,
    check_rekor: bool = False,
    rekor_url: str = DEFAULT_REKOR_URL,
) -> BundleVerification:
    """Run the full evidence-bundle verification chain (module docstring).
    Offline by default; `check_rekor=True` adds the online transparency-log
    inclusion check.

    Signer grounding (which key the custody + attestation signatures must come
    from) has three modes:
      - `authorized_signers` given → HARD pin to that set: a signer outside it
        FAILS verification (use for a private coordinator, or to pin against a
        specific AUTHORIZED_SIGNERS.md key).
      - default (neither arg) → SOFT pin against the embedded
        `KNOWN_PUBLIC_SIGNERS`: a public-network bundle is grounded
        (`signer_pin_mode="known"`), an unrecognized signer is reported
        `"unpinned"` but does NOT fail (it is still self-consistent).
      - `no_pin=True` → skip grounding entirely (`"skipped"`, self-consistency
        only).
    A soft default-pin never turns a genuine bundle into a false FAIL."""
    # Cross-version gate: refuse a bundle schema this SDK doesn't know rather
    # than verifying whatever subset of it happens to parse (an old SDK
    # "passing" a newer bundle would silently skip checks the newer schema
    # carries). Pre-EB-1 bundles have no schema member and keep the lenient
    # legacy path.
    schema = bundle.get("schema")
    if schema is not None and schema != BUNDLE_SCHEMA_V1:
        raise ValueError(
            f"unknown evidence-bundle schema {schema!r} — this auspexai-tenant "
            f"version understands {BUNDLE_SCHEMA_V1!r}; upgrade the SDK to "
            "verify this bundle"
        )
    t = bundle["transfer"]

    # 1. proof-of-transfer signature + external pinning
    record = (
        f"{t['result_set_root']}|{t['collected_by_pubkey']}|"
        f"{t['collected_at']}|{t['manifest_hash']}"
    ).encode()
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(t["coordinator_pubkey_hex"]))
        pub.verify(bytes.fromhex(t["coordinator_signature"]), record)
        transfer_sig_ok = True
    except (InvalidSignature, KeyError, ValueError):
        transfer_sig_ok = False
    # Signer grounding (see docstring). att_signers is the hard-pin set passed
    # down to the attestation check — None means "don't hard-fail the
    # attestation signer" (the soft default and --no-pin paths), since the
    # attestation is signed by the same coordinator key as the transfer.
    signer_hex = t.get("coordinator_pubkey_hex", "").lower()
    att_signers: list[str] | None
    if no_pin:
        signer_authorized = True
        signer_pin_mode = "skipped"
        att_signers = None
    elif authorized_signers is not None:
        signer_authorized = signer_hex in {s.lower() for s in authorized_signers}
        signer_pin_mode = "explicit"
        att_signers = authorized_signers
    elif signer_hex in {k.lower() for k in KNOWN_PUBLIC_SIGNERS}:
        signer_authorized = True
        signer_pin_mode = "known"
        att_signers = None
    else:
        signer_authorized = True
        signer_pin_mode = "unpinned"
        att_signers = None

    consensus = bundle.get("consensus_results") or []
    # C7 Inc 4: {unit_id: evidence} from the bundle's unit_consensus block —
    # tolerance units are ATTESTED by their deterministic representative's hash,
    # delivered as a promoted replica row; this block bridges the two. Empty for
    # exact / pre-Inc-4 bundles (all checks below then behave exactly as before).
    uc_by_unit = {
        u["unit_id"]: u
        for u in (bundle.get("unit_consensus") or [])
        if isinstance(u, dict) and u.get("unit_id")
    }

    # 2-5. attestation chain (when the bundle carries one)
    att_verification: AttestationVerification | None = None
    root_unified: bool | None = None
    completeness: bool | None = None
    inputs_bound: bool | None = None
    footprint_ok: bool | None = None
    governance_footprint: dict[str, Any] | None = None
    block = bundle.get("attestation")
    if block:
        att = _attestation_from_bundle(block)
        governance_footprint = att.governance_footprint
        if governance_footprint is not None:
            # Firewall #2 F6 (consumer side): the footprint's asserted
            # integrity_basis counts must match a fresh recount from the signed
            # predicate's own per-unit basis + diverged_units.
            claimed = (governance_footprint.get("integrity_basis") or {}).get("counts") or {}
            recount = recompute_integrity_basis_counts(att.units, att.diverged_units)
            footprint_ok = {k: claimed.get(k, 0) for k in recount} == recount
        att_verification = verify_attestation(
            att,
            authorized_signers=att_signers,
            check_rekor=check_rekor,
            rekor_url=rekor_url,
        )
        if t.get("root_kind") and t["root_kind"] != FLAT_ROOT_KIND:
            root_unified = t["result_set_root"] == att.merkle_root

        def _delivered_hash(r: dict[str, Any]) -> str:
            uc = uc_by_unit.get(r["unit_id"])
            if uc and uc.get("representative_hash"):
                return uc["representative_hash"]
            return r.get("semantic_hash") or ""

        delivered = {(r["unit_id"], _delivered_hash(r)) for r in consensus}
        attested = {(u["unit_id"], u["consensus_result_hash"]) for u in att.units}
        completeness = delivered == attested
        if att.algorithm == RESULT_SET_ALGORITHM_V1 and att.units:
            payloads = {w["unit_id"]: w.get("payload") for w in bundle.get("work_units") or []}
            inputs_bound = all(
                bool(u.get("unit_payload_sha256"))
                and payloads.get(u["unit_id"]) is not None
                and unit_payload_sha256(payloads[u["unit_id"]]) == u["unit_payload_sha256"]
                for u in att.units
            )

    # 6. worker signatures — strict on v1 bundles (the members are guaranteed
    # there, so a missing field is a strip-tamper, not an old coordinator).
    ws = verify_worker_signatures(consensus, strict=schema is not None)

    # C7 Inc 4: recompute each tolerance unit's attested representative hash from
    # the representative shipped in the bundle — the leaf is recomputable from
    # bytes in hand. An entry missing its representative (or failing the
    # recompute) is a strip/tamper, never skipped.
    tolerance_evidence: bool | None = None
    if uc_by_unit:
        tolerance_evidence = all(
            isinstance(u.get("representative"), dict)
            and bool(u.get("representative_hash"))
            and _semantic_hash(0, u["representative"]) == u["representative_hash"]
            for u in uc_by_unit.values()
        )

    # D16.2: the pre-registration binding + `design ≺ data` ordering.
    prereg_bound: bool | None = None
    design_precedes_data: bool | None = None
    pr_block = bundle.get("pre_registration")
    if pr_block:
        prereg_bound = _verify_prereg_binding(pr_block, bundle, t, att_signers)
        att_block_anchor = bundle.get("attestation") or {}
        if _real_anchor(
            pr_block.get("rekor_log_index"), pr_block.get("rekor_entry_uuid")
        ) and _real_anchor(
            att_block_anchor.get("rekor_log_index"), att_block_anchor.get("rekor_entry_uuid")
        ):
            design_precedes_data = int(pr_block["rekor_log_index"]) < int(
                att_block_anchor["rekor_log_index"]
            )

    # D16.2-D: deviation disclosure — count + per-record verification.
    deviations_disclosed: int | None = None
    deviations_ok: bool | None = None
    devs = bundle.get("deviations")
    if devs is not None:
        deviations_disclosed = len(devs)
        if devs:
            deviations_ok = all(
                isinstance(d, dict) and _verify_deviation(d, t.get("manifest_hash")) for d in devs
            )

    # 7. D16.1: bind the embedded manifest to the SIGNED manifest_hash (covered
    # by the transfer signature). None when either is absent (pre-D16.1 bundle /
    # old coordinator); False = the manifest body was swapped — so its
    # feature_schema is NOT the attested one and load_verified must not trust it.
    manifest_bound: bool | None = None
    embedded_manifest = bundle.get("manifest")
    signed_manifest_hash = t.get("manifest_hash")
    if embedded_manifest is not None and signed_manifest_hash:
        manifest_bound = _canonical_manifest_hash(embedded_manifest) == signed_manifest_hash

    return BundleVerification(
        transfer_signature_valid=transfer_sig_ok,
        transfer_signer_authorized=signer_authorized,
        attestation=att_verification,
        root_unified=root_unified,
        completeness_ok=completeness,
        inputs_bound_ok=inputs_bound,
        footprint_ok=footprint_ok,
        manifest_bound_ok=manifest_bound,
        governance_footprint=governance_footprint,
        worker_signatures=ws,
        signer_pin_mode=signer_pin_mode,
        tolerance_evidence_ok=tolerance_evidence,
        prereg_bound_ok=prereg_bound,
        design_precedes_data_ok=design_precedes_data,
        deviations_disclosed=deviations_disclosed,
        deviations_ok=deviations_ok,
    )


# ---- the evidence loader (§9 #47 §6) ----------------------------------------


class BundleVerificationError(Exception):
    """Raised by `load_verified` when the bundle fails verification. Carries
    the full `BundleVerification` so the caller can see WHICH check failed —
    but the loader never returns data from a bundle that doesn't verify."""

    def __init__(self, verification: BundleVerification) -> None:
        self.verification = verification
        failed = []
        v = verification
        if not v.transfer_signature_valid:
            failed.append("transfer signature")
        if not v.transfer_signer_authorized:
            failed.append("transfer signer not in authorized list")
        if v.attestation is not None and not v.attestation.ok:
            failed.append("attestation")
        if v.root_unified is False:
            failed.append("custody/attestation root mismatch")
        if v.completeness_ok is False:
            failed.append("delivered set != attested set")
        if v.inputs_bound_ok is False:
            failed.append("input binding (work-unit payload hashes)")
        if v.manifest_bound_ok is False:
            failed.append("manifest binding (embedded manifest != signed manifest_hash)")
        if v.tolerance_evidence_ok is False:
            failed.append("tolerance evidence (representative hash fails to recompute)")
        if v.prereg_bound_ok is False:
            failed.append("pre-registration binding (anchor/signature/design mismatch)")
        if v.design_precedes_data_ok is False:
            failed.append("design ≺ data ordering (the design was anchored AFTER the results)")
        if v.deviations_ok is False:
            failed.append("deviation records (a declared deviation fails to verify)")
        if v.worker_signatures.failed:
            failed.append(f"worker signatures ({', '.join(v.worker_signatures.failed)})")
        if not v.worker_signatures.ok and not v.worker_signatures.failed:
            failed.append(
                f"worker signature fields missing on "
                f"{v.worker_signatures.skipped_missing_fields} row(s) of a v1 bundle"
            )
        super().__init__("evidence bundle failed verification: " + "; ".join(failed))


def _flatten_into(row: dict[str, Any], prefix: str, value: Any) -> None:
    """Flatten nested dicts into dot-separated columns (vigiles-style payloads
    carry e.g. payload.lexical.tokens). Non-dict leaves — scalars AND lists —
    land as the cell value; the table writer JSON-encodes residual non-scalars
    so Parquet/CSV stay importable everywhere."""
    if isinstance(value, dict) and value:
        for k, v in value.items():
            _flatten_into(row, f"{prefix}.{k}", v)
    else:
        row[prefix] = value


def _read_bundle(bundle: Any) -> dict[str, Any]:
    """Accept a parsed bundle dict or a path to a saved bundle JSON file."""
    if isinstance(bundle, dict):
        return bundle
    from pathlib import Path

    return json.loads(Path(bundle).read_text(encoding="utf-8"))


def load_verified(
    bundle: Any,
    *,
    authorized_signers: list[str] | None = None,
    check_rekor: bool = False,
    rekor_url: str = DEFAULT_REKOR_URL,
):
    """Verify an evidence bundle, then return its results as a pandas
    DataFrame — analysis that BEGINS from a cryptographically verified
    dataset, in one call. Refuses (raises `BundleVerificationError`) if any
    check fails; there is deliberately no force/skip flag — to inspect a bad
    bundle, call `verify_bundle` directly.

    `bundle` is the dict from `TenantClient.export()` or a path to the JSON
    file `auspexai-tenant experiment export` saved.

    One row per consensus result. Columns: `unit_id`, `result_id`,
    `receipt_id`, `completed_at` (datetime), `semantic_hash`, `aged_off`,
    `integrity_basis` (firewall #1 corroboration strength), `served_weights`
    (§9 #13a worker-attested {model_id: gguf_sha256}, signature-covered),
    plus the work-unit INPUT payload flattened under `input.*` and the result
    OUTPUT payload flattened under `output.*` (NaN for aged-off rows — their
    receipt + semantic_hash still verify, but the payload is gone; collection
    is custody transfer). Round/sweep coordinates are tenant semantics — they
    live in your `input.*` columns or your unit-id convention, not in the
    platform's schema.

    Requires the `analysis` extra: `pip install auspexai-tenant[analysis]`.
    From here, `df.to_csv(...)` / `df.to_parquet(...)` open the Excel /
    Tableau / R door — or use the `auspexai-tenant bundle table` CLI.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "load_verified needs pandas — install the analysis extra: "
            "pip install 'auspexai-tenant[analysis]'"
        ) from e

    data = _read_bundle(bundle)
    verification = verify_bundle(
        data,
        authorized_signers=authorized_signers,
        check_rekor=check_rekor,
        rekor_url=rekor_url,
    )
    if not verification.ok:
        raise BundleVerificationError(verification)

    inputs = {w["unit_id"]: w.get("payload") or {} for w in data.get("work_units") or []}
    # Firewall #1: per-unit corroboration basis from the signed attestation, so a
    # researcher can stratify/filter by corroboration strength (the firewall #2
    # "correct for apparatus influence" use case).
    basis_by_unit: dict[str, Any] = {}
    if data.get("attestation"):
        att = _attestation_from_bundle(data["attestation"])
        basis_by_unit = {u["unit_id"]: u.get("integrity_basis") for u in att.units}
    rows = []
    for r in data.get("consensus_results") or []:
        row: dict[str, Any] = {
            "unit_id": r["unit_id"],
            "result_id": r.get("result_id"),
            "receipt_id": r.get("receipt_id"),
            "completed_at": r.get("completed_at"),
            "semantic_hash": r.get("semantic_hash"),
            "aged_off": bool(r.get("aged_off")),
            "integrity_basis": basis_by_unit.get(r["unit_id"]),
            # §9 #13a: the worker-ATTESTED served-weights digest {model_id:
            # gguf_sha256} (covered by the verified worker signature) — so a
            # researcher can confirm WHICH model produced each row, and
            # stratify/exclude rows whose served model differs.
            "served_weights": r.get("served_weights"),
            # A2 #32: the worker-SIGNED sandbox policy each row ran under
            # (covered by the verified signature) — so a researcher can stratify
            # by containment, not just trust the apparatus's aggregate.
            "ran_under": r.get("ran_under"),
        }
        _flatten_into(row, "input", inputs.get(r["unit_id"]) or {})
        _flatten_into(row, "output", r.get("payload") or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["completed_at"] = pd.to_datetime(df["completed_at"], format="ISO8601")
        df = df.sort_values("unit_id", kind="stable").reset_index(drop=True)
    # The apparatus footprint travels with the verified frame (firewall #2,
    # researcher-facing): df.attrs["governance_footprint"].
    df.attrs["governance_footprint"] = verification.governance_footprint
    # Firewall #1: diverged units are VALID evidentiary data, but their payloads
    # are never exported (the signed predicate carries unit ids + worker-signed
    # result hashes only) — so they cannot be DataFrame ROWS. They travel as
    # df.attrs["diverged_units"], the custody-verified record that those units
    # could not be corroborated. (2026-07-03 correction: this guide/API pair
    # previously implied a `diverged` row stratum that could never be non-empty.)
    diverged_units = []
    if data.get("attestation"):
        diverged_units = _attestation_from_bundle(data["attestation"]).diverged_units or []
    df.attrs["diverged_units"] = diverged_units

    # D16.1 Inc 3: surface the declared feature schema as the frame's data
    # dictionary. Read from the bundle's manifest — which verify_bundle just bound
    # to the SIGNED manifest_hash (manifest_bound_ok), so this is the attested
    # documentation, not an out-of-band copy. Keyed by the output.-prefixed column
    # so `df.attrs["feature_schema"]["output.lexical.type_token_ratio"]` lines up
    # with the column. Under §7 the declared set IS the whole interpretability
    # surface (raw outputs are never retained), so a self-documented frame matters.
    feature_schema = (data.get("manifest") or {}).get("feature_schema")
    if feature_schema:
        df.attrs["feature_schema"] = {
            f"output.{path}": decl for path, decl in feature_schema.items()
        }
        if not df.empty:
            _apply_valid_when(df, feature_schema)
    return df


# D16.1 Inc 3 — the feature-schema researcher surface (load_verified + the
# `bundle table --data-dictionary` CLI). KEEP IN LOCKSTEP with the manifest
# FeatureDeclaration contract (manifest.py) + the coordinator mirror.

_VALID_WHEN_OPS = {
    ">=": _operator.ge,
    "<=": _operator.le,
    ">": _operator.gt,
    "<": _operator.lt,
    "==": _operator.eq,
    "!=": _operator.ne,
}


def _apply_valid_when(df: Any, feature_schema: dict[str, Any]) -> None:
    """For each feature declaring a structured `valid_when` {field, op, value},
    add a boolean `output.<path>.valid` column = (output.<field> <op> value), so a
    feature that is only interpretable under a precondition (e.g. type_token_ratio
    is degenerate when tokens < 5) is FLAGGED in the frame, not discovered by
    accident. Skipped silently when the referenced field column is absent."""
    for path, decl in feature_schema.items():
        vw = decl.get("valid_when") if isinstance(decl, dict) else None
        if not isinstance(vw, dict):
            continue
        opfn = _VALID_WHEN_OPS.get(vw.get("op"))
        field_col = f"output.{vw.get('field')}"
        if opfn is None or "value" not in vw or field_col not in df.columns:
            continue
        df[f"output.{path}.valid"] = opfn(df[field_col], vw["value"])


def data_dictionary_rows(feature_schema: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten a manifest feature_schema into printable data-dictionary rows
    (column · kind · role · unit · range/categories · meaning · change_means) for
    the `bundle table --data-dictionary` CLI. Pure; shared so the CLI and any
    other consumer render the SAME dictionary."""
    rows: list[dict[str, str]] = []
    for path, d in feature_schema.items():
        if not isinstance(d, dict):
            continue
        rng = d.get("range")
        cats = d.get("categories")
        if isinstance(rng, dict):
            lo, hi = rng.get("min"), rng.get("max")
            bounds = f"[{lo}, {hi if hi is not None else '∞'}]"
        elif cats:
            bounds = "{" + ", ".join(map(str, cats)) + "}"
        elif d.get("algorithm"):
            bounds = str(d["algorithm"])
        elif d.get("kind") == "set":
            bounds = f"≤{d.get('max_cardinality')} {d.get('element_kind')}"
        else:
            bounds = ""
        rows.append(
            {
                "column": f"output.{path}",
                "kind": str(d.get("kind", "")),
                "role": str(d.get("role", "")),
                "unit": str(d.get("unit") or ""),
                "bounds": bounds,
                "meaning": str(d.get("meaning", "")),
                "change_means": str(d.get("change_means", "")),
            }
        )
    return rows
