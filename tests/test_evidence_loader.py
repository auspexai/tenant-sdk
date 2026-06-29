"""Evidence loader (§9 #47 §6) — verified-DataFrame loading + the bundle CLI.

The differentiator under test: analysis BEGINS from a cryptographically
verified dataset — `load_verified` refuses (no force flag) when any chain
check fails, and the `bundle table` CLI writes nothing from a bad bundle.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from auspexai_tenant.cli import main as cli_main
from auspexai_tenant.evidence import (
    BundleVerificationError,
    data_dictionary_rows,
    load_verified,
    verify_bundle,
)
from tests.test_evidence import _keys, _make_bundle, _pub_hex
from tests.test_evidence import _sign_worker_result as _sign


def _fs_manifest(valid_when: dict | None = None) -> dict:
    """A v0.3 manifest declaring a feature_schema for the `a` payload field that
    _make_bundle emits (output.a)."""
    decl: dict = {
        "meaning": "the answer value",
        "kind": "count",
        "role": "summary",
        "unit": "widgets",
        "range": {"min": 0},
        "change_means": "the computed value changed",
    }
    if valid_when is not None:
        decl["valid_when"] = valid_when
    return {"experiment_id": "exp-label", "schema_version": "0.3", "feature_schema": {"a": decl}}


def test_load_verified_returns_tidy_frame():
    ck, wk = _keys()
    df = load_verified(_make_bundle(ck, wk, n=3))
    assert list(df["unit_id"]) == ["u0", "u1", "u2"]  # stable unit order
    # linkage + integrity columns present
    for col in ("result_id", "receipt_id", "semantic_hash", "aged_off"):
        assert col in df.columns
    # inputs flatten under input.*, outputs under output.*
    assert list(df["input.q"]) == [0, 1, 2]
    assert list(df["output.a"]) == [0, 1, 2]
    assert pd.api.types.is_datetime64_any_dtype(df["completed_at"])


def test_load_verified_surfaces_integrity_basis_and_footprint():
    """Firewall #2 researcher surface: a per-row integrity_basis column to stratify
    by corroboration strength, and the apparatus footprint on df.attrs."""
    ck, wk = _keys()
    fp = {
        "schema_version": 1,
        "tenant": {"tier": "T2"},
        "integrity_basis": {
            "counts": {
                "within_cell_exact": 2,
                "within_cell_tolerance": 0,
                "process_only": 0,
                "diverged": 0,
            }
        },
    }
    df = load_verified(_make_bundle(ck, wk, n=2, unit_basis="within_cell_exact", footprint=fp))
    assert list(df["integrity_basis"]) == ["within_cell_exact", "within_cell_exact"]
    assert df.attrs["governance_footprint"]["tenant"]["tier"] == "T2"


def test_load_verified_surfaces_served_weights():
    """§9 #13a researcher surface: a per-row served_weights column (the
    worker-attested {model_id: gguf_sha256}, signature-covered) so a researcher
    can confirm which model produced each row."""
    ck, wk = _keys()
    df = load_verified(_make_bundle(ck, wk, n=2, served_weights={"gemma": "abc123"}))
    assert list(df["served_weights"]) == [{"gemma": "abc123"}, {"gemma": "abc123"}]


def test_load_verified_accepts_a_saved_bundle_path(tmp_path):
    ck, wk = _keys()
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_make_bundle(ck, wk)))
    df = load_verified(p)
    assert len(df) == 2


def test_load_verified_refuses_tampered_bundle_no_force_flag():
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"][0]["payload"] = {"a": "TAMPERED"}
    with pytest.raises(BundleVerificationError) as exc:
        load_verified(bundle)
    # the error names the failed check and carries the full verification
    assert "worker signatures" in str(exc.value)
    assert exc.value.verification.worker_signatures.failed == ["res-u0"]
    # deliberately no skip/force parameter on the loader
    import inspect

    assert "force" not in inspect.signature(load_verified).parameters


def test_load_verified_pins_signers():
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    with pytest.raises(BundleVerificationError) as exc:
        load_verified(bundle, authorized_signers=["ef" * 32])
    assert "authorized" in str(exc.value)
    assert len(load_verified(bundle, authorized_signers=[_pub_hex(ck)])) == 2


def test_aged_off_rows_load_with_nan_outputs():
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"][0]["payload"] = None
    bundle["consensus_results"][0]["aged_off"] = True
    df = load_verified(bundle)
    aged = df[df["aged_off"]]
    assert len(aged) == 1
    assert pd.isna(aged["output.a"]).all()
    # the receipt + semantic_hash still ride along for the aged-off row
    assert aged["semantic_hash"].notna().all()


def test_cli_bundle_table_writes_csv_and_parquet(tmp_path):
    ck, wk = _keys()
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_make_bundle(ck, wk)))
    runner = CliRunner()
    for suffix in ("csv", "parquet"):
        out = tmp_path / f"results.{suffix}"
        r = runner.invoke(cli_main, ["bundle", "table", str(p), "-o", str(out)])
        assert r.exit_code == 0, r.output
        assert "verified ✓" in r.output
        loaded = pd.read_csv(out) if suffix == "csv" else pd.read_parquet(out)
        assert len(loaded) == 2
        assert "output.a" in loaded.columns


def test_cli_bundle_table_refuses_bad_bundle(tmp_path):
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    bundle["consensus_results"].pop()  # break completeness
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle))
    out = tmp_path / "results.csv"
    r = CliRunner().invoke(cli_main, ["bundle", "table", str(p), "-o", str(out)])
    assert r.exit_code == 1
    assert "REFUSED" in r.output
    assert not out.exists()  # nothing written from an unverified bundle


def test_cli_bundle_verify_reports_and_exits(tmp_path):
    ck, wk = _keys()
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_make_bundle(ck, wk)))
    r = CliRunner().invoke(cli_main, ["bundle", "verify", str(p)])
    assert r.exit_code == 0, r.output
    assert "verified ✓" in r.output
    # honest about unpinned verification (random coordinator key, not an
    # embedded KNOWN_PUBLIC_SIGNERS key → grounded=False, but still self-consistent)
    assert "unpinned" in r.output
    # pinned + wrong key → fail
    r = CliRunner().invoke(cli_main, ["bundle", "verify", str(p), "--signer", "ef" * 32])
    assert r.exit_code == 1


def test_nested_payloads_flatten_and_write_parquet(tmp_path):
    """Live-data regression (D6 bundle, 2026-06-12): vigiles payloads nest
    dicts (payload.lexical.tokens) and carry lists (lexical.top_tokens) —
    nested dicts must flatten to dot columns and the table writer must
    JSON-encode residual lists so Parquet conversion never chokes."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk)
    for i, r in enumerate(bundle["consensus_results"]):
        r["payload"] = {
            "response_sha256": "ab" * 32,
            "lexical": {"tokens": 4 + i, "top_tokens": [["alpha", 2], ["beta", 1]]},
        }
        r["worker_signature"] = _sign(wk, r)
    df = load_verified(bundle)
    assert list(df["output.lexical.tokens"]) == [4, 5]
    assert isinstance(df["output.lexical.top_tokens"].iloc[0], list)  # faithful in pandas
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle))
    out = tmp_path / "nested.parquet"
    r = CliRunner().invoke(cli_main, ["bundle", "table", str(p), "-o", str(out)])
    assert r.exit_code == 0, r.output
    loaded = pd.read_parquet(out)
    assert json.loads(loaded["output.lexical.top_tokens"].iloc[0]) == [["alpha", 2], ["beta", 1]]


# ── D16.1 Inc 3 — the feature-schema researcher surface ──────────────────────


def test_feature_schema_surfaced_on_attrs():
    """The declared schema rides the verified frame, keyed by the output.-prefixed
    column, so a researcher reads the column's meaning straight off df.attrs."""
    ck, wk = _keys()
    df = load_verified(_make_bundle(ck, wk, n=3, manifest=_fs_manifest()))
    fs = df.attrs["feature_schema"]
    assert "output.a" in fs
    assert fs["output.a"]["meaning"] == "the answer value"
    assert fs["output.a"]["kind"] == "count"


def test_valid_when_adds_valid_column():
    """A feature interpretable only under a precondition gets a boolean
    output.<col>.valid column so the degenerate case is flagged in the frame."""
    ck, wk = _keys()
    df = load_verified(
        _make_bundle(
            ck, wk, n=3, manifest=_fs_manifest(valid_when={"field": "a", "op": ">=", "value": 1})
        )
    )
    # output.a = [0, 1, 2] → valid where a >= 1
    assert list(df["output.a.valid"]) == [False, True, True]


def test_no_feature_schema_means_no_attr():
    """A pre-D16.1 / non-v0.3 bundle (no feature_schema) surfaces no attr — and
    certainly no spurious .valid columns."""
    ck, wk = _keys()
    df = load_verified(_make_bundle(ck, wk, n=2))  # default manifest: no feature_schema
    assert "feature_schema" not in df.attrs
    assert not any(c.endswith(".valid") for c in df.columns)


def test_manifest_swap_fails_binding():
    """Tampering the embedded manifest (so it no longer hashes to the SIGNED
    manifest_hash) fails verification — load_verified must not surface an
    out-of-band feature_schema."""
    ck, wk = _keys()
    bundle = _make_bundle(ck, wk, n=2, manifest=_fs_manifest())
    bundle["manifest"]["feature_schema"]["a"]["meaning"] = (
        "TAMPERED"  # body changes; signed hash does not
    )
    v = verify_bundle(bundle)
    assert v.manifest_bound_ok is False and not v.ok
    with pytest.raises(BundleVerificationError, match="manifest binding"):
        load_verified(bundle)


def test_manifest_binding_ok_on_clean_bundle():
    ck, wk = _keys()
    assert verify_bundle(_make_bundle(ck, wk, manifest=_fs_manifest())).manifest_bound_ok is True


def test_data_dictionary_rows_pure():
    rows = data_dictionary_rows(_fs_manifest()["feature_schema"])
    assert len(rows) == 1
    r = rows[0]
    assert r["column"] == "output.a"
    assert r["kind"] == "count" and r["role"] == "summary"
    assert r["unit"] == "widgets" and r["bounds"] == "[0, ∞]"
    assert "answer value" in r["meaning"]


def test_bundle_table_data_dictionary_cli(tmp_path):
    ck, wk = _keys()
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_make_bundle(ck, wk, n=2, manifest=_fs_manifest())))
    res = CliRunner().invoke(cli_main, ["bundle", "table", str(p), "--data-dictionary"])
    assert res.exit_code == 0, res.output
    assert "output.a" in res.output
    assert "the answer value" in res.output
    assert "count" in res.output


def test_bundle_table_requires_out_or_dict(tmp_path):
    ck, wk = _keys()
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(_make_bundle(ck, wk, n=1, manifest=_fs_manifest())))
    res = CliRunner().invoke(cli_main, ["bundle", "table", str(p)])
    assert res.exit_code != 0
    assert "--out" in res.output and "--data-dictionary" in res.output
