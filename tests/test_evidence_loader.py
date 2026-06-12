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
from auspexai_tenant.evidence import BundleVerificationError, load_verified
from tests.test_evidence import _keys, _make_bundle, _pub_hex
from tests.test_evidence import _sign_worker_result as _sign


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
    assert "not pinned" in r.output  # honest about unpinned verification
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
