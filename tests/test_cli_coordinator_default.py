"""The shared researcher-command --coordinator option defaults to the public
network (v0.5.7).

`_coord_opt` backs every researcher-facing command (experiment list/status/
export/results/receipts/run/reduce, model catalog). It used to
be required=True, which made `experiment export …` (the evidence-custody path)
demand a --coordinator that `apply`/`package upload` did not — pure friction.
This pins the front-door default + AUSPEXAI_COORDINATOR_URL override so the
inconsistency cannot silently come back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from auspexai_tenant import cli as cli_mod
from auspexai_tenant.cli import main

PUBLIC = "https://coord.auspexai.network"


def _patch_make_client(monkeypatch: pytest.MonkeyPatch, seen: dict[str, str]) -> None:
    class FakeClient:
        def get_experiment(self, experiment_id: str) -> dict[str, str]:
            return {"experiment_id": experiment_id, "status": "completed"}

    def fake_make_client(coordinator: str, key_path: Path) -> FakeClient:
        seen["coordinator"] = coordinator
        return FakeClient()

    monkeypatch.setattr(cli_mod, "_make_client", fake_make_client)


def test_coord_opt_defaults_to_public_network(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    monkeypatch.delenv("AUSPEXAI_COORDINATOR_URL", raising=False)
    _patch_make_client(monkeypatch, seen)
    result = CliRunner().invoke(main, ["experiment", "status", "exp-x"])
    assert result.exit_code == 0, result.output
    assert seen["coordinator"] == PUBLIC


def test_coord_opt_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    monkeypatch.setenv("AUSPEXAI_COORDINATOR_URL", "https://coord.local.test")
    _patch_make_client(monkeypatch, seen)
    result = CliRunner().invoke(main, ["experiment", "status", "exp-x"])
    assert result.exit_code == 0, result.output
    assert seen["coordinator"] == "https://coord.local.test"


def test_coord_opt_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}
    monkeypatch.setenv("AUSPEXAI_COORDINATOR_URL", "https://coord.env.test")
    _patch_make_client(monkeypatch, seen)
    result = CliRunner().invoke(
        main, ["experiment", "status", "exp-x", "--coordinator", "https://coord.flag.test"]
    )
    assert result.exit_code == 0, result.output
    assert seen["coordinator"] == "https://coord.flag.test"


def test_experiment_export_creates_missing_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`export -o nested/dir/bundle.json` must not lose a delivered bundle to a
    FileNotFoundError — the coordinator already armed collection by then."""

    class FakeClient:
        def get_experiment(self, experiment_id: str) -> dict[str, object]:
            return {"experiment_id": experiment_id, "tenant_experiment_label": "lab-x"}

        def export(self, experiment_id: str) -> dict[str, object]:
            return {"transfer": {"transfer_id": "t-1", "root_kind": "attestation"}}

    monkeypatch.setattr(cli_mod, "_make_client", lambda c, k: FakeClient())
    out = tmp_path / "does" / "not" / "exist" / "bundle.json"
    result = CliRunner().invoke(
        main, ["experiment", "export", "exp-x", "--no-verify", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_experiment_export_default_lands_in_runs_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no -o, the bundle defaults to runs/<label>/bundle.json — keyed by the
    tenant LABEL (resolved from the exp- id), so it co-locates with the journal."""
    monkeypatch.delenv("AUSPEXAI_RUNS_DIR", raising=False)

    class FakeClient:
        def get_experiment(self, experiment_id: str) -> dict[str, object]:
            return {"experiment_id": experiment_id, "tenant_experiment_label": "drift-a1"}

        def export(self, experiment_id: str) -> dict[str, object]:
            return {"transfer": {"transfer_id": "t-1", "root_kind": "attestation"}}

    monkeypatch.setattr(cli_mod, "_make_client", lambda c, k: FakeClient())
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["experiment", "export", "exp-x", "--no-verify"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "drift-a1" / "bundle.json").exists()
