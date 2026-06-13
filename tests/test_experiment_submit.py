"""`experiment submit` (one-step courier) + run-target resolution (Part A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from auspexai_tenant import cli as cli_mod
from auspexai_tenant.cli import main
from auspexai_tenant.signing import MaintainerKey
from auspexai_tenant.upload import UploadResult

FIXTURES = Path(__file__).parent / "fixtures"


def _pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.json").write_text((FIXTURES / "valid_minimal.json").read_text())
    (pkg / "executor.py").write_text("# stub executor\n")
    return pkg


def _keyfile(tmp_path: Path) -> Path:
    kp = tmp_path / "key"
    MaintainerKey.generate().save(kp)
    return kp


class _FakeClient:
    def __init__(self, status: str = "stored") -> None:
        self._status = status
        self.uploaded: list[Path] = []

    def upload_package(self, package_dir):
        self.uploaded.append(Path(package_dir))
        return {"package_digest": "ab" * 32, "status": self._status}


def test_submit_signs_uploads_and_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = _pkg(tmp_path)
    kp = _keyfile(tmp_path)
    fc = _FakeClient()
    monkeypatch.setattr(cli_mod, "_make_client", lambda c, k: fc)

    seen: dict[str, object] = {}

    def fake_submit(manifest_path, sig_path, coordinator, key):
        seen["sig_existed_at_submit"] = Path(sig_path).exists()
        seen["uploaded_before_submit"] = bool(fc.uploaded)
        return UploadResult(status_code=201, body='{"experiment_id":"exp-XYZ12"}', ok=True)

    monkeypatch.setattr(cli_mod, "submit_experiment_from_files", fake_submit)

    r = CliRunner().invoke(main, ["experiment", "submit", str(pkg), "--key", str(kp)])
    assert r.exit_code == 0, r.output
    # label is read FROM the manifest (the courier signs+ships what was built)
    assert "label:      sentinel-test-minimal" in r.output
    assert "exp-XYZ12" in r.output
    assert "experiment run sentinel-test-minimal" in r.output
    # it signed (sig on disk) and uploaded the package BEFORE creating the experiment
    assert (pkg / "manifest.json.sig").exists()
    assert seen["sig_existed_at_submit"] is True
    assert seen["uploaded_before_submit"] is True


def test_submit_409_hints_unique_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = _pkg(tmp_path)
    kp = _keyfile(tmp_path)
    monkeypatch.setattr(cli_mod, "_make_client", lambda c, k: _FakeClient())
    monkeypatch.setattr(
        cli_mod,
        "submit_experiment_from_files",
        lambda *a, **k: UploadResult(status_code=409, body='{"error":"label_exists"}', ok=False),
    )
    r = CliRunner().invoke(main, ["experiment", "submit", str(pkg), "--key", str(kp)])
    assert r.exit_code == 1
    assert "make_unique_label" in r.output or "unique" in r.output.lower()


def test_submit_missing_manifest_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "empty"
    pkg.mkdir()
    kp = _keyfile(tmp_path)
    r = CliRunner().invoke(main, ["experiment", "submit", str(pkg), "--key", str(kp)])
    assert r.exit_code == 1
    assert "no manifest" in r.output.lower()


# ---- run-target resolution ------------------------------------------------

_EXPS = [
    {
        "experiment_id": "exp-1",
        "tenant_experiment_label": "lab-a",
        "submitted_at": "2026-06-13T01:00:00Z",
    },
    {
        "experiment_id": "exp-2",
        "tenant_experiment_label": "lab-b",
        "submitted_at": "2026-06-13T02:00:00Z",
    },
]


class _ResolveClient:
    def list_experiments(self):
        return _EXPS

    def get_experiment(self, eid):
        return {"experiment_id": eid, "tenant_experiment_label": "from-id"}


def test_resolve_latest():
    assert cli_mod._resolve_experiment(_ResolveClient(), "latest") == ("exp-2", "lab-b")


def test_resolve_by_label():
    assert cli_mod._resolve_experiment(_ResolveClient(), "lab-a") == ("exp-1", "lab-a")


def test_resolve_by_exp_id_skips_listing():
    assert cli_mod._resolve_experiment(_ResolveClient(), "exp-9") == ("exp-9", "from-id")


def test_resolve_unknown_label_exits():
    with pytest.raises(SystemExit):
        cli_mod._resolve_experiment(_ResolveClient(), "nope")


# ---- experiment build (generic manifest assembly) -------------------------


def test_experiment_build_assembles_valid_manifest(tmp_path: Path):
    """`experiment build` writes a schema-valid pkg/manifest.json from
    experiment.toml — the generic build, no per-tenant build.py."""
    from auspexai_tenant.manifest import Manifest

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "executor.py").write_text("# stub executor\n")
    (tmp_path / "experiment.toml").write_text(
        "[experiment]\n"
        'label = "lab-x"\n'
        'tenant_id = "lab"\n'
        'contact = "you@lab.org"\n'
        'model_id = "gemma-3-1b"\n'
        'research_goal = "' + ("a deterministic behavioral probe panel " * 3) + '"\n'
        'prompt_characteristics = "neutral instruction-following probes"\n'
        "[executor]\n"
        'command = ["python", "executor.py"]\n'
        "[reducer]\n"
        'kind = "builtin_hash_agreement"\n'
    )
    r = CliRunner().invoke(main, ["experiment", "build", str(pkg), "--exact-label"])
    assert r.exit_code == 0, r.output
    m = json.loads((pkg / "manifest.json").read_text())
    Manifest.model_validate(m)  # schema-valid
    assert m["experiment_id"] == "lab-x"  # --exact-label kept it verbatim
    assert m["models"][0]["id"] == "gemma-3-1b"
    assert m["executor"]["command"] == ["python", "executor.py"]
    assert len(m["executor"]["package_sha256"]) == 64  # digest over pkg/


def test_experiment_build_unique_label_and_missing_field(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "executor.py").write_text("# stub\n")
    # missing tenant_id → a clear error naming the table + key
    (tmp_path / "experiment.toml").write_text(
        '[experiment]\nlabel = "lab-y"\ncontact = "you@lab.org"\nmodel_id = "m"\n'
        'research_goal = "' + ("x" * 60) + '"\nprompt_characteristics = "neutral probe panel"\n'
        '[executor]\ncommand = ["python", "executor.py"]\n[reducer]\nkind = "builtin_hash_agreement"\n'
    )
    r = CliRunner().invoke(main, ["experiment", "build", str(pkg)])
    assert r.exit_code == 1
    assert "tenant_id" in r.output

    # default (no --exact-label) stamps a unique suffix on the label.
    (tmp_path / "experiment.toml").write_text(
        '[experiment]\nlabel = "lab-y"\ntenant_id = "lab"\ncontact = "you@lab.org"\nmodel_id = "m"\n'
        'research_goal = "' + ("x" * 60) + '"\nprompt_characteristics = "neutral probe panel"\n'
        '[executor]\ncommand = ["python", "executor.py"]\n[reducer]\nkind = "builtin_hash_agreement"\n'
    )
    r = CliRunner().invoke(main, ["experiment", "build", str(pkg)])
    assert r.exit_code == 0, r.output
    assert json.loads((pkg / "manifest.json").read_text())["experiment_id"].startswith("lab-y-")
