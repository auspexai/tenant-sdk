"""experiment.toml loader + make_unique_label (Part A)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from auspexai_tenant.experiment_config import (
    PROFILE_ENV_VAR,
    load_experiment_config,
    make_unique_label,
    manifest_dict_from_config,
)


def test_make_unique_label_is_deterministic_with_now():
    now = datetime(2026, 6, 13, 10, 30, 45, tzinfo=UTC)
    assert make_unique_label("vigiles-lab", now=now) == "vigiles-lab-20260613-103045"


def test_make_unique_label_keeps_manifest_id_pattern():
    import re

    now = datetime(2026, 6, 13, 10, 30, 45, tzinfo=UTC)
    label = make_unique_label("vigiles-lab", now=now)
    # the manifest experiment_id pattern
    assert re.fullmatch(r"[a-z][a-z0-9-]{2,127}", label)


def test_missing_file_reads_as_empty(tmp_path: Path):
    cfg = load_experiment_config(tmp_path)  # dir without experiment.toml
    assert cfg.label is None
    assert cfg.model_id is None
    assert cfg.driver_entrypoint is None
    assert cfg.experiment == {} and cfg.driver == {}


def test_parses_both_tables(tmp_path: Path):
    (tmp_path / "experiment.toml").write_text(
        "[experiment]\n"
        'label = "vigiles-lab"\n'
        'model_id = "gemma-3-1b"\n'
        "max_units = 500\n"
        "[driver]\n"
        'entrypoint = "drift_driver:build"\n'
        "cadence_seconds = 300\n"
    )
    cfg = load_experiment_config(tmp_path)
    assert cfg.label == "vigiles-lab"
    assert cfg.model_id == "gemma-3-1b"
    assert cfg.experiment["max_units"] == 500
    assert cfg.driver_entrypoint == "drift_driver:build"
    assert cfg.driver["cadence_seconds"] == 300  # opaque pass-through


def test_journal_path_auto_vs_explicit(tmp_path: Path):
    (tmp_path / "experiment.toml").write_text('[driver]\njournal = "auto"\n')
    assert load_experiment_config(tmp_path).journal_path("vig-1") == Path("vig-1.journal")
    (tmp_path / "experiment.toml").write_text('[driver]\njournal = "runs/my.journal"\n')
    assert load_experiment_config(tmp_path).journal_path("vig-1") == Path("runs/my.journal")
    # journal unset → also <label>.journal
    (tmp_path / "experiment.toml").write_text("[driver]\nentrypoint = 'd:b'\n")
    assert load_experiment_config(tmp_path).journal_path("vig-2") == Path("vig-2.journal")


def test_explicit_file_path(tmp_path: Path):
    f = tmp_path / "custom.toml"
    f.write_text('[experiment]\nlabel = "x-lab"\n')
    assert load_experiment_config(f).label == "x-lab"


def test_walks_up_to_find_config_from_subdir(tmp_path: Path, monkeypatch):
    """`load_experiment_config(None)` walks up from cwd — so `run`/`launch` work
    from a subdir (e.g. driver/) with no --config. This is the fix for the
    'no driver' error when running from driver/."""
    (tmp_path / "experiment.toml").write_text(
        "[experiment]\n"
        'label = "vig"\n'
        'key_path = "~/.config/auspexai-tenant/vig.key"\n'
        "[driver]\n"
        'entrypoint = "drift_driver:build"\n'
        'path = "driver"\n'
    )
    sub = tmp_path / "driver"
    sub.mkdir()
    monkeypatch.chdir(sub)
    cfg = load_experiment_config(None)  # None → walk up from cwd (driver/)
    assert cfg.source_path is not None
    assert cfg.label == "vig"
    assert cfg.driver_entrypoint == "drift_driver:build"
    assert cfg.driver_path == "driver"
    assert cfg.key_path == "~/.config/auspexai-tenant/vig.key"


def test_no_config_up_the_tree_reads_as_empty(tmp_path: Path, monkeypatch):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    cfg = load_experiment_config(None)
    assert cfg.source_path is None
    assert cfg.driver_entrypoint is None and cfg.driver_path is None


# ── [profiles.<name>] override-sets (Phase-3 released-capability model) ───────

_PROFILE_TOML = (
    "[experiment]\n"
    'label = "vig-lab"\n'
    'tenant_id = "vigiles-lab"\n'
    'contact = "v@auspexai.network"\n'
    'model_id = "gemma-3-1b"\n'
    'research_goal = "a goal long enough to be a paragraph for the schema validator to accept"\n'
    'prompt_characteristics = "benign fixed probes"\n'
    "replication = 1\n"
    "duration_hours = 1\n"
    'sensitive_content_flags = ["jailbreak"]\n'
    "[executor]\n"
    'command = ["python", "executor.py"]\n'
    "[reducer]\n"
    'kind = "builtin_hash_agreement"\n'
    "[driver]\n"
    'entrypoint = "drift_driver:build"\n'
    "run_seconds = 0\n"
    "max_rounds = 50\n"
    "[profiles.starter]\n"
    'description = "declawed"\n'
    "[profiles.starter.experiment]\n"
    "sensitive_content_flags = []\n"
    "replication = 2\n"
    "[profiles.starter.driver]\n"
    "run_seconds = 0\n"
    "[profiles.research]\n"
    "[profiles.research.experiment]\n"
    "duration_hours = 8\n"
    "[profiles.research.driver]\n"
    "run_seconds = 28800\n"
    "max_rounds = 120\n"
)


def _write_profile_toml(tmp_path: Path) -> Path:
    (tmp_path / "experiment.toml").write_text(_PROFILE_TOML)
    return tmp_path


def test_no_profile_is_backward_compatible(tmp_path: Path):
    cfg = load_experiment_config(_write_profile_toml(tmp_path))
    assert cfg.active_profile is None
    assert cfg.experiment["replication"] == 1
    assert cfg.experiment["sensitive_content_flags"] == ["jailbreak"]
    assert cfg.driver["run_seconds"] == 0 and cfg.driver["max_rounds"] == 50


def test_starter_profile_overrides_only_declared_keys(tmp_path: Path):
    cfg = load_experiment_config(_write_profile_toml(tmp_path), profile="starter")
    assert cfg.active_profile == "starter"
    # overridden by [profiles.starter.experiment]
    assert cfg.experiment["replication"] == 2
    assert cfg.experiment["sensitive_content_flags"] == []  # list REPLACED, not merged
    # not overridden → inherited from base [experiment]
    assert cfg.experiment["duration_hours"] == 1
    assert cfg.experiment["model_id"] == "gemma-3-1b"
    # a scalar profile key (description) is metadata — never leaks into a section
    assert "description" not in cfg.experiment and "description" not in cfg.driver


def test_research_profile_reaches_driver_knobs(tmp_path: Path):
    cfg = load_experiment_config(_write_profile_toml(tmp_path), profile="research")
    assert cfg.active_profile == "research"
    assert cfg.driver["run_seconds"] == 28800  # the long-horizon knob the driver reads
    assert cfg.driver["max_rounds"] == 120
    assert cfg.experiment["duration_hours"] == 8
    assert cfg.experiment["replication"] == 1  # inherited (research didn't override)


def test_profile_overrides_land_in_the_manifest(tmp_path: Path):
    cfg = load_experiment_config(_write_profile_toml(tmp_path), profile="starter")
    m = manifest_dict_from_config(cfg, package_sha256="0" * 64, label="vig-lab-x")
    assert m["replication_factor"] == 2
    assert m["sensitive_content_flags"] == []
    assert m["expected_duration_hours"] == 1.0


def test_unknown_profile_raises_with_available(tmp_path: Path):
    with pytest.raises(ValueError) as ei:
        load_experiment_config(_write_profile_toml(tmp_path), profile="nope")
    msg = str(ei.value)
    assert "nope" in msg and "starter" in msg and "research" in msg


def test_default_profile_applies_when_unset(tmp_path: Path):
    (tmp_path / "experiment.toml").write_text('default_profile = "starter"\n' + _PROFILE_TOML)
    cfg = load_experiment_config(tmp_path)  # no explicit profile
    assert cfg.active_profile == "starter"
    assert cfg.experiment["replication"] == 2


def test_env_var_profile_fallback(tmp_path: Path, monkeypatch):
    """The driver factory re-loads the config with no explicit profile — it must
    pick up the active profile from AUSPEXAI_PROFILE (the run/launch export)."""
    monkeypatch.setenv(PROFILE_ENV_VAR, "research")
    cfg = load_experiment_config(_write_profile_toml(tmp_path))  # no profile= arg
    assert cfg.active_profile == "research"
    assert cfg.driver["run_seconds"] == 28800


def test_explicit_profile_beats_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(PROFILE_ENV_VAR, "research")
    cfg = load_experiment_config(_write_profile_toml(tmp_path), profile="starter")
    assert cfg.active_profile == "starter"
    assert cfg.experiment["replication"] == 2


def test_explicit_profile_missing_file_raises(tmp_path: Path):
    with pytest.raises(ValueError, match=r"no experiment\.toml"):
        load_experiment_config(tmp_path, profile="starter")  # dir has no toml


def test_env_profile_missing_file_degrades_to_empty(tmp_path: Path, monkeypatch):
    """A stale env var must NOT break an unrelated empty-config load (only an
    EXPLICIT profile= against a missing file is a hard error)."""
    monkeypatch.setenv(PROFILE_ENV_VAR, "starter")
    cfg = load_experiment_config(tmp_path)  # dir has no toml, no explicit arg
    assert cfg.source_path is None and cfg.active_profile is None


def test_available_profiles(tmp_path: Path):
    cfg = load_experiment_config(_write_profile_toml(tmp_path))
    assert cfg.available_profiles == ["research", "starter"]
