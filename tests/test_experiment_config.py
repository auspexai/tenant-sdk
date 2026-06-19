"""experiment.toml loader + make_unique_label (Part A)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from auspexai_tenant.experiment_config import (
    load_experiment_config,
    make_unique_label,
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
