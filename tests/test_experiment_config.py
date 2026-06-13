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
