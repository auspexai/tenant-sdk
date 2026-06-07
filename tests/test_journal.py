"""Tests for the run-journal + resume reconstruction (M8 §3.4)."""

from __future__ import annotations

from auspexai_tenant.journal import RunJournal, resume_state


def _journal(tmp_path) -> RunJournal:
    return RunJournal(tmp_path / "run.journal")


def test_append_and_records_roundtrip(tmp_path) -> None:
    j = _journal(tmp_path)
    j.record_submit(0, [{"unit_id": "u1", "payload": {"v": 1}}, {"unit_id": "u2", "payload": {}}])
    j.record_round_done(0, "cursor-1", {"count": 2})
    recs = j.records()
    assert [r["kind"] for r in recs] == ["submit", "round_done"]
    assert [u["unit_id"] for u in recs[0]["units"]] == ["u1", "u2"]
    assert recs[0]["units"][0]["payload"] == {"v": 1}
    assert recs[1]["cursor"] == "cursor-1"
    assert recs[1]["aggregate"] == {"count": 2}


def test_records_empty_when_no_file(tmp_path) -> None:
    j = _journal(tmp_path)
    assert j.records() == []
    assert j.exists() is False


def test_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "run.journal"
    RunJournal(path).record_finalized()
    assert [r["kind"] for r in RunJournal(path).records()] == ["finalized"]


# ---- resume_state ----------------------------------------------------------


def test_resume_fresh() -> None:
    st = resume_state([])
    assert st.resumed is False
    assert st.terminal is None
    assert st.current_round == 0
    assert st.cursor is None
    assert st.aggregate is None
    assert st.pinned_units is None


def test_resume_after_completed_round() -> None:
    j_records = [
        {"kind": "submit", "round": 0, "units": [{"unit_id": "u1", "payload": {}}]},
        {"kind": "round_done", "round": 0, "cursor": "c1", "aggregate": {"count": 1}},
    ]
    st = resume_state(j_records)
    assert st.resumed is True
    assert st.terminal is None
    assert st.current_round == 1  # next round
    assert st.cursor == "c1"
    assert st.aggregate == {"count": 1}
    assert st.pinned_units is None  # round 1 not yet submitted


def test_resume_with_in_flight_submit() -> None:
    # Round 0 done; round 1 submitted but no round_done (crashed mid-round).
    records = [
        {"kind": "submit", "round": 0, "units": [{"unit_id": "u0", "payload": {}}]},
        {"kind": "round_done", "round": 0, "cursor": "c0", "aggregate": {"count": 1}},
        {
            "kind": "submit",
            "round": 1,
            "units": [{"unit_id": "u1", "payload": {"v": 1}}, {"unit_id": "u2", "payload": {}}],
        },
    ]
    st = resume_state(records)
    assert st.current_round == 1
    assert st.cursor == "c0"  # boundary is the last round_done
    assert st.aggregate == {"count": 1}
    assert [u["unit_id"] for u in st.pinned_units] == ["u1", "u2"]  # re-submit these idempotently
    assert st.pinned_units[0]["payload"] == {"v": 1}


def test_resume_terminal_finalized() -> None:
    records = [
        {"kind": "submit", "round": 0, "units": [{"unit_id": "u0", "payload": {}}]},
        {"kind": "round_done", "round": 0, "cursor": "c0", "aggregate": {}},
        {"kind": "finalized"},
    ]
    assert resume_state(records).terminal == "finalized"


def test_resume_terminal_aborted() -> None:
    assert resume_state([{"kind": "aborted"}]).terminal == "aborted"


def test_resume_latest_round_done_wins() -> None:
    records = [
        {"kind": "round_done", "round": 0, "cursor": "c0", "aggregate": {"n": 1}},
        {"kind": "round_done", "round": 1, "cursor": "c1", "aggregate": {"n": 2}},
    ]
    st = resume_state(records)
    assert st.current_round == 2
    assert st.cursor == "c1"
    assert st.aggregate == {"n": 2}
