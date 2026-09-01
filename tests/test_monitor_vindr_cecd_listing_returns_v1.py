from pathlib import Path

from scripts.monitor_vindr_cecd_listing_returns_v1 import (
    advance,
    return_paths,
    unexpected_files,
)
import scripts.monitor_vindr_cecd_listing_returns_v1 as monitor_module


def test_exact_four_role_filenames(tmp_path: Path) -> None:
    completed, attestations = return_paths(tmp_path)
    assert [path.name for path in completed.values()] == [
        "clinical_reviewer_1.completed.csv",
        "clinical_reviewer_2.completed.csv",
        "clinical_template_reviewer.completed.csv",
        "language_reviewer.completed.csv",
    ]
    assert [path.name for path in attestations.values()] == [
        "clinical_reviewer_1.attestation.json",
        "clinical_reviewer_2.attestation.json",
        "clinical_template_reviewer.attestation.json",
        "language_reviewer.attestation.json",
    ]


def test_unexpected_aliases_fail_closed_but_partial_copy_is_permitted(
    tmp_path: Path,
) -> None:
    completed, attestations = return_paths(tmp_path)
    allowed = {path.name for path in [*completed.values(), *attestations.values()]}
    (tmp_path / "language_annotator.completed.csv").write_text("wrong alias")
    (tmp_path / "clinical_reviewer_1.completed.csv.partial").write_text("copying")
    assert unexpected_files(tmp_path, allowed) == [
        "language_annotator.completed.csv"
    ]


def test_partial_returns_only_wait_and_never_validate(tmp_path: Path) -> None:
    completed, attestations = return_paths(tmp_path)
    next(iter(completed.values())).write_text("partial human return")
    state = advance(
        pack=tmp_path / "unused-pack",
        completed=completed,
        attestations=attestations,
    )
    assert state["stage"] == "waiting_for_four_independent_returns"
    assert state["returns_present"] == 1
    assert state["returns_required"] == 8
    assert "admission_decision_computed" not in state


def test_valid_returns_transition_only_to_human_adjudication_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    completed, attestations = return_paths(tmp_path / "inbox")
    for path in [*completed.values(), *attestations.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("human input\n", encoding="utf-8")
    monkeypatch.setattr(
        monitor_module,
        "validate_all",
        lambda **_: {"roles": [{"role": role, "rows": 1} for role in completed]},
    )
    monkeypatch.setattr(
        monitor_module,
        "prepare_handoff",
        lambda **_: {"fingerprint": "f" * 64},
    )
    handoff = tmp_path / "handoff"
    state = advance(
        pack=tmp_path / "pack",
        completed=completed,
        attestations=attestations,
        handoff_dir=handoff,
    )
    assert state["stage"] == "ready_for_human_adjudication"
    assert state["adjudication_still_required"] is True
    assert state["admission_receipt_created"] is False
    assert state["admission_decision_computed"] is False
