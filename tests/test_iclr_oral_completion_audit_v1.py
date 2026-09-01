from pathlib import Path

from anchor.corrected_sgta.build_iclr_oral_completion_audit_v1 import (
    build_audit,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_audit_is_fail_closed_and_preserves_surviving_order() -> None:
    audit = build_audit(ROOT)
    assert audit["paper_ready"] is False
    assert audit["submission_claim_authorized"] is False
    assert audit["candidate_ranking"][0]["name"] == "Clinical-Equivalence Composition Defect"
    assert audit["candidate_ranking"][0]["paper_type"] == "New Setting"
    assert (
        audit["candidate_ranking"][1][
            "ancestor_backoff_method_novelty_authorized"
        ]
        is False
    )
    assert (
        audit["candidate_ranking"][1]["verdict"]
        == "CURRENT_SUBSTRATE_NO_GO_REDESIGN_REQUIRED"
    )
    statuses = {row["id"]: row["status"] for row in audit["requirements"]}
    assert statuses["R1"] == "pending_external"
    assert statuses["R2"] == "missing"
    assert statuses["R4"] == "missing"
    assert statuses["R6"] == "source_complete_execution_partial"
    assert "Exact Treble remains blocked" in next(
        row["evidence"] for row in audit["requirements"] if row["id"] == "R6"
    )
    assert "dual-semantics common-protocol envelope" in next(
        row["evidence"] for row in audit["requirements"] if row["id"] == "R6"
    )
    assert audit["killed_branches"]["exact_treble_reproduction"].startswith(
        "blocked_"
    )
    assert audit["killed_branches"]["study_image_scope_aliasing"].startswith(
        "no_go_current_substrate"
    )
    assert audit["killed_branches"]["claim_boundary_regrounding"].startswith(
        "no_go_current_artifacts"
    )
    assert (
        audit["killed_branches"]["reversible_static_clinical_law_scan"]
        == "all_candidates_failed_hard_gates_no_gpu"
    )
    assert "treble_collision_contract_code" in audit["inputs"]
    assert "cecd_dual_preflight_binder" in audit["inputs"]
    assert (
        audit["killed_branches"]["specificity_current_substrate"]
        == "no_go_parent_state_not_identifiable"
    )
    pcem = audit["blocked_contingencies"][
        "projection_conditioned_evidence_misbinding"
    ]
    assert pcem["status"] == "ACCESS_BLOCKED_UNIDENTIFIED"
    assert pcem["collision_boundary"] == "CONDITIONALLY_OPEN_BUT_STRONGLY_NARROWED"
    assert pcem["image_download_authorized"] is False
    assert pcem["gpu_authorized"] is False
    assert "pcem_image_use_protocol" in audit["inputs"]
    assert "sisc_truth_gate_result" in audit["inputs"]
    assert "sisc_truth_gate_code" in audit["inputs"]
    assert "claim_boundary_regrounding_result" in audit["inputs"]
    assert "claim_boundary_regrounding_code" in audit["inputs"]
    assert "reversible_static_law_scan" in audit["inputs"]
    assert "method_config" in audit["inputs"]
    assert "method_evidence" in audit["inputs"]
    assert "artifact_registry" in audit["inputs"]
    assert "baseline_coverage_audit" in audit["inputs"]
    assert "baseline_coverage_code" in audit["inputs"]
    assert "internal_control_contract" in audit["inputs"]
    assert "internal_control_audit" in audit["inputs"]
    assert "internal_control_audit_code" in audit["inputs"]
    r5 = next(row for row in audit["requirements"] if row["id"] == "R5")
    assert "causal image-use common protocol" in r5["evidence"]
    assert audit["human_labels_synthesized"] is False


def test_markdown_does_not_convert_engineering_into_a_paper_claim() -> None:
    text = render_markdown(build_audit(ROOT))
    assert "Engineering completion is not treated as scientific completion" in text
    assert "No positive mechanism" in text
    assert "PCEM is not a surviving ranked mechanism yet" in text
    assert "current Specificity pack is construct-only" in text
    assert "SUPERSEDED" not in text  # the packet points to the firewall, not stale prose
