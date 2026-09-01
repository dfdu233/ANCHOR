#!/usr/bin/env python3
"""Build a provenance-bound, fail-closed audit of ICLR-paper readiness.

The packet deliberately distinguishes engineering/evaluation qualification from
the empirical claims required by a mechanism paper.  It never promotes a
candidate because code, prompts, or reviewer packs exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "iclr-oral-completion-audit-v1"


INPUTS = {
    "scope_gate": "docs/PAPER_SCOPE_GATE_20260802.md",
    "reader_boundary_skeleton": "docs/ICLR_READER_BOUNDARY_PAPER_SKELETON_V1.md",
    "specificity_skeleton": "docs/SPECIFICITY_RATCHET_ICLR_PAPER_SKELETON_20260802.md",
    "specificity_collision_recheck": "docs/SPECIFICITY_RATCHET_FATAL_COLLISION_RECHECK_20260803.md",
    "specificity_parent_state_no_go": "docs/SPECIFICITY_RATCHET_PARENT_STATE_NO_GO_20260803.md",
    "specificity_parent_state_audit": "corrected_runs/specificity_ratchet/parent_before_constraint_audit_v1.json",
    "collision_audit": "docs/MECHANISM_COLLISION_AUDIT_CECD_SPECIFICITY_20260802.md",
    "treble_collision": "docs/TREBLE_CECD_COLLISION_PROTOCOL.md",
    "treble_collision_contract_code": "anchor/corrected_sgta/treble_collision_contract.py",
    "cecd_dual_preflight_binder": "anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py",
    "cecd_fail_closed_audit": "docs/CECD_PIPELINE_FAIL_CLOSED_AUDIT_20260803.md",
    "cecd_ecc_collision": "docs/CECD_EQUIVALENCE_CURVATURE_CANCELLATION_COLLISION_AUDIT_20260803.md",
    "pcem_gate": "docs/PCEM_CHEXCHONET_SUBSTRATE_GATE_20260803.md",
    "pcem_image_use_protocol": "docs/CAUSAL_IMAGE_USE_COMMON_PROTOCOL_V1.md",
    "pcem_collision_update": "results_reference/pcem_2026_collision_update_v1.json",
    "sisc_truth_gate_doc": "docs/SISC_OUTCOME_BLIND_TRUTH_GATE_20260803.md",
    "sisc_per_view_no_go": "docs/STUDY_IMAGE_SCOPE_ALIASING_PER_VIEW_TRUTH_NO_GO_20260803.md",
    "sisc_truth_gate_code": "anchor/corrected_sgta/build_sisc_truth_gate.py",
    "sisc_truth_gate_result": "corrected_runs/sisc_truth_gate_v1/sisc_feasibility.json",
    "claim_boundary_regrounding_code": "anchor/corrected_sgta/audit_claim_boundary_regrounding_substrate_v1.py",
    "claim_boundary_regrounding_result": "corrected_runs/claim_boundary_regrounding_audit_v1/substrate_audit.json",
    "reversible_static_law_scan": "docs/REVERSIBLE_STATIC_CLINICAL_LAW_SCAN_20260803.md",
    "specificity_pack": "corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2/summary.json",
    "natural_completion_decision": "corrected_runs/specificity_ratchet/natural_oe_diagnostic_completion_huatuo_v1/decision_v1.json",
    "cecd_pack": "corrected_runs/vindr_v2/cecd_human_admission_v2/pack_integrity.json",
    "native_oe_acceptance": "corrected_runs/unified_eval/full/native_oe_greedy256_acceptance_v1.json",
    "huatuo_oe_qualification": "corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v3_512/qualification.json",
    "method_config": "configs/unified_eval/method_ladder_v1.json",
    "method_ladder": "corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json",
    "method_evidence": "corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json",
    "artifact_registry": "corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl",
    "baseline_coverage_audit": "corrected_runs/unified_eval/provenance/baseline_coverage_audit_v5.json",
    "baseline_coverage_code": "anchor/medeval/audit_baseline_coverage_v3.py",
    "baseline_coverage_doc": "docs/BASELINE_COVERAGE_AUDIT_20260803.md",
    "internal_control_contract": "configs/unified_eval/internal_baseline_control_contract_v1.json",
    "internal_control_audit": "corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v4.json",
    "internal_control_audit_code": "anchor/medeval/audit_internal_baseline_controls_v1.py",
    "common_rag": "corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_summary.json",
    "metric_side_probe": "corrected_runs/metric_calibration_probe_v2/two_model_pilot_decision_v3.json",
    "physician_delivery": "corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/deliveries_v1/delivery_manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_records(root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for name, relative in INPUTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required audit input missing: {path}")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return records


def build_audit(root: Path) -> dict[str, Any]:
    scope = (root / INPUTS["scope_gate"]).read_text(encoding="utf-8")
    reader_skeleton = (root / INPUTS["reader_boundary_skeleton"]).read_text(
        encoding="utf-8"
    )
    specificity_collision = (
        root / INPUTS["specificity_collision_recheck"]
    ).read_text(encoding="utf-8")
    specificity_parent_no_go = (
        root / INPUTS["specificity_parent_state_no_go"]
    ).read_text(encoding="utf-8")
    treble_collision = (root / INPUTS["treble_collision"]).read_text(
        encoding="utf-8"
    )
    treble_contract_code = (
        root / INPUTS["treble_collision_contract_code"]
    ).read_text(encoding="utf-8")
    dual_preflight_binder = (
        root / INPUTS["cecd_dual_preflight_binder"]
    ).read_text(encoding="utf-8")
    cecd_ecc_collision = (root / INPUTS["cecd_ecc_collision"]).read_text(
        encoding="utf-8"
    )
    pcem_gate = (root / INPUTS["pcem_gate"]).read_text(encoding="utf-8")
    pcem_image_use_protocol = (
        root / INPUTS["pcem_image_use_protocol"]
    ).read_text(encoding="utf-8")
    sisc_gate_doc = (root / INPUTS["sisc_truth_gate_doc"]).read_text(
        encoding="utf-8"
    )
    sisc_no_go_doc = (root / INPUTS["sisc_per_view_no_go"]).read_text(
        encoding="utf-8"
    )
    reversible_static_scan = (
        root / INPUTS["reversible_static_law_scan"]
    ).read_text(encoding="utf-8")
    specificity = load_json(root / INPUTS["specificity_pack"])
    specificity_parent = load_json(root / INPUTS["specificity_parent_state_audit"])
    natural = load_json(root / INPUTS["natural_completion_decision"])
    cecd = load_json(root / INPUTS["cecd_pack"])
    native = load_json(root / INPUTS["native_oe_acceptance"])
    huatuo = load_json(root / INPUTS["huatuo_oe_qualification"])
    methods = load_json(root / INPUTS["method_ladder"])
    method_evidence = load_json(root / INPUTS["method_evidence"])
    baseline_coverage = load_json(root / INPUTS["baseline_coverage_audit"])
    internal_controls = load_json(root / INPUTS["internal_control_audit"])
    rag = load_json(root / INPUTS["common_rag"])
    metric = load_json(root / INPUTS["metric_side_probe"])
    physician_delivery = load_json(root / INPUTS["physician_delivery"])
    pcem_collision = load_json(root / INPUTS["pcem_collision_update"])
    sisc = load_json(root / INPUTS["sisc_truth_gate_result"])
    claim_boundary = load_json(root / INPUTS["claim_boundary_regrounding_result"])

    if "**Reject and Pivot.**" not in scope:
        raise ValueError("authoritative paper scope gate no longer says Reject and Pivot")
    if "SUPERSEDED — DO NOT DRAFT" not in reader_skeleton:
        raise ValueError("dead reader-boundary skeleton is not visibly superseded")
    if (
        "METHOD DOWNGRADE / MECHANISM-ONLY CONDITIONAL GO / CURRENT GPU NO-GO"
        not in specificity_collision
        or "HSC NeurIPS 2024 是直接先例" not in specificity_collision
        or "删除 nearest-ancestor decoding 的独立 novelty claim"
        not in specificity_collision
    ):
        raise ValueError("Specificity method collision boundary is stale or permissive")
    strict = specificity_parent.get("strict_parent_summaries", {})
    naming = specificity_parent.get("scientific_naming_gate", {})
    if (
        specificity_parent.get("status") != "no_go_current_pack"
        or naming.get("crossing_authorized") is not False
        or specificity_parent.get("gates", {}).get(
            "current_pack_surface_construct_certifiable"
        )
        is not False
        or strict.get("dev", {}).get("repeated_exact_constraint_blocks") != 0
        or strict.get("test", {}).get("repeated_exact_constraint_blocks") != 0
        or "NO-GO for calling the current estimator" not in specificity_parent_no_go
    ):
        raise ValueError("Specificity parent-state NO-GO boundary is stale or permissive")
    if (
        "2025.findings-emnlp.1000" not in treble_collision
        or "blocked, not reproduced" not in treble_collision
        or "same order in proceedings and source" not in treble_collision
        or "cecd-treble-dual-semantics-preflight-v1" not in treble_collision
        or "cecd-treble-dual-semantics-envelope-v1" not in treble_collision
        or "Neither is called paper-native or exact" not in treble_collision
    ):
        raise ValueError("Treble proceedings/source collision audit is stale or permissive")
    if (
        "exact-v1 self-attestation cannot resolve" not in treble_contract_code
        or "DUAL_SEMANTICS_OUTCOME_SCHEMA" not in treble_contract_code
        or "DUAL_SEMANTICS_PREFLIGHT_SCHEMA" not in treble_contract_code
        or '"paper_claim_authorized": False' not in treble_contract_code
        or '"general_gpu_authorized": False' not in dual_preflight_binder
        or '"paper_native_treble_authorized": False' not in dual_preflight_binder
        or '"method_outputs_consumed": False' not in dual_preflight_binder
    ):
        raise ValueError("Treble dual-semantics executable boundary is stale or permissive")
    if (
        "standalone algorithmic novelty claim" not in cecd_ecc_collision
        or "conditional causal probe inside CECD" not in cecd_ecc_collision
        or "full-orbit averaging" not in cecd_ecc_collision
    ):
        raise ValueError("CECD interaction-projection novelty boundary is stale")
    if (
        "ACCESS-BLOCKED / UNIDENTIFIED" not in pcem_gate
        or "pcem-echo-access-monitor-v1" not in pcem_gate
        or "gpu_authorized=false" not in pcem_gate
    ):
        raise ValueError("PCEM access/data gate is stale or compute-permissive")
    if (
        "anchor-causal-image-use-triad-v1" not in pcem_image_use_protocol
        or "representation_capture_authorized=false" not in pcem_image_use_protocol
        or "gpu_authorized=false" not in pcem_image_use_protocol
        or "No real-model run is currently authorized" not in pcem_image_use_protocol
    ):
        raise ValueError("PCEM image-use protocol is stale or compute-permissive")
    sisc_provenance = sisc.get("provenance", {})
    sisc_fingerprint_payload = sisc_provenance.get("fingerprint_payload")
    if not isinstance(sisc_fingerprint_payload, dict):
        raise ValueError("SISC truth gate lacks a fingerprint payload")
    expected_sisc_fingerprint = hashlib.sha256(
        json.dumps(
            sisc_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    sisc_gates = sisc.get("gates", {})
    if (
        sisc.get("protocol") != "sisc_outcome_blind_truth_gate_v1"
        or sisc.get("decision") != "NO-GO"
        or sisc.get("gpu_authorized") is not False
        or sisc.get("outcomes_opened") is not False
        or sisc_provenance.get("fingerprint") != expected_sisc_fingerprint
        or sisc.get("counts", {}).get("paired_studies") != 44
        or sisc.get("counts", {}).get("eligible_findings") != []
        or sisc_gates.get("patient_and_study_disjoint_split") is not True
        or any(
            sisc_gates.get(name) is not False
            for name in (
                "paired_studies_at_least_100",
                "at_least_3_eligible_findings",
                "each_eligible_finding_at_least_30",
                "absence_and_unassessable_separable",
            )
        )
        or "NO-GO before model outcomes or GPU" not in sisc_gate_doc
        or "STRICT NO-GO" not in sisc_no_go_doc
        or "missing box" not in sisc_no_go_doc
    ):
        raise ValueError("SISC scope/truth boundary is stale, unverified, or permissive")
    claim_provenance = claim_boundary.get("provenance", {})
    claim_fingerprint_payload = {
        "protocol_id": claim_boundary.get("protocol_id"),
        "auditor_sha256": claim_provenance.get("auditor_sha256"),
        "inputs": {
            name: row.get("sha256")
            for name, row in claim_provenance.get("inputs", {}).items()
        },
    }
    expected_claim_fingerprint = hashlib.sha256(
        json.dumps(
            claim_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        claim_boundary.get("protocol_id")
        != "clinical-claim-boundary-regrounding-substrate-audit-v1"
        or claim_boundary.get("status") != "no_go_current_substrate"
        or claim_boundary.get("fingerprint") != expected_claim_fingerprint
        or claim_boundary.get("gates", {}).get("gpu_authorized") is not False
        or claim_boundary.get("gates", {}).get(
            "formal_mechanism_analysis_authorized"
        )
        is not False
        or claim_boundary.get("outcome_blind_contract", {}).get(
            "sealed_confirmation_opened"
        )
        is not False
        or claim_boundary.get("scientific_naming_gate", {}).get(
            "claim_boundary_regrounding_failure_authorized"
        )
        is not False
    ):
        raise ValueError("claim-boundary substrate audit is stale or permissive")
    if (
        "ALL-NO-GO" not in reversible_static_scan
        or "no GPU experiment is authorized" not in reversible_static_scan
        or "do not launch Huatuo, Hulu or LLaVA-Med runs" not in reversible_static_scan
        or "Hard-gate failures override these scores" not in reversible_static_scan
    ):
        raise ValueError("reversible static-law scan is stale or compute-permissive")
    pcem_fingerprint = pcem_collision.get("fingerprint")
    pcem_without_fingerprint = dict(pcem_collision)
    pcem_without_fingerprint.pop("fingerprint", None)
    expected_pcem_fingerprint = hashlib.sha256(
        json.dumps(
            pcem_without_fingerprint,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    pcem_boundary = pcem_collision.get("updated_collision_boundary", {})
    if (
        pcem_fingerprint != expected_pcem_fingerprint
        or pcem_boundary.get("decision")
        != "CONDITIONALLY_OPEN_BUT_STRONGLY_NARROWED"
        or pcem_boundary.get("gpu_authorized") is not False
        or pcem_collision["sources"]["causal_image_use_triad"]["direct_collision"].get(
            "view_dependent_grounding"
        )
        != "occupied"
    ):
        raise ValueError("PCEM 2026 collision boundary is stale, unverified, or permissive")
    if natural.get("decision") != "no_go_diagnostic_completion_mechanism":
        raise ValueError("natural-completion branch decision changed unexpectedly")
    if metric.get("decision") != "STOP_AFTER_N8" or metric.get("n97_authorized"):
        raise ValueError("metric side-probe is not fail-closed")
    if not cecd.get("passed") or cecd.get("status") != "awaiting_independent_human_reviews":
        raise ValueError("CECD pack is not in the expected pre-human state")
    if not native.get("passed") or not huatuo.get("passed"):
        raise ValueError("native OE generation qualification regressed")
    native_models = {row["model"] for row in native.get("models", [])}
    if native_models != {"hulu", "llava"}:
        raise ValueError("native OE acceptance must bind Hulu and LLaVA")
    method_summary = methods.get("summary", {})
    evidence_summary = method_evidence.get("summary", {})
    if (
        method_summary.get("n") != 24
        or method_summary.get("pass") != 11
        or method_summary.get("not_admissible") != 13
        or methods.get("config_sha256")
        != sha256(root / INPUTS["method_config"])
        or method_evidence.get("t0_audit_sha256")
        != sha256(root / INPUTS["method_ladder"])
        or method_evidence.get("artifact_registry_sha256")
        != sha256(root / INPUTS["artifact_registry"])
        or evidence_summary.get("methods") != 24
        or evidence_summary.get("t0_pass") != 11
        or evidence_summary.get("t0_not_admissible") != 13
        or evidence_summary.get("stale_registry_events") != 0
        or evidence_summary.get("full_pass") != []
    ):
        raise ValueError("current 24-method ladder/evidence closure is stale or permissive")
    baseline_provenance = baseline_coverage.get("provenance", {})
    baseline_fingerprint_payload = {
        "version": baseline_coverage.get("version"),
        "inputs": {
            name: row.get("sha256")
            for name, row in baseline_provenance.items()
        },
        "method_names": [
            row.get("name") for row in baseline_coverage.get("methods", [])
        ],
        "gates": baseline_coverage.get("gates"),
    }
    expected_baseline_fingerprint = hashlib.sha256(
        json.dumps(
            baseline_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    baseline_gates = baseline_coverage.get("gates", {})
    baseline_summary = baseline_coverage.get("summary", {})
    if (
        baseline_coverage.get("version") != "baseline-coverage-audit-v3"
        or baseline_coverage.get("status") != "partial_no_efficacy_table"
        or baseline_coverage.get("fingerprint") != expected_baseline_fingerprint
        or baseline_coverage.get("paper_baseline_claim_authorized") is not False
        or baseline_summary.get("method_count") != 24
        or baseline_summary.get("t1_missing_after_t0_pass") != []
        or baseline_summary.get("t2_missing_after_t0_pass") != []
        or baseline_summary.get("t2_failed_after_t0_pass")
        != ["calibrated_abstention"]
        or baseline_summary.get("full_pass") != []
        or baseline_gates.get("configuration_closure") is not True
        or baseline_gates.get("registry_fresh") is not True
        or baseline_gates.get("source_qualification_complete") is not True
        or baseline_gates.get("t1_identity_complete_for_t0_pass") is not True
        or baseline_gates.get("t2_functional_complete_for_t0_pass") is not False
        or baseline_gates.get("clinical_claim_evaluation_complete") is not False
        or baseline_gates.get("common_rag_causal_grounding_passed") is not False
        or baseline_gates.get("report_generation_controls_passed") is not False
        or baseline_gates.get("generic_vlm_control_present") is not False
        or baseline_gates.get("internal_control_contract_enforced") is not True
        or baseline_gates.get("paper_main_table_authorized") is not False
    ):
        raise ValueError("baseline coverage audit is stale or falsely complete")
    internal_summary = internal_controls.get("summary", {})
    internal_fingerprint_payload = {
        "version": internal_controls.get("version"),
        "contract_sha256": internal_controls.get("contract", {}).get("sha256"),
        "method_evidence_sha256": internal_controls.get("method_evidence", {}).get("sha256"),
        "artifact_registry_sha256": internal_controls.get("artifact_registry", {}).get("sha256"),
        "summary": internal_summary,
    }
    expected_internal_fingerprint = hashlib.sha256(
        json.dumps(
            internal_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    expected_internal_names = [
        "temperature_length_controls",
        "self_consistency",
        "calibrated_abstention",
    ]
    baseline_internal = baseline_summary.get("internal_control_qualification", {})
    if (
        internal_controls.get("version")
        != "internal-baseline-control-qualification-audit-v1"
        or internal_controls.get("status") != "partial_fail_closed"
        or internal_controls.get("paper_control_claim_authorized") is not False
        or internal_controls.get("fingerprint") != expected_internal_fingerprint
        or internal_controls.get("contract", {}).get("sha256")
        != sha256(root / INPUTS["internal_control_contract"])
        or internal_controls.get("method_evidence", {}).get("sha256")
        != sha256(root / INPUTS["method_evidence"])
        or internal_controls.get("artifact_registry", {}).get("sha256")
        != sha256(root / INPUTS["artifact_registry"])
        or [row.get("name") for row in internal_controls.get("methods", [])]
        != expected_internal_names
        or internal_summary.get("controls") != 3
        or internal_summary.get("t1_pass") != expected_internal_names
        or internal_summary.get("t2_pass") != expected_internal_names[:2]
        or internal_summary.get("t2_missing") != []
        or internal_summary.get("t2_failed") != expected_internal_names[2:]
        or internal_summary.get("t3_pass") != []
        or internal_summary.get("full_pass") != []
        or internal_summary.get("stale_registry_events") != 0
        or baseline_internal.get("status") != "partial_fail_closed"
        or baseline_internal.get("t2_pass") != expected_internal_names[:2]
        or baseline_internal.get("t2_missing") != []
        or baseline_internal.get("t2_failed") != expected_internal_names[2:]
        or baseline_internal.get("full_pass") != []
        or baseline_internal.get("fingerprint")
        != internal_controls.get("fingerprint")
    ):
        raise ValueError("internal baseline controls are stale or falsely promoted")
    if rag.get("dataset_pooling_forbidden") is not True:
        raise ValueError("common-RAG summary no longer forbids dataset pooling")

    requirements = [
        {
            "id": "R1",
            "requirement": "Independent clinical admission of a natural, important hallucination construct",
            "status": "pending_external",
            "evidence": (
                f"The current Specificity pack has {specificity['selection']['selected_images']} "
                f"images and {specificity['n_edges']} candidate edges, but its pre-outcome "
                "parent-state audit is NO-GO and cannot identify a crossing. CECD reviewer "
                "sheets are blank and awaiting independent reviews."
            ),
        },
        {
            "id": "R2",
            "requirement": "Held-out mechanism replicated in at least two primary model families",
            "status": "missing",
            "evidence": (
                "Reader-residual Early erasure failed before confirmation; the current "
                "Specificity substrate failed parent-state identifiability; CECD has no "
                "admitted real-model mechanism result."
            ),
        },
        {
            "id": "R3",
            "requirement": "Separating causal intervention with strong random/null controls",
            "status": "missing",
            "evidence": "No surviving pivot has passed its observational gate, so causal scoring is unauthorized.",
        },
        {
            "id": "R4",
            "requirement": "OE/report mitigation improves clinical fidelity at matched claim count and coverage",
            "status": "missing",
            "evidence": (
                "No fixed-K/no-exchange method is authorized; lexical mitigation outputs do not "
                "define clinical hallucination efficacy."
            ),
        },
        {
            "id": "R5",
            "requirement": "Unified, identity-qualified CE/OE generation and evaluation substrate",
            "status": "engineering_complete_clinical_metrics_pending",
            "evidence": (
                "Huatuo, Hulu, and LLaVA OE generation artifacts pass response-form gates; "
                "clinical claim efficacy remains explicitly pending."
                " The causal image-use common protocol now freezes the same four-condition, "
                "cluster-bootstrap and provenance gates across model backends, but has no "
                "authorized real-model result."
            ),
        },
        {
            "id": "R6",
            "requirement": "Recent decoding and RAG baselines are source-qualified and honestly reported",
            "status": "source_complete_execution_partial",
            "evidence": (
                f"The current T0 audit closes all {method_summary['n']} configured methods: "
                f"{method_summary['pass']} executable and {method_summary['not_admissible']} "
                "source/checkpoint/license/architecture exclusions. All executable methods pass "
                "T1; temperature/length and self-consistency pass functional T2, while calibrated "
                "abstention was executed but failed its frozen per-model non-degeneracy cutoff "
                "without outcome-driven retuning. No mitigation passes full clinical efficacy; "
                "common RAG has zero "
                "causal-grounding cells; report controls and the generic-VLM control are absent. Exact "
                "Treble remains blocked after proceedings/source revalidation and is not "
                "impersonated by a local surrogate. A dual-semantics common-protocol "
                "envelope is frozen as the only fallback, but has no authorized run or "
                "outcome and does not claim exact paper-native reproduction."
            ),
        },
        {
            "id": "R7",
            "requirement": "Independent physician evaluation of hallucination, omission, refusal, hedging, and usefulness",
            "status": "pending_external",
            "evidence": (
                f"The role-isolated physician delivery contains "
                f"{physician_delivery['reviewers']['A']['groups']} groups and "
                f"{physician_delivery['reviewers']['A']['answer_units']} answer units per "
                "reviewer; no returned labels are used here."
            ),
        },
        {
            "id": "R8",
            "requirement": "Collision-resistant novelty and an oral-level claim ceiling",
            "status": "conditional_only",
            "evidence": (
                "CECD is the first operational candidate and retains only centered cross-modal "
                "non-separability. Specificity remains a future problem formulation only: HSC "
                "occupies ancestor retreat, CEBC/ZINA/CoEV occupy evidence-bounded editing, "
                "and the current pack cannot observe a parent-to-child state transition. "
                "CECD interaction-component projection is a classical additive-subspace "
                "operator and is authorized only as a conditional causal probe, not a method. "
                "PCEM has a plausible independent-truth substrate but remains access-blocked and "
                "cannot enter the ranking before its CPU join and construct gates pass; generic "
                "view-dependent image grounding and cardiomegaly grounding are already occupied "
                "by a 2026 causal audit."
                " A backend-neutral executable qualification gate exists, but it explicitly "
                "cannot authorize representation capture or GPU work."
            ),
        },
        {
            "id": "R9",
            "requirement": "Complete paper figures, tables, manuscript, and reproducibility release",
            "status": "not_started_by_design",
            "evidence": (
                "Affirmative paper sections are prohibited until a pivot passes construct and "
                "mechanism gates; the obsolete reader-boundary skeleton is superseded."
            ),
        },
    ]

    candidates = [
        {
            "rank": 1,
            "name": "Clinical-Equivalence Composition Defect",
            "paper_type": "New Setting",
            "verdict": "CONDITIONAL_GO_PENDING_HUMAN_ADMISSION",
            "why": (
                "A centered mixed derivative across independently admitted render and "
                "language equivalences remains distinct from one-axis paraphrase work."
            ),
            "next_gate": "Four role-isolated human returns must pass the frozen CECD admission analyzer.",
            "oral_ceiling_requires": (
                "replicated non-additive clinical-error information in Huatuo and Hulu plus "
                "a causal explanation beyond prompt or render main effects"
            ),
        },
        {
            "rank": 2,
            "name": "Specificity Ratchet (future substrate only)",
            "paper_type": "New Problem + Mechanism",
            "verdict": "CURRENT_SUBSTRATE_NO_GO_REDESIGN_REQUIRED",
            "why": (
                "The abstract supported-parent to unsupported-descendant problem remains "
                "interesting, but the current deletion-derived parent construction cannot "
                "identify a native crossing. Ancestor backoff itself is occupied."
            ),
            "next_gate": (
                "Build a new outcome-blind substrate with an observed or held-out-validated "
                "parent state, at least ten repeated semantic blocks per split, and at least "
                "three edge types before any G0 or GPU work."
            ),
            "oral_ceiling_requires": (
                "native parent preservation plus constraint reversal, crossing-specific causal "
                "rescue, two model families, at least three edge types, and fixed-content "
                "clinical improvement beyond HSC and CEBC/ZINA/CoEV-style correction"
            ),
            "ancestor_backoff_method_novelty_authorized": False,
        },
        {
            "rank": 3,
            "name": "Unified physician OE/mitigation audit",
            "paper_type": "Supporting evaluation module",
            "verdict": "RETAIN_AS_CONTROL_NOT_MAINLINE",
            "why": (
                "It can establish length/omission/refusal exchanges and qualify baselines, "
                "but it does not by itself supply a new mechanism."
            ),
            "next_gate": "Two independent reviews, clarification freeze, and blinded adjudication.",
            "oral_ceiling_requires": "integration beneath a separately validated mechanism",
        },
    ]

    complete = {"complete", "qualified_with_declared_exclusions"}
    paper_ready = all(row["status"] in complete for row in requirements)
    if paper_ready:
        raise AssertionError("completion audit must not silently promote the current project")
    return {
        "version": VERSION,
        "paper_ready": False,
        "submission_claim_authorized": False,
        "current_authoritative_verdict": "REJECT_CURRENT_READER_BOUNDARY_AND_PIVOT",
        "requirements": requirements,
        "candidate_ranking": candidates,
        "killed_branches": {
            "reader_early_erasure": "data-refuted at frozen Huatuo development gate",
            "natural_diagnostic_completion": natural["decision"],
            "metric_calibration_state": metric["decision"],
            "specificity_current_substrate": "no_go_parent_state_not_identifiable",
            "exact_treble_reproduction": "blocked_unresolved_proceedings_source_semantics_and_license",
            "CBD": "permanently prohibited by cross-model fabrication/omission regressions",
            "study_image_scope_aliasing": (
                "no_go_current_substrate_missing_independent_paired_three_state_truth"
            ),
            "claim_boundary_regrounding": (
                "no_go_current_artifacts_missing_independent_truth_and_claim_local_counterfactual"
            ),
            "reversible_static_clinical_law_scan": "all_candidates_failed_hard_gates_no_gpu",
        },
        "blocked_contingencies": {
            "projection_conditioned_evidence_misbinding": {
                "status": "ACCESS_BLOCKED_UNIDENTIFIED",
                "collision_boundary": pcem_boundary["decision"],
                "next_automatic_step": "CPU-only MIMIC-CXR AP/PA to MIMIC-IV-ECHO temporal join audit",
                "image_download_authorized": False,
                "gpu_authorized": False,
                "promotion_condition": (
                    "independent echo construct and positive/negative/borderline bins require "
                    "cardiology/radiology review after the count gate"
                ),
            }
        },
        "inputs": input_records(root),
        "human_labels_synthesized": False,
        "next_action": (
            "wait_for_CECD_independent_returns; retain current Specificity returns as "
            "construct-only and never authorize its GPU chain"
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# ICLR Oral completion audit",
        "",
        f"- Current verdict: `{audit['current_authoritative_verdict']}`",
        f"- Paper ready: `{str(audit['paper_ready']).lower()}`",
        f"- Submission claim authorized: `{str(audit['submission_claim_authorized']).lower()}`",
        "",
        "Engineering completion is not treated as scientific completion. A reviewer pack,",
        "runtime gate, or passing unit test cannot substitute for an independently grounded",
        "phenomenon, a two-model mechanism, a causal intervention, and no-exchange utility.",
        "",
        "## Requirement-to-evidence matrix",
        "",
        "| ID | Requirement | Status | Current evidence |",
        "|---|---|---|---|",
    ]
    for row in audit["requirements"]:
        lines.append(
            f"| {row['id']} | {row['requirement']} | `{row['status']}` | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Surviving conditional pivots",
            "",
            "| Rank | Candidate | Verdict | Why it remains | Next gate |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in audit["candidate_ranking"]:
        lines.append(
            f"| {row['rank']} | {row['name']} | `{row['verdict']}` | "
            f"{row['why']} | {row['next_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Access-blocked contingencies",
            "",
            "PCEM is not a surviving ranked mechanism yet. Its persistent monitor may run only ",
            "the CPU schema/temporal-join gate after an authorized MIMIC-IV-ECHO mount; image ",
            "download and GPU inference remain unauthorized until the clinical construct is admitted.",
        ]
    )
    lines.extend(
        [
            "",
            "## Claim firewall",
            "",
            "- The old reader-boundary skeleton is historical design provenance, not a draft.",
            "- No positive mechanism, mitigation, clinical efficacy, or patient-mm claim is authorized.",
            "- The current Specificity pack is construct-only and cannot launch a GPU job even if its reviews return.",
            "- The next GPU job may be launched only by the frozen gate corresponding to a valid human return.",
            "- Missing releases remain `not_admissible`; local ports do not impersonate paper-native baselines.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    audit = build_audit(root)
    audit_path = output / "audit.json"
    markdown_path = output / "EVIDENCE.md"
    atomic_write(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_write(markdown_path, render_markdown(audit) + "\n")
    manifest = {
        "version": VERSION,
        "paper_ready": False,
        "code_sha256": sha256(Path(__file__)),
        "outputs": {
            audit_path.name: {"sha256": sha256(audit_path), "bytes": audit_path.stat().st_size},
            markdown_path.name: {
                "sha256": sha256(markdown_path),
                "bytes": markdown_path.stat().st_size,
            },
        },
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
