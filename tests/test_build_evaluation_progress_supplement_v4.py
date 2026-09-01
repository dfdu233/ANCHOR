from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import anchor.medeval.build_evaluation_progress_supplement_v4 as module


def _fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    names = (
        "base_audit", "rag_audit", "baseline_audit", "internal_generation", "internal_form",
        "internal_postprocess", "internal_huatuo_archive", "internal_hulu_archive",
        "internal_huatuo_monitor", "internal_hulu_monitor", "llava_generation",
        "llava_postprocess", "llava_archive", "llava_monitor", "system_handoff",
        "system_huatuo_canary", "system_hulu_canary", "system_job_state",
    )
    paths = {name: tmp_path / f"{name}.json" for name in names}
    for path in paths.values():
        path.write_text("{}")

    def digest(path: Path) -> str:
        return f"sha:{Path(path).stem}"

    arms = {f"arm_{index}": {"passed": True, "rows": 120} for index in range(9)}
    archive = {
        "version": "anchor-physician-oe-review-archive-v2", "passed": True,
        "delivery_index_sha256": "delivery", "archives": [{"passed": True}],
    }
    monitor = {
        "version": "anchor-physician-oe-clinical-pipeline-monitor-v1",
        "clinical_labels_synthesized": False,
        "private_mapping_joined_before_consensus": False,
        "stage": "waiting_for_independent_reviews", "missing": ["reviewer_A"],
    }
    records = {
        "base_audit": {"paper_ready": False, "submission_claim_authorized": False, "human_labels_synthesized": False},
        "rag_audit": {
            "protocol_version": "rag-dual-track-qualification-v2", "tracks_kept_separate": True,
            "any_rag_efficacy_authorized": False, "factmm_released_asset_role": "generator",
        },
        "baseline_audit": {
            "status": "partial_no_efficacy_table", "paper_baseline_claim_authorized": False,
            "summary": {"full_pass": []},
        },
        "internal_generation": {
            "protocol_version": "internal-control-generation-t3-audit-v2", "stage": "T3", "passed": True,
            "reference_answers_used_for_qualification": False, "clinical_labels_used_for_qualification": False,
            "models": {name: {"passed": True, "arms": deepcopy(arms)} for name in ("huatuo", "hulu")},
        },
        "internal_form": {
            "protocol_version": "internal-control-generation-form-audit-v1", "passed": True,
            "physician_pack_operationally_authorized": True, "clinical_efficacy_authorized": False,
            "reference_answers_used_for_qualification": False, "clinical_labels_used_for_qualification": False,
            "records": [{"eligible": True, "reference_fields_accessed_by_auditor": False} for _ in range(18)],
        },
        "internal_postprocess": {
            "protocol_version": "internal-controls-t3-postprocess-v2", "physician_labels_present": False,
            "clinical_efficacy_authorized": False, "generation_audit_sha256": digest(paths["internal_generation"]),
            "generation_form_audit_sha256": digest(paths["internal_form"]),
            "review_packs": {
                name: {"archive_verification_sha256": digest(paths[f"internal_{name}_archive"]), "delivery_index_sha256": "delivery"}
                for name in ("huatuo", "hulu")
            },
        },
        "internal_huatuo_archive": deepcopy(archive), "internal_hulu_archive": deepcopy(archive),
        "internal_huatuo_monitor": deepcopy(monitor), "internal_hulu_monitor": deepcopy(monitor),
        "llava_generation": {
            "protocol_version": "llava-mitigation-t3-generation-audit-v1", "all_operational_gates_passed": True,
            "physician_pack_authorized": True, "clinical_efficacy_authorized": False,
            "reference_answers_used": False, "clinical_labels_used": False,
            "method_off_identity": {"passed": True, "generated_token_exact_rate": 1.0},
            "method_records": [
                {"rows": 120, "eligible": True, "trace_complete": True, "cap_hits": 0, "reference_fields_absent": True}
                for _ in range(10)
            ],
        },
        "llava_postprocess": {
            "protocol_version": "llava-mitigation-t3-physician-postprocess-v1",
            "generation_operationally_qualified": True, "clinical_labels_present": False,
            "clinical_efficacy_authorized": False, "generation_audit_sha256": digest(paths["llava_generation"]),
            "archive_verification_sha256": digest(paths["llava_archive"]), "delivery_index_sha256": "delivery",
        },
        "llava_archive": deepcopy(archive), "llava_monitor": deepcopy(monitor),
        "system_handoff": {
            "schema_version": "cecd-system-pih-native-eager-canary-handoff-v3-family-specific-query-shape",
            "memory_correction": {"scientific_tolerance_unchanged": True},
            "source_bindings": {"runtime": {"sha256": "runtime"}},
        },
        "system_huatuo_canary": {
            "schema_version": "cecd-system-pih-native-eager-canary-artifact-v1", "model_family": "huatuo",
            "integration_source": {"sha256": "runtime"}, "selected_heads_consumed": False,
            "status": "native_eager_canary_passed", "result": {"passed": True, "max_absolute_error": 0.0},
        },
        "system_hulu_canary": {
            "schema_version": "cecd-system-pih-native-eager-canary-artifact-v1", "model_family": "hulu",
            "integration_source": {"sha256": "runtime"}, "selected_heads_consumed": False,
            "status": "failed", "result": {"passed": False, "argmax_equal": True, "max_absolute_error": 3.0},
        },
        "system_job_state": {"status": "failed", "exit_code": 2},
    }
    monkeypatch.setattr(module, "_load", lambda path: deepcopy(records[Path(path).stem]))
    monkeypatch.setattr(module, "sha256_file", digest)
    return paths, records


def test_v4_binds_engineering_completion_without_authorizing_efficacy(tmp_path, monkeypatch) -> None:
    paths, _ = _fixtures(tmp_path, monkeypatch)
    result = module.build(**paths)
    assert result["evaluation_state"]["internal_t3_generation_qualified"] is True
    assert result["evaluation_state"]["llava_t3_generation_qualified"] is True
    assert result["evaluation_state"]["any_decoding_efficacy_authorized"] is False
    assert result["clinical_efficacy_authorized"] is False
    assert result["paper_ready"] is False
    assert result["evaluation_state"]["system_pih"]["cross_model_common_protocol_admissible"] is False


def test_v4_rejects_synthesized_clinical_labels(tmp_path, monkeypatch) -> None:
    paths, records = _fixtures(tmp_path, monkeypatch)
    records["llava_monitor"]["clinical_labels_synthesized"] = True
    monkeypatch.setattr(module, "_load", lambda path: deepcopy(records[Path(path).stem]))
    with pytest.raises(ValueError, match="labels synthesized"):
        module.build(**paths)


def test_v4_rejects_relaxed_or_improperly_passing_hulu_canary(tmp_path, monkeypatch) -> None:
    paths, records = _fixtures(tmp_path, monkeypatch)
    records["system_handoff"]["memory_correction"]["scientific_tolerance_unchanged"] = False
    monkeypatch.setattr(module, "_load", lambda path: deepcopy(records[Path(path).stem]))
    with pytest.raises(ValueError, match="tolerance changed"):
        module.build(**paths)

    records["system_handoff"]["memory_correction"]["scientific_tolerance_unchanged"] = True
    records["system_hulu_canary"]["status"] = "native_eager_canary_passed"
    records["system_hulu_canary"]["result"]["passed"] = True
    with pytest.raises(ValueError, match="must remain failed"):
        module.build(**paths)
