"""Bind completed T3 engineering evidence without promoting clinical efficacy.

This supplement is intentionally fail-closed: operationally qualified generations and
verified physician archives are prerequisites for, not substitutes for, independent
clinical review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_monitor(record: dict[str, Any], name: str) -> None:
    _require(
        record.get("version") == "anchor-physician-oe-clinical-pipeline-monitor-v1",
        f"{name}: unexpected monitor protocol",
    )
    _require(record.get("clinical_labels_synthesized") is False, f"{name}: labels synthesized")
    _require(
        record.get("private_mapping_joined_before_consensus") is False,
        f"{name}: private mapping joined before consensus",
    )
    _require(record.get("stage") == "waiting_for_independent_reviews", f"{name}: unexpected stage")
    _require(bool(record.get("missing")), f"{name}: missing real-review dependencies not recorded")


def _validate_archive(record: dict[str, Any], name: str) -> None:
    _require(record.get("version") == "anchor-physician-oe-review-archive-v2", f"{name}: bad archive version")
    _require(record.get("passed") is True, f"{name}: archive verification failed")
    _require(bool(record.get("archives")), f"{name}: no archives")
    _require(all(row.get("passed") is True for row in record["archives"]), f"{name}: bad archive")


def build(
    *,
    base_audit: Path,
    rag_audit: Path,
    baseline_audit: Path,
    internal_generation: Path,
    internal_form: Path,
    internal_postprocess: Path,
    internal_huatuo_archive: Path,
    internal_hulu_archive: Path,
    internal_huatuo_monitor: Path,
    internal_hulu_monitor: Path,
    llava_generation: Path,
    llava_postprocess: Path,
    llava_archive: Path,
    llava_monitor: Path,
    system_handoff: Path,
    system_huatuo_canary: Path,
    system_hulu_canary: Path,
    system_job_state: Path,
) -> dict[str, Any]:
    paths = {
        "base_audit": base_audit,
        "rag_audit": rag_audit,
        "baseline_audit": baseline_audit,
        "internal_generation": internal_generation,
        "internal_form": internal_form,
        "internal_postprocess": internal_postprocess,
        "internal_huatuo_archive": internal_huatuo_archive,
        "internal_hulu_archive": internal_hulu_archive,
        "internal_huatuo_monitor": internal_huatuo_monitor,
        "internal_hulu_monitor": internal_hulu_monitor,
        "llava_generation": llava_generation,
        "llava_postprocess": llava_postprocess,
        "llava_archive": llava_archive,
        "llava_monitor": llava_monitor,
        "system_handoff": system_handoff,
        "system_huatuo_canary": system_huatuo_canary,
        "system_hulu_canary": system_hulu_canary,
        "system_job_state": system_job_state,
    }
    data = {name: _load(path) for name, path in paths.items()}

    base = data["base_audit"]
    _require(base.get("paper_ready") is False, "base audit must remain not paper-ready")
    _require(base.get("submission_claim_authorized") is False, "base audit authorizes submission")
    _require(base.get("human_labels_synthesized") is False, "base audit synthesized human labels")

    rag = data["rag_audit"]
    _require(rag.get("protocol_version") == "rag-dual-track-qualification-v2", "bad RAG audit")
    _require(rag.get("tracks_kept_separate") is True, "RAG tracks were mixed")
    _require(rag.get("any_rag_efficacy_authorized") is False, "RAG efficacy improperly authorized")

    baseline = data["baseline_audit"]
    _require(baseline.get("status") == "partial_no_efficacy_table", "unexpected baseline state")
    _require(baseline.get("paper_baseline_claim_authorized") is False, "baseline claim authorized")
    _require(not baseline.get("summary", {}).get("full_pass"), "baseline audit contains a full pass")

    generation = data["internal_generation"]
    _require(generation.get("protocol_version") == "internal-control-generation-t3-audit-v2", "bad internal T3 audit")
    _require(generation.get("stage") == "T3" and generation.get("passed") is True, "internal T3 failed")
    _require(generation.get("reference_answers_used_for_qualification") is False, "internal references used")
    _require(generation.get("clinical_labels_used_for_qualification") is False, "internal clinical labels used")
    _require(set(generation.get("models", {})) == {"huatuo", "hulu"}, "internal model set changed")
    for model_name, model in generation["models"].items():
        _require(model.get("passed") is True, f"{model_name}: internal generation failed")
        _require(len(model.get("arms", {})) == 9, f"{model_name}: expected nine arms")
        for arm_name, arm in model["arms"].items():
            _require(arm.get("passed") is True and arm.get("rows") == 120, f"{model_name}/{arm_name}: bad arm")

    form = data["internal_form"]
    _require(form.get("protocol_version") == "internal-control-generation-form-audit-v1", "bad form audit")
    _require(form.get("passed") is True, "internal generation-form audit failed")
    _require(form.get("physician_pack_operationally_authorized") is True, "internal pack not authorized")
    _require(form.get("clinical_efficacy_authorized") is False, "internal efficacy authorized")
    _require(form.get("reference_answers_used_for_qualification") is False, "form audit used references")
    _require(form.get("clinical_labels_used_for_qualification") is False, "form audit used clinical labels")
    _require(len(form.get("records", [])) == 18, "internal form audit must contain 18 records")
    _require(all(row.get("eligible") is True for row in form["records"]), "ineligible internal form")
    _require(
        all(row.get("reference_fields_accessed_by_auditor") is False for row in form["records"]),
        "co-resident references were accessed",
    )

    internal_post = data["internal_postprocess"]
    _require(internal_post.get("protocol_version") == "internal-controls-t3-postprocess-v2", "bad internal postprocess")
    _require(internal_post.get("physician_labels_present") is False, "internal physician labels already present")
    _require(internal_post.get("clinical_efficacy_authorized") is False, "internal efficacy authorized")
    _require(internal_post.get("generation_audit_sha256") == sha256_file(internal_generation), "internal generation hash drift")
    _require(internal_post.get("generation_form_audit_sha256") == sha256_file(internal_form), "internal form hash drift")
    for model_name, archive_key in (("huatuo", "internal_huatuo_archive"), ("hulu", "internal_hulu_archive")):
        archive = data[archive_key]
        _validate_archive(archive, model_name)
        review = internal_post.get("review_packs", {}).get(model_name, {})
        _require(review.get("archive_verification_sha256") == sha256_file(paths[archive_key]), f"{model_name}: archive hash drift")
        _require(review.get("delivery_index_sha256") == archive.get("delivery_index_sha256"), f"{model_name}: delivery hash drift")

    llava = data["llava_generation"]
    _require(llava.get("protocol_version") == "llava-mitigation-t3-generation-audit-v1", "bad LLaVA T3 audit")
    _require(llava.get("all_operational_gates_passed") is True, "LLaVA T3 operational gate failed")
    _require(llava.get("physician_pack_authorized") is True, "LLaVA physician pack not authorized")
    _require(llava.get("clinical_efficacy_authorized") is False, "LLaVA efficacy authorized")
    _require(llava.get("reference_answers_used") is False, "LLaVA references used")
    _require(llava.get("clinical_labels_used") is False, "LLaVA clinical labels used")
    _require(llava.get("method_off_identity", {}).get("passed") is True, "LLaVA method-off failed")
    _require(llava["method_off_identity"].get("generated_token_exact_rate") == 1.0, "LLaVA method-off not exact")
    _require(len(llava.get("method_records", [])) == 10, "LLaVA T3 must contain ten methods")
    for row in llava["method_records"]:
        _require(row.get("rows") == 120 and row.get("eligible") is True, "bad LLaVA method record")
        _require(row.get("trace_complete") is True and row.get("cap_hits") == 0, "bad LLaVA trace/cap state")
        _require(row.get("reference_fields_absent") is True, "reference field in LLaVA generation")

    llava_post = data["llava_postprocess"]
    _require(llava_post.get("protocol_version") == "llava-mitigation-t3-physician-postprocess-v1", "bad LLaVA postprocess")
    _require(llava_post.get("generation_operationally_qualified") is True, "LLaVA generation unqualified")
    _require(llava_post.get("clinical_labels_present") is False, "LLaVA physician labels already present")
    _require(llava_post.get("clinical_efficacy_authorized") is False, "LLaVA clinical efficacy authorized")
    _require(llava_post.get("generation_audit_sha256") == sha256_file(llava_generation), "LLaVA generation hash drift")
    _validate_archive(data["llava_archive"], "llava")
    _require(llava_post.get("archive_verification_sha256") == sha256_file(llava_archive), "LLaVA archive hash drift")
    _require(llava_post.get("delivery_index_sha256") == data["llava_archive"].get("delivery_index_sha256"), "LLaVA delivery hash drift")

    for monitor_name in ("internal_huatuo_monitor", "internal_hulu_monitor", "llava_monitor"):
        _validate_monitor(data[monitor_name], monitor_name)

    handoff = data["system_handoff"]
    _require(
        handoff.get("schema_version") == "cecd-system-pih-native-eager-canary-handoff-v3-family-specific-query-shape",
        "bad System/PIH handoff",
    )
    _require(handoff.get("memory_correction", {}).get("scientific_tolerance_unchanged") is True, "System tolerance changed")
    source_sha = handoff.get("source_bindings", {}).get("runtime", {}).get("sha256")
    huatuo_canary = data["system_huatuo_canary"]
    hulu_canary = data["system_hulu_canary"]
    for model_name, canary in (("huatuo", huatuo_canary), ("hulu", hulu_canary)):
        _require(canary.get("schema_version") == "cecd-system-pih-native-eager-canary-artifact-v1", f"{model_name}: bad canary schema")
        _require(canary.get("integration_source", {}).get("sha256") == source_sha, f"{model_name}: source hash drift")
        _require(canary.get("selected_heads_consumed") is False, f"{model_name}: selected heads consumed")
    _require(huatuo_canary.get("status") == "native_eager_canary_passed", "Huatuo canary failed")
    _require(huatuo_canary.get("result", {}).get("passed") is True, "Huatuo result failed")
    _require(huatuo_canary["result"].get("max_absolute_error") == 0.0, "Huatuo canary not exact")
    _require(hulu_canary.get("status") == "failed", "Hulu canary must remain failed")
    _require(hulu_canary.get("result", {}).get("passed") is False, "Hulu canary improperly passed")
    _require(hulu_canary["result"].get("argmax_equal") is True, "Hulu diagnostic argmax changed")
    _require(hulu_canary["result"].get("max_absolute_error") == 3.0, "unexpected Hulu numerical result")
    job = data["system_job_state"]
    _require(job.get("status") == "failed" and job.get("exit_code") == 2, "System job did not fail closed")

    result: dict[str, Any] = {
        "version": "iclr-evaluation-progress-supplement-v4",
        "paper_ready": False,
        "submission_claim_authorized": False,
        "human_labels_synthesized": False,
        "clinical_efficacy_authorized": False,
        "evaluation_state": {
            "internal_t3_generation_qualified": True,
            "internal_physician_packs_ready": True,
            "llava_t3_generation_qualified": True,
            "llava_physician_pack_ready": True,
            "clinical_labels_present": False,
            "any_decoding_efficacy_authorized": False,
            "physician_review_stage": "waiting_for_independent_reviews",
            "rag_tracks_kept_separate": True,
            "rag_efficacy_authorized": False,
            "factmm_released_asset_role": rag.get("factmm_released_asset_role"),
            "system_pih": {
                "huatuo_native_eager_canary": "pass_exact",
                "hulu_native_eager_canary": "failed_numerical_equivalence",
                "hulu_argmax_equal_diagnostic_only": True,
                "cross_model_common_protocol_admissible": False,
                "tolerance_relaxed": False,
                "pih_efficacy_authorized": False,
            },
        },
        "requirements": {
            "R4": "operational_t3_complete_clinical_adjudication_pending",
            "R5": "engineering_complete_clinical_metrics_pending",
            "R6": "source_closed_operational_t3_partial_clinical_efficacy_pending",
            "R7": "physician_packs_ready_waiting_for_independent_returns",
        },
        "claim_boundary": (
            "T3 generation qualification, token-exact method-off identity, and verified blinded archives "
            "do not establish clinical efficacy. No decoding or RAG method may enter an efficacy table "
            "until independent physician returns and frozen adjudication complete. Hulu System remains "
            "inadmissible under the unchanged numerical-equivalence tolerance."
        ),
        "inputs": {name: _input_record(path) for name, path in paths.items()},
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "base_audit", "rag_audit", "baseline_audit", "internal_generation", "internal_form",
        "internal_postprocess", "internal_huatuo_archive", "internal_hulu_archive",
        "internal_huatuo_monitor", "internal_hulu_monitor", "llava_generation",
        "llava_postprocess", "llava_archive", "llava_monitor", "system_handoff",
        "system_huatuo_canary", "system_hulu_canary", "system_job_state",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = vars(args)
    output = kwargs.pop("output")
    result = build(**kwargs)
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
