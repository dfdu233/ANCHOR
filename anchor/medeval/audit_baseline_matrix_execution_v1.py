#!/usr/bin/env python3
"""Fail-closed execution ledger for the frozen paper baseline matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file

VERSION = "baseline-matrix-execution-audit-v1"
ROOT = Path(__file__).resolve().parents[2]


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(errors="replace") as handle:
        return sum(bool(line.strip()) for line in handle)


def llava_chunks(base: Path, dataset: dict[str, Any], method: str) -> tuple[int, list[str]]:
    source = "mmedrag" if dataset["task"] == "report_generation" else ("vqa_rad" if dataset["id"] == "vqa_rad_official_oe" else "medheval")
    task = {"mixed_ce": "close_vqa", "open_vqa": "open_vqa", "report_generation": "report_generation"}[dataset["task"]]
    names = {"OPERA": "opera", "AvisC": "avisc"}
    folder = base / "llava_methods" / source / dataset["id"] / task / names.get(method, method)
    files = sorted(folder.glob("chunk_*.answers.jsonl"))
    return sum(count_jsonl(path) for path in files), [str(path) for path in files]


def native_path(base: Path, model: str, dataset: dict[str, Any], method: str) -> Path:
    family = "native_ce" if dataset["task"] == "mixed_ce" else "native"
    return base / family / model / dataset["id"] / method / "answers.jsonl"


def common_artifacts(base: Path, model: str, dataset: dict[str, Any], method: str) -> tuple[int, list[str]]:
    if method in {"greedy", "beam"}:
        path = native_path(base, model, dataset, method)
        return count_jsonl(path), [str(path)]
    if method == "shared_medical_rag":
        family = "shared_rag_report_generation" if dataset["task"] == "report_generation" else "shared_rag_generation"
        path = base / family / model / dataset["id"] / "rag" / "answers.jsonl"
        return count_jsonl(path), [str(path)]
    if model == "llava" and method in {"VCD", "DoLa", "OPERA", "PAI", "AvisC", "VISTA"}:
        return llava_chunks(base, dataset, method)
    if method in {"VCD", "DoLa"} and model in {"huatuo", "hulu", "qwen"}:
        path = base / "cross_model_methods" / model / method.lower() / dataset["id"] / "answers.jsonl"
        return count_jsonl(path), [str(path)]
    return 0, []


def applicability(model: str, method: str) -> tuple[str, str | None]:
    if method == "SECOND":
        return "N/A", "official root repository has no method-code license; T1/T2 cannot be paper-certified"
    if method in {"OPERA", "PAI", "AvisC", "VISTA"} and model != "llava":
        return "N/A", "official intervention is architecture-specific; no certified faithful port for this model"
    if method == "VCD" and model == "qwen":
        return "pending", "faithful Qwen visual-contrast port still under T1/T2 audit"
    return "applicable", None


def score_artifact(base: Path, model: str, dataset: dict[str, Any], method: str, artifacts: list[str]) -> Path:
    dataset_id = dataset["id"]
    if dataset["task"] == "report_generation":
        short = "iu_xray" if dataset_id == "iu_xray_report" else "mimic_cxr"
        if method in {"greedy", "beam"}:
            key = Path("native") / model / short / method
        elif method == "shared_medical_rag":
            key = Path("shared_rag") / model / short / "rag"
        elif model == "llava":
            key = Path("llava_methods") / short / ({"OPERA": "opera", "AvisC": "avisc"}.get(method, method))
        else:
            key = Path("cross") / model / short / method.lower()
        return base / "report_scores" / key / "summary.json"
    name = "evaluation_ce_v7.json" if dataset["task"] == "mixed_ce" else "evaluation_lexical_auxiliary.json"
    if method in {"VCD", "DoLa"} and model in {"huatuo", "hulu", "qwen"}:
        return base / "derived_scores" / model / method / dataset_id / name
    if len(artifacts) == 1:
        return Path(artifacts[0]).parent / name
    return base / "derived_scores" / model / method / dataset_id / name


def qualification_state(path: Path, expected: int, task: str) -> tuple[str, str | None]:
    if not path.is_file():
        return "missing", f"qualification artifact missing: {path}"
    try:
        payload = json.loads(path.read_text())
        summary = payload.get("summary", payload)
    except (OSError, json.JSONDecodeError):
        return "stale", f"unreadable qualification artifact: {path}"
    if summary.get("passed") is False:
        return "failed", (
            "frozen output-quality gate failed"
            f" (nonempty={summary.get('nonempty_rate')}, parse={summary.get('parse_rate')},"
            f" fragments={summary.get('function_word_only_rate')}, cap_hit={summary.get('cap_hit_rate')});"
            f" evidence: {path}"
        )
    required = {
        "passed": True,
        "expected_count": expected,
        "received_count": expected,
        "exact_qid_alignment": True,
    }
    for key, value in required.items():
        if summary.get(key) != value:
            return "stale", f"qualification field {key} is not bound to the frozen cell: {path}"
    manifest = Path(str(summary.get("manifest", "")))
    answers = [Path(str(value)) for value in summary.get("answers", [])]
    hashes = summary.get("answer_sha256", [])
    expected_protocol = (
        "ce-generation-qualification-v2-task-aware-structural"
        if task == "mixed_ce"
        else "oe-generation-qualification-v3-structural"
    )
    if (
        summary.get("protocol_version") != expected_protocol
        or not manifest.is_file()
        or summary.get("manifest_sha256") != sha256_file(manifest)
        or not answers
        or len(answers) != len(hashes)
        or any(not answer.is_file() or digest != sha256_file(answer) for answer, digest in zip(answers, hashes))
    ):
        return "stale", f"qualification provenance is absent or stale: {path}"
    return "passed", None


def score_binding_failure(path: Path, task: str) -> str | None:
    if not path.is_file():
        return f"score artifact missing: {path}"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return f"unreadable score artifact: {path}"
    evaluator_by_task = {
        "mixed_ce": ROOT / "anchor" / "corrected_sgta" / "evaluate_medheval_answers.py",
        "open_vqa": ROOT / "anchor" / "medeval" / "evaluate_oe_vqa.py",
        "report_generation": ROOT / "corrected_sgta" / "evaluate_oe_reports.py",
    }
    evaluator = evaluator_by_task.get(task)
    if evaluator is None or not evaluator.is_file():
        return f"current evaluator source is unavailable for task={task}: {evaluator}"
    current_evaluator_sha = sha256_file(evaluator)
    if task == "mixed_ce":
        recorded_evaluator = Path(str(payload.get("evaluator_source", "")))
        if (
            not recorded_evaluator.is_file()
            or recorded_evaluator.resolve() != evaluator.resolve()
            or payload.get("evaluator_source_sha256") != current_evaluator_sha
        ):
            return f"CE score is not bound to the current evaluator source: {path}"
        answer = Path(str(payload.get("answers", "")))
        if not answer.is_file() or payload.get("answers_sha256") != sha256_file(answer):
            return f"CE score is not hash-bound to its current answers: {path}"
        questions = Path(str(payload.get("questions", "")))
        if not questions.is_file() or payload.get("questions_sha256") != sha256_file(questions):
            return f"CE score is not hash-bound to its current questions: {path}"
    elif task == "open_vqa":
        recorded_evaluator = Path(str(payload.get("evaluator_source", "")))
        if (
            not recorded_evaluator.is_file()
            or recorded_evaluator.resolve() != evaluator.resolve()
            or payload.get("evaluator_source_sha256") != current_evaluator_sha
        ):
            return f"OE score is not bound to the current evaluator source: {path}"
        answers = [Path(str(value)) for value in payload.get("answers", [])]
        hashes = payload.get("answer_sha256", [])
        manifest = Path(str(payload.get("manifest", "")))
        if (
            not manifest.is_file()
            or payload.get("manifest_sha256") != sha256_file(manifest)
            or not answers
            or len(answers) != len(hashes)
            or any(not answer.is_file() or digest != sha256_file(answer) for answer, digest in zip(answers, hashes))
        ):
            return f"OE score provenance is absent or stale: {path}"
    else:
        if payload.get("config", {}).get("code_sha256") != current_evaluator_sha:
            return f"report score is not bound to the current evaluator source: {path}"
        inputs = payload.get("config", {}).get("inputs", [])
        if not inputs:
            return f"report score has no bound input: {path}"
        for item in inputs:
            source = Path(str(item.get("path", "")))
            if not source.is_file() or item.get("sha256") != sha256_file(source):
                return f"report score input provenance is stale: {path}"
    return None


def comparison_binding_failure(path: Path) -> str | None:
    if not path.is_file():
        return f"comparison missing: {path}"
    try:
        provenance = json.loads(path.read_text()).get("provenance", {})
    except (OSError, json.JSONDecodeError):
        return f"comparison unreadable: {path}"
    for path_key, hash_key in (
        ("manifest", "manifest_sha256"),
        ("baseline_answers", "baseline_answers_sha256"),
        ("candidate_answers", "candidate_answers_sha256"),
    ):
        source = Path(str(provenance.get(path_key, "")))
        if not source.is_file() or provenance.get(hash_key) != sha256_file(source):
            return f"comparison provenance is stale: {path}"
    return None


def score_gate_failure(path: Path, task: str) -> str | None:
    if task != "mixed_ce" or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return f"unreadable CE score artifact: {path}"
    parse_rate = payload.get("primary_multiclass", {}).get("parse_rate", payload.get("parse_rate"))
    if parse_rate is not None and float(parse_rate) < 0.90:
        return f"frozen CE parse-rate gate failed ({float(parse_rate):.4f} < 0.90); evidence: {path}"
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/unified_eval/baseline_matrix_v1.json")
    p.add_argument("--run-root", type=Path, default=ROOT / "corrected_runs/paper_baselines_v1/full_matrix_v1")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    datasets = config["datasets"]
    models = [row["id"] for row in config["models"] if row["track"] in {"main", "main_control"}]
    cells = []
    for model in models:
        for method in config["common_methods"]:
            apply, reason = applicability(model, method)
            if model == "qwen" and method == "VCD":
                gate = args.run_root / "cross_model_methods/qwen/vcd/t1_t2_audit.json"
                if gate.is_file():
                    if json.loads(gate.read_text()).get("passed"):
                        apply, reason = "applicable", None
                    else:
                        apply, reason = "N/A", f"faithful Qwen VCD T1/T2 gate failed; evidence: {gate}"
            if model == "qwen" and method == "DoLa":
                gate = args.run_root / "gates/qwen_dola/summary.json"
                if gate.is_file() and not json.loads(gate.read_text()).get("passed"):
                    apply, reason = "N/A", f"Qwen DoLa T1/T2 gate failed; evidence: {gate}"
            if model == "llava" and method in {"VCD", "DoLa", "OPERA", "PAI", "AvisC", "VISTA"}:
                gate_method = {"OPERA": "opera", "AvisC": "avisc"}.get(method, method)
                gate = args.run_root / "gates/llava_methods" / gate_method / "t1_t2_audit.json"
                if gate.is_file() and not json.loads(gate.read_text()).get("passed"):
                    apply, reason = "N/A", f"current-source LLaVA-Med {method} T1/T2 gate failed; evidence: {gate}"
            if model in {"huatuo", "hulu"} and method in {"VCD", "DoLa"}:
                gate = args.run_root.parent / "cross_model_gates_v2" / model / method.lower() / "t1_t2_audit.json"
                if gate.is_file() and not json.loads(gate.read_text()).get("passed"):
                    apply, reason = "N/A", f"cross-architecture {model} {method} T1/T2 gate failed; evidence: {gate}"
            for dataset in datasets:
                cell_reason = reason
                expected = int(dataset["rows"])
                count, artifacts = common_artifacts(args.run_root, model, dataset, method)
                score = score_artifact(args.run_root, model, dataset, method, artifacts)
                qualification = score.parent / "qualification.json"
                qualification_status, qualification_reason = qualification_state(qualification, expected, dataset["task"])
                score_quality_reason = score_gate_failure(score, dataset["task"])
                binding_reason = score_binding_failure(score, dataset["task"]) if score.is_file() else None
                if apply == "N/A":
                    status = "N/A"
                elif count == expected and (qualification_status == "failed" or score_quality_reason):
                    status, cell_reason = "N/A", qualification_reason or score_quality_reason
                elif count == expected and qualification_status == "passed" and score.is_file() and not binding_reason:
                    status = "completed"
                elif count == expected:
                    status, cell_reason = "generated_unscored", qualification_reason or binding_reason
                elif count:
                    status = "running_or_partial"
                else:
                    status = "pending"
                cells.append({"track": "training_free", "model": model, "method": method, "dataset": dataset["id"], "expected": expected, "actual": count, "status": status, "reason": cell_reason, "artifacts": artifacts, "qualification_artifact": str(qualification), "score_artifact": str(score)})

    variants = {"base": "base", "HA-DPO": "ha-dpo", "OPA-DPO": "opa-dpo", "DA-DPO": "da-dpo", "SENTINEL": "sentinel", "Less-is-More": "less-is-more", "FactMM-generator": "factmm-rag-generator", "VHR": "vhr"}
    trained_t0_path = args.run_root.parent / "trained_llava_t0_v1.json"
    trained_t0 = json.loads(trained_t0_path.read_text()) if trained_t0_path.is_file() else None
    trained_t0_rows = {row["method"]: row for row in trained_t0.get("methods", [])} if trained_t0 else {}
    trained_gate_path = args.run_root.parent / "trained_llava_t2_v2/t2_audit.json"
    trained_gate = json.loads(trained_gate_path.read_text()) if trained_gate_path.is_file() else None
    vhr_gate_path = args.run_root / "trained_llava15/vhr_gates/t1_t2_audit.json"
    vhr_gate = json.loads(vhr_gate_path.read_text()) if vhr_gate_path.is_file() else None
    for method, variant in variants.items():
        for dataset in datasets:
            expected = int(dataset["rows"])
            path = args.run_root / "trained_llava15" / variant / dataset["id"] / "answers.jsonl"
            count = count_jsonl(path)
            if dataset["task"] == "report_generation":
                short = "iu_xray" if dataset["id"] == "iu_xray_report" else "mimic_cxr"
                score = args.run_root / "report_scores" / "trained" / variant / short / "summary.json"
            else:
                score = path.parent / ("evaluation_ce_v7.json" if dataset["task"] == "mixed_ce" else "evaluation_lexical_auxiliary.json")
            qualification = score.parent / "qualification.json"
            qualification_status, qualification_reason = qualification_state(qualification, expected, dataset["task"])
            score_quality_reason = score_gate_failure(score, dataset["task"])
            binding_reason = score_binding_failure(score, dataset["task"]) if score.is_file() else None
            gate_reason = None
            t0_row = trained_t0_rows.get(variant)
            if method != "VHR" and t0_row is not None and t0_row.get("status") == "N/A":
                gate_reason = f"trained-method T0 failed: {', '.join(t0_row.get('reasons', []))}; evidence: {trained_t0_path}"
            elif method == "VHR" and vhr_gate is not None and not vhr_gate.get("passed"):
                gate_reason = f"official VHR T1/T2 gate failed; evidence: {vhr_gate_path}"
            elif method != "VHR" and trained_gate is not None and variant not in trained_gate.get("passed_variants", []):
                gate_reason = f"official-entry 32-case token identity failed; evidence: {trained_gate_path}"
            if gate_reason:
                status = "N/A"
            elif count == expected and (qualification_status == "failed" or score_quality_reason):
                status = "N/A"
            else:
                status = "completed" if count == expected and qualification_status == "passed" and score.is_file() and not binding_reason else ("generated_unscored" if count == expected else ("running_or_partial" if count else "pending"))
            reason = gate_reason or qualification_reason or score_quality_reason or binding_reason or (None if method != "VHR" else "official-compatible implementation is still in T1/T2")
            cells.append({"track": "trained_llava15", "model": "llava15", "method": method, "dataset": dataset["id"], "expected": expected, "actual": count, "status": status, "reason": reason, "artifacts": [str(path)], "qualification_artifact": str(qualification), "score_artifact": str(score)})

    diagnostic = [{"method": name, "status": "N/A", "reason": reason} for name, reason in config["diagnostic_na"].items()]
    auxiliary_controls = []
    for model in models:
        for dataset in datasets:
            expected = int(dataset["rows"])
            if dataset["task"] == "report_generation":
                answers = args.run_root / "shared_rag_report_generation" / model / dataset["id"] / "no_context/answers.jsonl"
                short = "iu_xray" if dataset["id"] == "iu_xray_report" else "mimic_cxr"
                score = args.run_root / "report_scores/shared_rag" / model / short / "no_context/summary.json"
            else:
                answers = args.run_root / "shared_rag_generation" / model / dataset["id"] / "no_context/answers.jsonl"
                score = answers.parent / ("evaluation_ce_v7.json" if dataset["task"] == "mixed_ce" else "evaluation_lexical_auxiliary.json")
            count = count_jsonl(answers)
            qualification = score.parent / "qualification.json"
            qualification_status, qualification_reason = qualification_state(qualification, expected, dataset["task"])
            score_quality_reason = score_gate_failure(score, dataset["task"])
            binding_reason = score_binding_failure(score, dataset["task"]) if score.is_file() else None
            reason = qualification_reason or score_quality_reason or binding_reason
            if count == expected and (qualification_status == "failed" or score_quality_reason):
                status = "N/A"
            elif count == expected and qualification_status == "passed" and score.is_file() and not binding_reason:
                status = "completed"
            elif count == expected:
                status = "generated_unscored"
            elif count:
                status = "running_or_partial"
            else:
                status = "pending"
            auxiliary_controls.append({"control": "shared_rag_matched_no_context", "model": model, "dataset": dataset["id"], "expected": expected, "actual": count, "status": status, "reason": reason, "answers": str(answers), "qualification_artifact": str(qualification), "score_artifact": str(score)})
    causal_root = args.run_root / "rag_controls_n200/knowledge_mimic_ce"
    causal_manifest = causal_root / "manifest.json"
    causal_manifest_passed = False
    if causal_manifest.is_file():
        try:
            causal_manifest_passed = json.loads(causal_manifest.read_text()).get("passed") is True
        except (OSError, json.JSONDecodeError):
            causal_manifest_passed = False
    for model in models:
        for condition in ("shuffled_context", "image_swap"):
            answers = causal_root / "generation" / model / condition / "answers.jsonl"
            score = answers.parent / "evaluation_ce_v7.json"
            qualification = answers.parent / "qualification.json"
            count = count_jsonl(answers)
            qualification_status, qualification_reason = qualification_state(qualification, 200, "mixed_ce")
            score_quality_reason = score_gate_failure(score, "mixed_ce")
            binding_reason = score_binding_failure(score, "mixed_ce") if score.is_file() else None
            reason = qualification_reason or score_quality_reason or binding_reason
            if not causal_manifest_passed:
                status, reason = "pending", f"frozen n=200 causal-control manifest has not passed: {causal_manifest}"
            elif count == 200 and (qualification_status == "failed" or score_quality_reason):
                status = "N/A"
            elif count == 200 and qualification_status == "passed" and score.is_file() and not binding_reason:
                status = "completed"
            elif count == 200:
                status = "generated_unscored"
            elif count:
                status = "running_or_partial"
            else:
                status = "pending"
            auxiliary_controls.append({
                "control": f"shared_rag_{condition}_n200",
                "model": model,
                "dataset": "knowledge_mimic_ce",
                "expected": 200,
                "actual": count,
                "status": status,
                "reason": reason,
                "answers": str(answers),
                "qualification_artifact": str(qualification),
                "score_artifact": str(score),
                "control_manifest": str(causal_manifest),
            })
    for model in models:
        comparison_root = causal_root / "comparisons" / model
        comparison_paths = [
            comparison_root / "rag_vs_shuffled_context.json",
            comparison_root / "rag_vs_image_swap.json",
            comparison_root / "rag_vs_no_context.json",
        ]
        reasons = [reason for path in comparison_paths if (reason := comparison_binding_failure(path))]
        auxiliary_controls.append({
            "control": "shared_rag_causal_comparisons_n200",
            "model": model,
            "dataset": "knowledge_mimic_ce",
            "expected": 3,
            "actual": sum(path.is_file() for path in comparison_paths),
            "status": "completed" if not reasons else "pending",
            "reason": "; ".join(reasons) if reasons else None,
            "answers": None,
            "qualification_artifact": None,
            "score_artifact": str(comparison_root),
            "comparison_artifacts": [str(path) for path in comparison_paths],
        })
    counts = Counter(row["status"] for row in cells)
    control_counts = Counter(row["status"] for row in auxiliary_controls)
    output = {"version": VERSION, "config": str(args.config.resolve()), "cells": cells, "auxiliary_controls": auxiliary_controls, "diagnostic_na": diagnostic, "summary": {"cells": len(cells), **dict(sorted(counts.items())), "auxiliary_controls": {"cells": len(auxiliary_controls), **dict(sorted(control_counts.items()))}}, "complete": all(row["status"] in {"completed", "N/A"} for row in [*cells, *auxiliary_controls])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
