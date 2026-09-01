"""Error-code decomposition for the frozen five-arm nested OOF stack.

This CPU-only analysis replays the already-defined five-fold crossfit from
``analyze_intervention_code_cxr_v2`` and explains where its corrections and
harms occur.  It does not tune, select, or refit a different stack.

Outputs:
* ``result.json``: aggregate strata and plain/RAG conditional effects;
* ``rows.jsonl``: one auditable record per shared, parseable binary sample.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    _load_expert,
    _metrics,
)
from anchor.corrected_sgta.analyze_intervention_code_cxr_v2 import (
    ARMS,
    fit_decode_indices,
    make_features,
)


MANIFEST = Path("corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json")
PARENT = Path("corrected_runs/intervention_code_cxr_v2/result.json")
OUT_DIR = Path("corrected_runs/intervention_code_error_decomposition_v1")
SUMMARY_OUT = OUT_DIR / "result.json"
ROWS_OUT = OUT_DIR / "rows.jsonl"
SEED = 20260810
BOOTSTRAPS = 2_000


FINDING_PATTERNS = [
    ("pneumothorax", r"pneumothorax"),
    ("pleural_effusion", r"pleural effusion|pleural fluid|effusion"),
    ("airspace_opacity", r"consolidat|infiltrat|pneumonia|airspace|opacity"),
    ("mass_or_nodule", r"mass|nodule|lesion|tumou?r"),
    ("edema_or_vasculature", r"edema|oedema|vascular|congestion"),
    ("device", r"tube|line|catheter|pacemaker|device|port"),
    ("cardiac", r"heart|cardiac|cardiomediastinal"),
    ("mediastinum", r"mediast"),
    ("bone", r"bony|osseous|fracture|spine|rib|skeletal"),
    ("diaphragm_or_free_air", r"diaphragm|intraperitoneal air|free air"),
    ("lung_general", r"lung|pulmonary|pleural"),
]

TEMPORAL = re.compile(r"\b(compared|interval|unchanged|new|improv|worsen|increas|decreas|previous|prior)\b", re.I)
CLINICAL = re.compile(r"\b(history|symptom|diagnos|cause|etiolog|treatment|recommend|prognos|management)\b", re.I)
BROAD = re.compile(r"\b(any abnormalit|acute (medical )?condition|condition of|pathology)\b", re.I)
NORMALITY = re.compile(r"\b(normal|clear|intact|within normal|unremarkable|no acute)\b", re.I)
ABNORMALITY = re.compile(r"\b(abnormal|acute|patholog|irregular)\b", re.I)
PRESENCE = re.compile(r"\b(present|presence|evidence|signs?|show|seen|visible|observ|identify|indicate)\b", re.I)
ATTRIBUTE = re.compile(r"\b(size|contour|enlarg|widen|level|appearance|structure)\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_of(cluster: str) -> int:
    return int(hashlib.sha256(str(cluster).encode()).hexdigest()[:8], 16) % 5


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["question_id"])] = row
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def answer_form(text: str) -> str:
    normalized = normalize_text(text)
    stripped = re.sub(r"[^a-z]+", " ", normalized).strip()
    if stripped in {"yes", "no"}:
        return "exact_binary_label"
    if re.match(r"^(yes|no)\b", stripped):
        return "label_plus_explanation"
    if re.search(r"\b(uncertain|possibly|perhaps|may|might|cannot|unable|unclear)\b", stripped):
        return "hedged_or_uncertain"
    return "other_parseable_binary"


def length_bucket(tokens: float) -> str:
    value = int(round(tokens))
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 8:
        return "3_8"
    return "9_plus"


def text_relation(plain: dict[str, Any], rag: dict[str, Any]) -> str:
    if plain["decision"] != rag["decision"]:
        return f"decision_{plain['decision']}_to_{rag['decision']}"
    if normalize_text(plain["text"]) == normalize_text(rag["text"]):
        return "same_decision_exact_text"
    if rag["tokens"] > plain["tokens"]:
        return "same_decision_rag_more_verbose"
    if rag["tokens"] < plain["tokens"]:
        return "same_decision_rag_less_verbose"
    return "same_decision_style_change"


def question_features(text: str) -> dict[str, str]:
    lowered = normalize_text(text)
    family = "other"
    for name, pattern in FINDING_PATTERNS:
        if re.search(pattern, lowered, re.I):
            family = name
            break
    if "xxxx" in lowered:
        observability = "source_placeholder"
    elif TEMPORAL.search(lowered):
        observability = "temporal_or_comparative"
    elif CLINICAL.search(lowered):
        observability = "clinical_or_causal_inference"
    elif BROAD.search(lowered):
        observability = "broad_global_judgment"
    elif family in {
        "pneumothorax", "pleural_effusion", "airspace_opacity",
        "mass_or_nodule", "edema_or_vasculature", "device",
    }:
        observability = "direct_named_finding"
    elif family != "other":
        observability = "direct_structure_attribute"
    else:
        observability = "other_visual_question"
    if NORMALITY.search(lowered):
        framing = "normality_or_clearance"
    elif ABNORMALITY.search(lowered):
        framing = "abnormality_or_acute"
    elif PRESENCE.search(lowered):
        framing = "presence_or_evidence"
    elif ATTRIBUTE.search(lowered):
        framing = "attribute_or_measurement"
    else:
        framing = "other_framing"
    first = re.match(r"\s*([a-z]+)", lowered)
    stem = first.group(1) if first else "other"
    words = re.findall(r"[a-z0-9]+", lowered)
    if len(words) <= 8:
        question_length = "short_le_8"
    elif len(words) <= 14:
        question_length = "medium_9_14"
    else:
        question_length = "long_15_plus"
    return {
        "finding_family": family,
        "observability_text_class": observability,
        "question_framing": framing,
        "question_stem": stem,
        "question_length_bucket": question_length,
    }


def outcome_label(baseline_correct: bool, stack_correct: bool) -> str:
    if baseline_correct and stack_correct:
        return "stable_correct"
    if (not baseline_correct) and stack_correct:
        return "rescue"
    if baseline_correct and (not stack_correct):
        return "harm"
    return "stable_error"


def error_transition(target: int, baseline: int, stack: int) -> str:
    if baseline == target and stack == target:
        return "stable_correct"
    if baseline != target and stack != target:
        return "stable_error"
    if baseline != target and stack == target:
        return "baseline_FP_to_correct" if target == 0 else "baseline_FN_to_correct"
    return "correct_to_stack_FP" if target == 0 else "correct_to_stack_FN"


def mechanism_subtype(row: dict[str, Any]) -> str:
    outcome = row["stack_outcome"]
    if outcome not in {"rescue", "harm"}:
        return outcome
    target = row["target"]
    correct_arms = sum(output["decision"] == target for output in row["outputs"].values())
    rag_transitions = []
    for model in ("huatuo", "hulu"):
        plain = row["outputs"][f"{model}_plain"]["decision"]
        rag = row["outputs"][f"{model}_rag"]["decision"]
        if plain != rag:
            rag_transitions.append("rescue" if rag == target else "harm")
    if outcome == "rescue":
        if correct_arms >= 3:
            return "majority_supported_rescue"
        if "rescue" in rag_transitions:
            return "minority_rag_direction_rescue"
        return "minority_confidence_override_rescue"
    if correct_arms <= 2:
        return "majority_supported_harm"
    if "harm" in rag_transitions:
        return "minority_rag_direction_harm"
    return "minority_confidence_override_harm"


def aggregate_dimension(
    rows: list[dict[str, Any]],
    field: str,
    getter: Callable[[dict[str, Any]], Any] | None = None,
    min_n: int = 5,
) -> list[dict[str, Any]]:
    getter = getter or (lambda row: row.get(field))
    total_rescues = sum(row["stack_outcome"] == "rescue" for row in rows)
    total_harms = sum(row["stack_outcome"] == "harm" for row in rows)
    values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(getter(row))
        values.setdefault(value, []).append(row)
    output = []
    for value, local in values.items():
        if len(local) < min_n:
            continue
        baseline_errors = sum(not row["baseline_correct"] for row in local)
        baseline_correct = len(local) - baseline_errors
        stack_errors = sum(not row["stack_correct"] for row in local)
        rescues = sum(row["stack_outcome"] == "rescue" for row in local)
        harms = sum(row["stack_outcome"] == "harm" for row in local)
        share_all = len(local) / len(rows)
        rescue_share = rescues / total_rescues if total_rescues else 0.0
        harm_share = harms / total_harms if total_harms else 0.0
        output.append({
            "value": value,
            "n": len(local),
            "share_all": share_all,
            "baseline_error_rate": baseline_errors / len(local),
            "stack_error_rate": stack_errors / len(local),
            "rescues": rescues,
            "harms": harms,
            "net_rescues_minus_harms": rescues - harms,
            "rescue_rate_given_baseline_error": rescues / baseline_errors if baseline_errors else None,
            "harm_rate_given_baseline_correct": harms / baseline_correct if baseline_correct else None,
            "share_of_all_rescues": rescue_share,
            "share_of_all_harms": harm_share,
            "rescue_concentration_ratio": rescue_share / share_all if share_all else None,
            "harm_concentration_ratio": harm_share / share_all if share_all else None,
        })
    return sorted(output, key=lambda item: (-item["n"], item["value"]))


def cluster_bootstrap_error_delta(
    rows: list[dict[str, Any]], model: str, seed: int
) -> dict[str, Any]:
    if not rows:
        return {"delta": None, "ci95": [None, None], "clusters": 0}
    clusters = sorted(set(str(row["cluster"]) for row in rows))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    delta_sum = np.zeros(len(clusters), dtype=float)
    counts = np.zeros(len(clusters), dtype=float)
    for row in rows:
        index = cluster_index[str(row["cluster"])]
        plain_error = row["outputs"][f"{model}_plain"]["decision"] != row["target"]
        rag_error = row["outputs"][f"{model}_rag"]["decision"] != row["target"]
        delta_sum[index] += float(rag_error) - float(plain_error)
        counts[index] += 1.0
    observed = float(delta_sum.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        draws.append(float(delta_sum[selected].sum() / counts[selected].sum()))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_rag_minus_plain_error_rate": observed,
        "ci95": [float(lo), float(hi)],
        "clusters": len(clusters),
        "replicates": BOOTSTRAPS,
    }


def plain_rag_effect(
    rows: list[dict[str, Any]], model: str, seed: int, bootstrap: bool = True
) -> dict[str, Any]:
    plain_key, rag_key = f"{model}_plain", f"{model}_rag"
    plain_errors = sum(row["outputs"][plain_key]["decision"] != row["target"] for row in rows)
    rag_errors = sum(row["outputs"][rag_key]["decision"] != row["target"] for row in rows)
    disagreement = [
        row for row in rows
        if row["outputs"][plain_key]["decision"] != row["outputs"][rag_key]["decision"]
    ]
    rescues = sum(row["outputs"][rag_key]["decision"] == row["target"] for row in disagreement)
    harms = len(disagreement) - rescues
    return {
        "n": len(rows),
        "plain_error_rate": plain_errors / len(rows) if rows else None,
        "rag_error_rate": rag_errors / len(rows) if rows else None,
        "disagreement_n": len(disagreement),
        "disagreement_rate": len(disagreement) / len(rows) if rows else None,
        "rag_rescues_within_disagreements": rescues,
        "rag_harms_within_disagreements": harms,
        "rag_rescue_fraction_within_disagreements": rescues / len(disagreement) if disagreement else None,
        "cluster_bootstrap": (
            cluster_bootstrap_error_delta(rows, model, seed) if bootstrap else None
        ),
    }


def conditional_plain_rag(rows: list[dict[str, Any]], model: str, seed: int) -> dict[str, Any]:
    output = {
        "all": plain_rag_effect(rows, model, seed),
        "ground_truth_no_FP_axis": plain_rag_effect(
            [row for row in rows if row["target"] == 0], model, seed + 1
        ),
        "ground_truth_yes_FN_axis": plain_rag_effect(
            [row for row in rows if row["target"] == 1], model, seed + 2
        ),
    }
    for dimension in ("observability_text_class", "finding_family", "question_framing"):
        values = sorted(set(row[dimension] for row in rows))
        output[f"by_{dimension}"] = {
            value: plain_rag_effect(
                [row for row in rows if row[dimension] == value],
                model,
                seed + 10 + index,
                bootstrap=False,
            )
            for index, value in enumerate(values)
            if sum(row[dimension] == value for row in rows) >= 20
        }
    return output


def main() -> None:
    parent = json.loads(PARENT.read_text())
    expected = parent["cohorts"]["full_five_arm_code"]["five_fold_crossfit"]
    names = list(ARMS)
    loaded = {name: _load_expert(path) for name, path in ARMS.items()}
    answers = {name: load_jsonl(path / "answers.jsonl") for name, path in ARMS.items()}
    manifest_rows = json.loads(MANIFEST.read_text())
    manifest = {str(row["qid"]): row for row in manifest_rows}
    qids = sorted(set.intersection(*(set(loaded[name]) for name in names)))
    pred = np.asarray([[loaded[name][qid]["pred"] for name in names] for qid in qids])
    nll = np.asarray([[loaded[name][qid]["nll"] for name in names] for qid in qids])
    tokens = np.asarray([[loaded[name][qid]["tokens"] for name in names] for qid in qids])
    target = np.asarray([loaded[names[0]][qid]["target"] for qid in qids])
    clusters = np.asarray([loaded[names[0]][qid]["cluster"] for qid in qids])
    folds = np.asarray([fold_of(cluster) for cluster in clusters])
    stack_prediction = np.zeros(len(qids), dtype=int)
    baseline_prediction = np.zeros(len(qids), dtype=int)
    baseline_arm = np.empty(len(qids), dtype=object)
    fold_thresholds = {}
    for heldout in range(5):
        test = np.flatnonzero(folds == heldout)
        validation_fold = (heldout + 1) % 5
        validation = np.flatnonzero(folds == validation_fold)
        train = np.flatnonzero((folds != heldout) & (folds != validation_fold))
        decoded, threshold = fit_decode_indices(
            make_features(pred, nll, tokens), target, train, validation
        )
        stack_prediction[test] = decoded[test]
        validation_scores = [
            _metrics(target[validation], pred[validation, column])["balanced_accuracy"]
            for column in range(pred.shape[1])
        ]
        selected = int(np.argmax(validation_scores))
        baseline_prediction[test] = pred[test, selected]
        baseline_arm[test] = names[selected]
        fold_thresholds[str(heldout)] = {
            "validation_fold": validation_fold,
            "threshold": threshold,
            "selected_single": names[selected],
        }

    observed_baseline = _metrics(target, baseline_prediction)
    observed_stack = _metrics(target, stack_prediction)
    if abs(observed_baseline["balanced_accuracy"] - expected["baseline"]["balanced_accuracy"]) > 1e-12:
        raise RuntimeError("nested OOF baseline replay mismatch")
    if abs(observed_stack["balanced_accuracy"] - expected["paired_code"]["balanced_accuracy"]) > 1e-12:
        raise RuntimeError("nested OOF stack replay mismatch")

    rows = []
    for index, qid in enumerate(qids):
        question = manifest[qid]
        q_features = question_features(str(question["source_question"]))
        outputs = {}
        for column, name in enumerate(names):
            answer = answers[name][qid]
            text = str(answer.get("text", ""))
            outputs[name] = {
                "decision": int(pred[index, column]),
                "text": text,
                "answer_form": answer_form(text),
                "nll": float(nll[index, column]),
                "tokens": int(round(tokens[index, column])),
                "length_bucket": length_bucket(tokens[index, column]),
            }
        base_correct = bool(baseline_prediction[index] == target[index])
        stack_correct = bool(stack_prediction[index] == target[index])
        row = {
            "qid": qid,
            "cluster": str(clusters[index]),
            "fold": int(folds[index]),
            "source_question": str(question["source_question"]),
            "image": str(question["img_name"]),
            "target": int(target[index]),
            "target_label": "yes" if target[index] else "no",
            "baseline_arm": str(baseline_arm[index]),
            "baseline_prediction": int(baseline_prediction[index]),
            "stack_prediction": int(stack_prediction[index]),
            "baseline_correct": base_correct,
            "stack_correct": stack_correct,
            "stack_outcome": outcome_label(base_correct, stack_correct),
            "error_transition": error_transition(
                int(target[index]), int(baseline_prediction[index]), int(stack_prediction[index])
            ),
            "response_code": "".join(str(int(value)) for value in pred[index]),
            "positive_arm_count": int(np.sum(pred[index])),
            **q_features,
            "outputs": outputs,
            "huatuo_plain_rag_relation": text_relation(
                outputs["huatuo_plain"], outputs["huatuo_rag"]
            ),
            "hulu_plain_rag_relation": text_relation(
                outputs["hulu_plain"], outputs["hulu_rag"]
            ),
            "cross_model_plain_relation": (
                "agree" if outputs["huatuo_plain"]["decision"] == outputs["hulu_plain"]["decision"]
                else "disagree"
            ),
            "any_plain_rag_decision_disagreement": bool(
                outputs["huatuo_plain"]["decision"] != outputs["huatuo_rag"]["decision"]
                or outputs["hulu_plain"]["decision"] != outputs["hulu_rag"]["decision"]
            ),
            "any_verbose_output": bool(any(
                output["answer_form"] == "label_plus_explanation"
                for output in outputs.values()
            )),
        }
        row["mechanism_subtype"] = mechanism_subtype(row)
        rows.append(row)

    dimensions: dict[str, Callable[[dict[str, Any]], Any] | None] = {
        "finding_family": None,
        "observability_text_class": None,
        "question_framing": None,
        "question_stem": None,
        "question_length_bucket": None,
        "target_label": None,
        "baseline_arm": None,
        "response_code": None,
        "positive_arm_count": None,
        "error_transition": None,
        "mechanism_subtype": None,
        "cross_model_plain_relation": None,
        "any_plain_rag_decision_disagreement": None,
        "any_verbose_output": None,
        "huatuo_plain_rag_relation": None,
        "hulu_plain_rag_relation": None,
        "huatuo_plain_answer_form": lambda row: row["outputs"]["huatuo_plain"]["answer_form"],
        "huatuo_rag_answer_form": lambda row: row["outputs"]["huatuo_rag"]["answer_form"],
        "hulu_plain_answer_form": lambda row: row["outputs"]["hulu_plain"]["answer_form"],
        "hulu_rag_answer_form": lambda row: row["outputs"]["hulu_rag"]["answer_form"],
    }
    decompositions = {
        name: aggregate_dimension(rows, name, getter)
        for name, getter in dimensions.items()
    }
    outcome_counts = Counter(row["stack_outcome"] for row in rows)
    def stratum(dimension: str, value: str) -> dict[str, Any]:
        return next(
            item for item in decompositions[dimension]
            if item["value"] == value
        )

    disagreement = stratum("any_plain_rag_decision_disagreement", "True")
    no_disagreement = stratum("any_plain_rag_decision_disagreement", "False")
    count_two = stratum("positive_arm_count", "2")
    count_three = stratum("positive_arm_count", "3")
    boundary_rescues = count_two["rescues"] + count_three["rescues"]
    boundary_harms = count_two["harms"] + count_three["harms"]
    top_response_codes = sorted(
        decompositions["response_code"],
        key=lambda item: (-(item["rescues"] + item["harms"]), -abs(item["net_rescues_minus_harms"])),
    )[:8]
    hulu_effect = conditional_plain_rag(rows, "hulu", SEED + 200)
    huatuo_effect = conditional_plain_rag(rows, "huatuo", SEED + 100)
    result = {
        "status": "cached_cpu_nested_oof_error_code_decomposition",
        "version": "intervention-code-error-decomposition-v1",
        "dataset": "MedHEval CXR-VisHal binary five-arm intersection",
        "n": len(rows),
        "arm_order_for_response_code": names,
        "inputs": {
            str(MANIFEST): sha256_file(MANIFEST),
            str(PARENT): sha256_file(PARENT),
            **{
                str(path / "answers.jsonl"): sha256_file(path / "answers.jsonl")
                for path in ARMS.values()
            },
        },
        "replay": {
            "folds": 5,
            "fold_thresholds_and_selected_singles": fold_thresholds,
            "baseline": observed_baseline,
            "paired_code": observed_stack,
            "matches_parent_artifact_exactly": True,
        },
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcome_rates": {
            key: int(value) / len(rows) for key, value in sorted(outcome_counts.items())
        },
        "decompositions": decompositions,
        "plain_rag_conditional_effects": {
            "huatuo": huatuo_effect,
            "hulu": hulu_effect,
        },
        "mechanism_candidates": {
            "intervention_disagreement_boundary": {
                "n": disagreement["n"],
                "share_all": disagreement["share_all"],
                "rescues": disagreement["rescues"],
                "share_of_all_rescues": disagreement["share_of_all_rescues"],
                "harms": disagreement["harms"],
                "share_of_all_harms": disagreement["share_of_all_harms"],
                "net_rescues_minus_harms": disagreement["net_rescues_minus_harms"],
                "no_disagreement_net_rescues_minus_harms": no_disagreement[
                    "net_rescues_minus_harms"
                ],
                "falsifiable_subtype": (
                    "Same-case intervention disagreement is a high-risk decision boundary. "
                    "A future source-only mechanism must predict rescue versus harm within this "
                    "boundary, not merely detect disagreement."
                ),
            },
            "two_or_three_positive_arm_boundary": {
                "n": count_two["n"] + count_three["n"],
                "rescues": boundary_rescues,
                "share_of_all_rescues": boundary_rescues / outcome_counts["rescue"],
                "harms": boundary_harms,
                "share_of_all_harms": boundary_harms / outcome_counts["harm"],
                "top_response_codes_by_changed_cases": top_response_codes,
                "falsifiable_subtype": (
                    "Corrections and harms should be mediated by structured split codes, "
                    "especially concordant plain-to-RAG transitions across model families, rather "
                    "than by subset size or unanimous votes."
                ),
            },
            "model_specific_rag_error_polarity": {
                "huatuo_FP_axis": huatuo_effect["ground_truth_no_FP_axis"],
                "huatuo_FN_axis": huatuo_effect["ground_truth_yes_FN_axis"],
                "hulu_FP_axis": hulu_effect["ground_truth_no_FP_axis"],
                "hulu_FN_axis": hulu_effect["ground_truth_yes_FN_axis"],
                "falsifiable_subtype": (
                    "Hulu RAG exhibits an absent-finding false-positive inflation phenotype, "
                    "whereas Huatuo RAG is closer to balanced/slightly beneficial. A retrieval "
                    "content or language-prior intervention should selectively mediate the Hulu "
                    "FP axis if this is a genuine mechanism."
                ),
            },
            "question_semantics_are_secondary": {
                "observability_strata": decompositions["observability_text_class"],
                "finding_strata": decompositions["finding_family"],
                "falsifiable_subtype": (
                    "No text-derived observability or finding family currently isolates the stack "
                    "effect as sharply as the response code. Temporal and verbose strata are small "
                    "risk markers, not established mechanism classes."
                ),
            },
        },
        "examples": {
            outcome: [
                {
                    "qid": row["qid"],
                    "question": row["source_question"],
                    "target": row["target_label"],
                    "baseline_arm": row["baseline_arm"],
                    "response_code": row["response_code"],
                    "mechanism_subtype": row["mechanism_subtype"],
                }
                for row in rows if row["stack_outcome"] == outcome
            ][:30]
            for outcome in ("rescue", "harm")
        },
        "interpretation_boundary": {
            "analysis_is_descriptive_not_stack_tuning": True,
            "observability_is_text_rule_not_clinical_annotation": True,
            "question_population": (
                "Only the 3,390 shared parseable binary rows used by the existing five-arm "
                "nested OOF analysis; benchmark multi-choice rows are excluded."
            ),
            "mechanism_claim": (
                "Stratum enrichment nominates mechanism subtypes for confirmation; it does not "
                "establish that question wording or RAG causally creates the stack effect."
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ROWS_OUT.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result["rows_artifact"] = {
        "path": str(ROWS_OUT),
        "sha256": sha256_file(ROWS_OUT),
        "n": len(rows),
    }
    SUMMARY_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "n": result["n"],
        "outcome_counts": result["outcome_counts"],
        "summary": str(SUMMARY_OUT),
        "rows": result["rows_artifact"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
