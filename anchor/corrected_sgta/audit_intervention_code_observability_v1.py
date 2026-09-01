"""Observability-stratified audit of cached CXR-VisHal intervention-code gains.

The lexical taxonomy is intentionally conservative and is not treated as
clinician ground truth.  It separates questions that appear answerable from one
current radiograph from questions requiring history, comparison, procedure,
etiology, or management knowledge. Unknown questions remain unknown.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

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


QUESTION_PATH = Path("corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json")
OUT_DIR = Path("corrected_runs/intervention_code_cxr_v2/observability_audit_v1")
CACHED_PARENT_RESULT = Path("corrected_runs/intervention_code_cxr_v2/result.json")

# Ordered precedence prevents a visible word such as "opacity" from turning a
# temporal or causal question into a directly observable one.
RULES = {
    "management": [
        r"\brecommend(?:ed|ation)?\b", r"\bfollow[- ]?up\b", r"\bnext step\b",
        r"\bmanagement\b", r"\btreat(?:ment|ed)?\b", r"\baction\b",
        r"\bshould (?:be|the patient)\b", r"\bneeds? to be adjust(?:ed|ment)\b",
    ],
    "history": [
        r"\bhistory\b", r"\bhistorical\b", r"\bpreviously diagnosed\b",
        r"\bsymptoms?\b", r"\bclinical history\b", r"\bpatient had\b",
        r"\bpast medical\b", r"\bknown (?:case|history|diagnosis)\b",
    ],
    "temporal_comparison": [
        r"\bprior (?:exam|examination|study|image|film)\b",
        r"\bprevious (?:exam|examination|study|image|film|images|exams)\b",
        r"\bcompared (?:with|to)\b", r"\bsince (?:the )?(?:prior|previous|last)\b",
        r"\binterval\b", r"\bnew\b", r"\bnewly\b",
        r"\b(?:has|have|had)\b.{0,80}\b(?:changed|increased|decreased|improved|worsened)\b",
        r"\b(?:increase|decrease) in\b",
        r"\bchanges?\b.{0,50}\b(?:compared|since|prior|previous)\b",
        r"\bunchanged\b", r"\bstable\b", r"\bremain(?:ed|s)? the same\b",
        r"\bimprov(?:e|ed|ement|ing)\b", r"\bworsen(?:ed|ing)?\b",
        r"\bprogress(?:ed|ive|ion)\b", r"\bresolv(?:e|ed|ing)\b",
        r"\breaccumulat(?:e|ed|ion)\b", r"\bremoved since\b",
        r"\bextubat(?:e|ed|ion)\b",
    ],
    "procedure": [
        r"\bprocedure\b", r"\bsurger(?:y|ies|ical)\b", r"\boperat(?:ed|ion)\b",
        r"\bundergone\b", r"\bstatus post\b", r"\bpost[- ]?operative\b",
        r"\bpost[- ]?surgical\b", r"\bcabg\b", r"\bbypass graft\b",
        r"\bmedian sternotomy\b", r"\bvats\b", r"\btransplant(?:ed|ation)?\b",
    ],
    "etiology_diagnosis": [
        r"\b(?:likely|possible|probable) cause\b", r"\bcaused by\b", r"\bdue to\b",
        r"\betiology\b", r"\bwhat (?:has )?caused\b", r"\bdiagnosis\b",
        r"\bdifferential\b", r"\bconcerning for\b", r"\bcompatible with\b",
        r"\bwhat condition (?:is|does|could|might)\b",
        r"\bwhat (?:does|could|might) .{0,60} (?:indicate|represent|suggest)\b",
        r"\bmore consistent with\b", r"\bmost consistent with\b",
        r"\bfindings? (?:be )?(?:indicative|suggestive) of\b",
        r"\bair[- ]fluid level .{0,30}\bindicat(?:e|es|ing)\b",
    ],
    "unobservable_other": [
        r"\bmodality\b", r"\bpatient['’]s age\b", r"\bupper arm pain\b",
        r"\bchest (?:is experiencing )?pain\b", r"\bhypoxia\b",
        r"\bcough\b", r"\bbloody sputum\b", r"\bclinical condition\b",
    ],
    "direct_visual": [
        # Findings and general radiographic state.
        r"\b(?:opacity|opacities|opacification)\b", r"\batelectasis\b",
        r"\bconsolidations?\b", r"\b(?:infiltrates?|infiltration)\b", r"\bpneumonia\b",
        r"\b(?:effusions?|pleural fluid)\b", r"\bpneumothora(?:x|ces)\b", r"\bhemothorax\b",
        r"\bpneumomediastinum\b", r"\bempyema\b",
        r"\bedema\b", r"\bvascular congestion\b", r"\bfluid overload\b",
        r"\b(?:cardiomegaly|cardiac enlargement)\b", r"\bheart (?:size|enlarged|enlargement)\b",
        r"\bcardiomediastinal (?:silhouette|contour|contours)\b",
        r"\bmediastin(?:um|al)\b", r"\bhilar\b", r"\blung volumes?\b",
        r"\b(?:hyperinflation|hyperinflated|hyperexpanded|well[- ]expanded|normally (?:inflated|expanded)|well aerated)\b",
        r"\b(?:emphysema|emphysematous)\b", r"\bscarring\b",
        r"\bfibrosis\b", r"\bgranuloma|granulomatous\b", r"\bnodule|nodules\b",
        r"\bmass|masses\b", r"\bcalcification|calcified\b",
        r"\bpleural thickening\b", r"\bblunting\b", r"\bcostophrenic\b",
        r"\b(?:fracture|deformity)\b", r"\b(?:osseous|bony)\b", r"\bdegenerative\b",
        r"\b(?:kyphosis|scoliosis|spondylosis|osteopenia)\b", r"\bvertebr(?:a|al|ae)\b",
        r"\bhiatal hernia\b", r"\beventration\b", r"\badenopathy\b",
        r"\bdiaphragm|hemidiaphragm\b", r"\bfree air\b", r"\bairspace disease\b",
        r"\bbronchiectasis\b", r"\bcavitary|cavity\b", r"\bcollapse\b",
        r"\bcontour|contours\b", r"\bsilhouette\b", r"\bdensity|densities\b",
        r"\bmarkings\b", r"\bpulmonary (?:vascularity|arter(?:y|ies))\b",
        r"\baort(?:a|ic)\b", r"\bforeign bod(?:y|ies)\b", r"\bacute findings?\b",
        r"\bsubdiaphragmatic air\b", r"\bgastric (?:air bubble|pull up)\b",
        r"\babnormalit(?:y|ies)\b", r"\bnormal\b",
        r"\bclear\b", r"\bvisible\b", r"\bseen\b", r"\bshow\b",
        r"\bevidence of\b", r"\bsigns? of\b", r"\bappearance\b",
        # Anatomy/location.
        r"\bright|left|bilateral\b", r"\bupper|lower|middle lobe\b",
        r"\bapex|apical|base|basilar\b", r"\bretrocardiac\b",
        r"\bperihilar|infrahilar\b", r"\bhemithorax\b", r"\btrachea\b",
        r"\bchest wall\b", r"\bclavicle|humerus|rib|spine\b",
        # Devices and their directly visible position.
        r"\b(?:catheter|central line|picc|port[- ]?a[- ]?cath)\b",
        r"\b(?:endotracheal|et tube|tracheostomy|enteric|feeding|nasogastric)\b",
        r"\b(?:chest tube|drainage tube|pigtail)\b", r"\b(?:intubated|intubation)\b",
        r"\b(?:pacemaker|defibrillator)\b",
        r"\bintraaortic balloon pump|iabp\b", r"\bdevice|hardware|wire|wires\b",
        r"\b(?:surgical )?clips?\b", r"\btip\b", r"\bposition(?:ed|ing)?\b",
        r"\bplacement\b", r"\banterior[- ]posterior|\bap view\b",
        r"\bapices fully included\b",
    ],
}

PRECEDENCE = [
    "management", "history", "temporal_comparison", "procedure",
    "etiology_diagnosis", "unobservable_other", "direct_visual",
]


def classify(question: str) -> tuple[str, str | None]:
    text = question.lower()
    for category in PRECEDENCE:
        for pattern in RULES[category]:
            if re.search(pattern, text):
                return category, pattern
    return "unknown", None


def split_fold(cluster: str) -> int:
    return int(hashlib.sha256(str(cluster).encode()).hexdigest()[:8], 16) % 5


def binary_parse_coverage(directory: Path) -> dict:
    evaluation = json.loads((directory / "evaluation_ce_v7.json").read_text())
    total = parsed = 0
    for row in evaluation["details"]:
        truth = row.get("ground_truth")
        prediction = row.get("prediction")
        truth = truth[0] if isinstance(truth, list) and len(truth) == 1 else truth
        prediction = prediction[0] if isinstance(prediction, list) and len(prediction) == 1 else prediction
        if str(truth).strip().lower() not in {"yes", "no"}:
            continue
        total += 1
        parsed += int(str(prediction).strip().lower() in {"yes", "no"})
    return {"binary_target_n": total, "parseable_n": parsed, "parseable_rate": parsed / total}


def metric_or_null(target: np.ndarray, pred: np.ndarray) -> dict:
    out = _metrics(target, pred)
    out["positive_n"] = int(np.sum(target == 1))
    out["negative_n"] = int(np.sum(target == 0))
    if out["positive_n"] == 0 or out["negative_n"] == 0:
        out["balanced_accuracy"] = None
    return out


def bootstrap(target, candidate, baseline, clusters, draws=5000) -> dict:
    rng = np.random.default_rng(20260810)
    unique = np.unique(clusters)
    # Aggregate once by cluster; repeatedly concatenating raw claim indices is
    # unnecessarily expensive and gives exactly the same resampling statistic.
    counts = []
    for cluster in unique:
        idx = np.flatnonzero(clusters == cluster)
        yy, cc, bb = target[idx], candidate[idx], baseline[idx]
        counts.append([
            len(idx), np.sum(yy == cc), np.sum(yy == bb),
            np.sum(yy == 1), np.sum((yy == 1) & (cc == 1)), np.sum((yy == 1) & (bb == 1)),
            np.sum(yy == 0), np.sum((yy == 0) & (cc == 0)), np.sum((yy == 0) & (bb == 0)),
        ])
    counts = np.asarray(counts, dtype=float)
    acc, bacc = [], []
    for _ in range(draws):
        sample = counts[rng.integers(0, len(unique), size=len(unique))].sum(axis=0)
        acc.append(float((sample[1] - sample[2]) / sample[0]))
        if sample[3] and sample[6]:
            cand_bacc = 0.5 * (sample[4] / sample[3] + sample[7] / sample[6])
            base_bacc = 0.5 * (sample[5] / sample[3] + sample[8] / sample[6])
            bacc.append(float(cand_bacc - base_bacc))
    return {
        "draws": draws,
        "accuracy_delta": float(np.mean(target == candidate) - np.mean(target == baseline)),
        "accuracy_delta_ci95": [float(x) for x in np.quantile(acc, [0.025, 0.975])],
        "balanced_accuracy_delta": (
            float(_metrics(target, candidate)["balanced_accuracy"] - _metrics(target, baseline)["balanced_accuracy"])
            if len(np.unique(target)) == 2 else None
        ),
        "balanced_accuracy_delta_ci95": (
            [float(x) for x in np.quantile(bacc, [0.025, 0.975])] if bacc else None
        ),
    }


def main() -> None:
    names = ["huatuo_plain", "huatuo_rag", "hulu_plain", "hulu_rag"]
    loaded = {name: _load_expert(ARMS[name]) for name in names}
    arm_coverage = {name: binary_parse_coverage(ARMS[name]) for name in names}
    qids = sorted(set.intersection(*(set(loaded[name]) for name in names)))
    pred = np.asarray([[loaded[name][qid]["pred"] for name in names] for qid in qids])
    nll = np.asarray([[loaded[name][qid]["nll"] for name in names] for qid in qids])
    tokens = np.asarray([[loaded[name][qid]["tokens"] for name in names] for qid in qids])
    target = np.asarray([loaded[names[0]][qid]["target"] for qid in qids])
    clusters = np.asarray([loaded[names[0]][qid]["cluster"] for qid in qids])
    questions = {str(row.get("id", row.get("qid"))): row for row in json.loads(QUESTION_PATH.read_text())}

    paired = np.zeros(len(qids), dtype=int)
    baseline = np.zeros(len(qids), dtype=int)
    selected_by_fold = []
    folds = np.asarray([split_fold(cluster) for cluster in clusters])
    features = make_features(pred, nll, tokens)
    for heldout in range(5):
        test = np.flatnonzero(folds == heldout)
        validation = np.flatnonzero(folds == ((heldout + 1) % 5))
        train = np.flatnonzero((folds != heldout) & (folds != ((heldout + 1) % 5)))
        decoded, threshold = fit_decode_indices(features, target, train, validation)
        paired[test] = decoded[test]
        validation_scores = [
            _metrics(target[validation], pred[validation, column])["balanced_accuracy"]
            for column in range(len(names))
        ]
        selected = int(np.argmax(validation_scores))
        baseline[test] = pred[test, selected]
        selected_by_fold.append({
            "fold": heldout,
            "single": names[selected],
            "threshold": threshold,
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "test_n": int(len(test)),
        })

    categories, matched = [], []
    for qid in qids:
        row = questions.get(qid)
        if row is None:
            categories.append("unknown")
            matched.append(None)
        else:
            category, pattern = classify(row.get("source_question", row.get("question", "")))
            categories.append(category)
            matched.append(pattern)
    categories = np.asarray(categories)
    cached_parent = json.loads(CACHED_PARENT_RESULT.read_text())
    cached_crossfit = cached_parent["cohorts"]["two_models_four_treatments"]["five_fold_crossfit"]

    strata = {}
    category_indices = {
        category: np.flatnonzero(categories == category)
        for category in PRECEDENCE + ["unknown"]
    }
    nonvisual_categories = [
        "management", "history", "temporal_comparison", "procedure",
        "etiology_diagnosis", "unobservable_other",
    ]
    category_indices["not_directly_observable_aggregate"] = np.flatnonzero(
        np.isin(categories, nonvisual_categories)
    )
    for category, idx in category_indices.items():
        if not len(idx):
            continue
        singles = {name: metric_or_null(target[idx], pred[idx, col]) for col, name in enumerate(names)}
        retrospective_best = max(
            names,
            key=lambda name: (
                singles[name]["balanced_accuracy"] if singles[name]["balanced_accuracy"] is not None else -1,
                singles[name]["accuracy"],
            ),
        )
        strata[category] = {
            "n": int(len(idx)),
            "clusters": int(len(np.unique(clusters[idx]))),
            "single_arms": singles,
            "retrospective_best_single_diagnostic_only": retrospective_best,
            "crossfit_selected_single": metric_or_null(target[idx], baseline[idx]),
            "crossfit_paired_code": metric_or_null(target[idx], paired[idx]),
            "paired_vs_crossfit_single_cluster_bootstrap": bootstrap(
                target[idx], paired[idx], baseline[idx], clusters[idx]
            ),
        }

    result = {
        "status": "completed_cpu_heuristic_observability_audit",
        "dataset": "MedHEval CXR-VisHal binary intersection",
        "cohort": names,
        "n": len(qids),
        "clusters": int(len(np.unique(clusters))),
        "global_reproduction": {
            "crossfit_selected_single": metric_or_null(target, baseline),
            "crossfit_paired_code": metric_or_null(target, paired),
            "paired_vs_crossfit_single_cluster_bootstrap": bootstrap(
                target, paired, baseline, clusters
            ),
        },
        "cached_parent_comparison": {
            "cached_crossfit_selected_single": cached_crossfit["baseline"],
            "cached_crossfit_paired_code": cached_crossfit["paired_code"],
            "current_exact_match": (
                cached_crossfit["baseline"] == _metrics(target, baseline)
                and cached_crossfit["paired_code"] == _metrics(target, paired)
            ),
            "warning": (
                "The current baseline prediction reproduces exactly, but the paired classifier differs by "
                "a few claims from the cached parent result. The parent artifact lacks environment and input "
                "hash binding, so this may reflect sklearn-version or artifact drift; use current crossfit "
                "predictions for every subgroup comparison."
            ),
        },
        "complete_case_audit": {
            "per_arm": arm_coverage,
            "four_arm_intersection": {
                "n": len(qids),
                "rate_vs_binary_target": len(qids) / next(iter(arm_coverage.values()))["binary_target_n"],
            },
            "warning": (
                "The paired analysis is complete-case only.  Non-random parse failures can bias both "
                "the apparent single-arm strength and the paired gain."
            ),
        },
        "rule_contract": {
            "precedence": PRECEDENCE,
            "patterns": RULES,
            "semantics": {
                "direct_visual": "appears answerable from one current radiograph: finding, anatomy/location, or visible device position",
                "history": "requires patient history or symptoms not established by pixels",
                "temporal_comparison": "requires a prior study or temporal state",
                "procedure": "asks what procedure/surgery occurred rather than only what hardware is visible",
                "etiology_diagnosis": "asks cause, differential, or interpretation beyond the directly named visual finding",
                "management": "asks recommendation, treatment, follow-up, or next action",
                "unobservable_other": "other explicitly non-pixel information",
                "unknown": "no conservative rule matched; not silently counted as observable",
            },
            "warning": "Lexical heuristic for sensitivity analysis only; not clinician annotation or expert truth.",
        },
        "crossfit_contract": {
            "fold": "sha256(study cluster) modulo 5",
            "heldout": "one fold",
            "validation": "next fold",
            "train": "remaining three folds",
            "baseline": "best single arm selected on that fold's validation set",
            "paired": "HistGradientBoosting on aligned plain/RAG predictions, NLL, token counts, vote and agreement",
            "selected_by_fold": selected_by_fold,
        },
        "source_quality_warning": (
            "The frozen coverage audit marks Hulu CXR plain and RAG outputs N/A for output-quality failure; "
            "this analysis diagnoses the existing intervention result but cannot make it paper-admissible."
        ),
        "strata": strata,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (OUT_DIR / "assignments.jsonl").open("w") as handle:
        for i, qid in enumerate(qids):
            qrow = questions.get(qid, {})
            handle.write(json.dumps({
                "question_id": qid,
                "cluster_id": str(clusters[i]),
                "question": qrow.get("source_question", qrow.get("question")),
                "category": str(categories[i]),
                "matched_pattern": matched[i],
                "ground_truth": int(target[i]),
                "crossfit_single": int(baseline[i]),
                "crossfit_paired": int(paired[i]),
                "arm_predictions": {name: int(pred[i, col]) for col, name in enumerate(names)},
            }) + "\n")
    print(json.dumps({
        category: {
            "n": item["n"],
            "single": item["crossfit_selected_single"],
            "paired": item["crossfit_paired_code"],
            "delta": item["paired_vs_crossfit_single_cluster_bootstrap"],
        }
        for category, item in strata.items()
    }, indent=2))


if __name__ == "__main__":
    main()
