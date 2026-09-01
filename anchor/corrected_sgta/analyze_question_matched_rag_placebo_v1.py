"""Question- and label-matched placebo for the CXR RAG response code.

The earlier cross-patient placebo also changed question semantics.  This audit
shuffles each RAG arm only among cases with the exact same source question and
ground-truth label, within every outer fold.  It therefore preserves question
prior, label stratum, arm marginals, and model identity while breaking only
patient/image alignment where an exchangeable peer exists.
"""

from __future__ import annotations

import hashlib
import json
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


INPUT = Path("corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json")
OUT = Path("corrected_runs/question_matched_rag_placebo_v1/result.json")
NAMES = ["huatuo_plain", "huatuo_rag", "hulu_plain", "hulu_rag"]


def _fast_bootstrap_delta(
    y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, cluster: np.ndarray
) -> dict[str, float | list[float]]:
    unique = np.unique(cluster)
    counts = np.zeros((len(unique), 8), dtype=float)
    for row, value in enumerate(unique):
        take = cluster == value
        for offset, pred in ((0, candidate), (4, baseline)):
            counts[row, offset + 0] = np.sum(take & (y == 1) & (pred == 1))
            counts[row, offset + 1] = np.sum(take & (y == 0) & (pred == 0))
            counts[row, offset + 2] = np.sum(take & (y == 0) & (pred == 1))
            counts[row, offset + 3] = np.sum(take & (y == 1) & (pred == 0))
    rng = np.random.default_rng(42)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=5000)
    boot = weights @ counts

    def bacc(values: np.ndarray) -> np.ndarray:
        tp, tn, fp, fn = values.T
        return 0.5 * (tp / np.maximum(1, tp + fn) + tn / np.maximum(1, tn + fp))

    delta = bacc(boot[:, :4]) - bacc(boot[:, 4:])
    point = (
        _metrics(y, candidate)["balanced_accuracy"] - _metrics(y, baseline)["balanced_accuracy"]
    )
    return {
        "delta": float(point),
        "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
    }


def main() -> None:
    loaded = {name: _load_expert(ARMS[name]) for name in NAMES}
    qids = sorted(set.intersection(*(set(loaded[name]) for name in NAMES)))
    manifest = {row["qid"]: row for row in json.loads(INPUT.read_text())}
    pred = np.asarray([[loaded[name][qid]["pred"] for name in NAMES] for qid in qids])
    nll = np.asarray([[loaded[name][qid]["nll"] for name in NAMES] for qid in qids])
    tokens = np.asarray([[loaded[name][qid]["tokens"] for name in NAMES] for qid in qids])
    y = np.asarray([loaded[NAMES[0]][qid]["target"] for qid in qids])
    cluster = np.asarray([loaded[NAMES[0]][qid]["cluster"] for qid in qids])
    question = np.asarray([manifest[qid]["source_question"].strip().lower() for qid in qids])
    fold = np.asarray([
        int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5 for value in cluster
    ])

    paired = np.zeros(len(y), dtype=int)
    question_placebo = np.zeros(len(y), dtype=int)
    question_label_placebo = np.zeros(len(y), dtype=int)
    question_test_eligible = np.zeros(len(y), dtype=bool)
    question_label_test_eligible = np.zeros(len(y), dtype=bool)
    best_single = np.zeros(len(y), dtype=int)
    rng = np.random.default_rng(20260810)
    coverage = {
        "question_only": {"reassigned": 0, "eligible": 0, "group_sizes": []},
        "question_and_label": {"reassigned": 0, "eligible": 0, "group_sizes": []},
    }

    for heldout in range(5):
        test = np.flatnonzero(fold == heldout)
        validation = np.flatnonzero(fold == ((heldout + 1) % 5))
        train = np.flatnonzero((fold != heldout) & (fold != ((heldout + 1) % 5)))
        decoded, _ = fit_decode_indices(make_features(pred, nll, tokens), y, train, validation)
        paired[test] = decoded[test]

        placebo_outputs = []
        for placebo_name, use_label in (("question_only", False), ("question_and_label", True)):
            ppred, pnll, ptokens = pred.copy(), nll.copy(), tokens.copy()
            for part in (train, validation, test):
                strata: dict[tuple, list[int]] = {}
                for index in part:
                    key = (question[index], int(y[index])) if use_label else (question[index],)
                    strata.setdefault(key, []).append(int(index))
                for members in strata.values():
                    if len(members) < 2:
                        continue
                    members_array = np.asarray(members)
                    if np.array_equal(part, test):
                        if use_label:
                            question_label_test_eligible[members_array] = True
                        else:
                            question_test_eligible[members_array] = True
                        coverage[placebo_name]["group_sizes"].append(len(members))
                        coverage[placebo_name]["eligible"] += len(members) * 2
                    for column in (1, 3):
                        order = rng.permutation(members_array)
                        if np.all(order == members_array):
                            order = np.roll(order, 1)
                        ppred[members_array, column] = pred[order, column]
                        pnll[members_array, column] = nll[order, column]
                        ptokens[members_array, column] = tokens[order, column]
                        if np.array_equal(part, test):
                            coverage[placebo_name]["reassigned"] += int(
                                np.sum(order != members_array)
                            )
            decoded_placebo, _ = fit_decode_indices(
                make_features(ppred, pnll, ptokens), y, train, validation
            )
            placebo_outputs.append(decoded_placebo[test])
        question_placebo[test], question_label_placebo[test] = placebo_outputs

        validation_bacc = [
            _metrics(y[validation], pred[validation, column])["balanced_accuracy"]
            for column in range(len(NAMES))
        ]
        chosen = int(np.argmax(validation_bacc))
        best_single[test] = pred[test, chosen]

    result = {
        "status": "completed_fatal_patient_alignment_control",
        "n": int(len(y)),
        "arms": NAMES,
        "placebo_contract": {
            "question_only": "exact source question + outer-fold matched RAG shuffle; no target label used",
            "question_and_label": "exact source question + ground-truth label + outer-fold matched RAG shuffle; conservative oracle control",
        },
        "placebo_coverage": {
            name: {
                "eligible_rag_cells": int(values["eligible"]),
                "actually_reassigned_rag_cells": int(values["reassigned"]),
                "fraction_of_all_rag_cells_reassigned": float(values["reassigned"] / (len(y) * 2)),
                "median_exchange_group_size": (
                    float(np.median(values["group_sizes"])) if values["group_sizes"] else 0.0
                ),
            }
            for name, values in coverage.items()
        },
        "best_single": _metrics(y, best_single),
        "paired_code": _metrics(y, paired),
        "question_only_matched_placebo": _metrics(y, question_placebo),
        "question_label_matched_placebo": _metrics(y, question_label_placebo),
        "exchangeable_question_only_subset": {
            "n": int(question_test_eligible.sum()),
            "paired": _metrics(y[question_test_eligible], paired[question_test_eligible]),
            "placebo": _metrics(
                y[question_test_eligible], question_placebo[question_test_eligible]
            ),
            "paired_vs_placebo": _fast_bootstrap_delta(
                y[question_test_eligible],
                paired[question_test_eligible],
                question_placebo[question_test_eligible],
                cluster[question_test_eligible],
            ),
        },
        "exchangeable_question_label_subset": {
            "n": int(question_label_test_eligible.sum()),
            "paired": _metrics(
                y[question_label_test_eligible], paired[question_label_test_eligible]
            ),
            "placebo": _metrics(
                y[question_label_test_eligible],
                question_label_placebo[question_label_test_eligible],
            ),
            "paired_vs_placebo": _fast_bootstrap_delta(
                y[question_label_test_eligible],
                paired[question_label_test_eligible],
                question_label_placebo[question_label_test_eligible],
                cluster[question_label_test_eligible],
            ),
        },
        "paired_vs_best_single": _fast_bootstrap_delta(y, paired, best_single, cluster),
        "paired_vs_question_label_matched_placebo": _fast_bootstrap_delta(
            y, paired, question_label_placebo, cluster
        ),
        "paired_vs_question_only_matched_placebo": _fast_bootstrap_delta(
            y, paired, question_placebo, cluster
        ),
        "decision_rule": (
            "Patient-specific response evidence requires paired code to beat the label-free exact-question "
            "placebo with a cluster-bootstrap CI excluding zero and substantial placebo coverage."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
