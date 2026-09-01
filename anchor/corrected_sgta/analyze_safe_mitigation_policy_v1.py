#!/usr/bin/env python3
"""Risk-controlled mitigation policy over cached Huatuo interventions.

The policy treats the strongest single intervention as standard care and a
learned intervention-code predictor as a candidate treatment.  A disjoint
calibration split certifies that, among cases where the policy changes the
baseline answer, rescues are more frequent than harms using an exact one-sided
binomial bound.  The test split is opened only after this rule is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier

from anchor.corrected_sgta.analyze_competition_routing_pilot_v1 import (
    EXPERTS, _bootstrap_delta, _load_expert, _metrics, _split,
)


OUT = Path("corrected_runs/safe_mitigation_policy_v1/result.json")
NAMES = ["huatuo_native", "huatuo_common_prompt", "huatuo_rag"]


def fine_split(cluster: str) -> str:
    coarse = _split(cluster)
    if coarse != "train":
        return "calibration" if coarse == "validation" else "test"
    import hashlib
    bucket = int(hashlib.sha256(("policy:" + cluster).encode()).hexdigest()[:8], 16) % 3
    return "fit" if bucket < 2 else "tune"


def upper_harm_probability(harms: int, changes: int, alpha: float = 0.05) -> float:
    if changes == 0:
        return 1.0
    if harms == changes:
        return 1.0
    return float(beta.ppf(1 - alpha, harms + 1, changes - harms))


def main() -> None:
    loaded = {name: _load_expert(EXPERTS[name]) for name in NAMES}
    qids = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    rows = [[loaded[name][qid] for name in NAMES] for qid in qids]
    pred = np.asarray([[item["pred"] for item in row] for row in rows])
    nll = np.asarray([[item["nll"] for item in row] for row in rows])
    tokens = np.asarray([[item["tokens"] for item in row] for row in rows])
    target = np.asarray([row[0]["target"] for row in rows])
    clusters = np.asarray([row[0]["cluster"] for row in rows])
    split = np.asarray([fine_split(cluster) for cluster in clusters])
    index = {name: np.flatnonzero(split == name) for name in ("fit", "tune", "calibration", "test")}
    vote = pred.mean(1, keepdims=True)
    features = np.concatenate([pred, nll, tokens, vote, np.abs(vote - 0.5) * 2], axis=1)

    # Standard care is selected without touching calibration/test labels.
    baseline_column = max(
        range(len(NAMES)),
        key=lambda col: _metrics(target[index["tune"]], pred[index["tune"], col])["balanced_accuracy"],
    )
    baseline = pred[:, baseline_column]
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=100, max_leaf_nodes=9,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(features[index["fit"]], target[index["fit"]])
    probability = model.predict_proba(features)[:, 1]
    tune = index["tune"]
    tune_baseline = _metrics(target[tune], baseline[tune])
    decision_choices = []
    for value in np.linspace(0.15, 0.75, 61):
        proposed = (probability >= value).astype(int)
        metric = _metrics(target[tune], proposed[tune])
        feasible = metric["fp"] <= tune_baseline["fp"] and metric["fn"] <= tune_baseline["fn"]
        decision_choices.append((feasible, metric["balanced_accuracy"], -abs(value - 0.5), value))
    feasible_decision, _, _, decision_threshold = max(decision_choices)
    if not feasible_decision:
        raise RuntimeError("no directionally non-inferior candidate threshold")
    candidate = (probability >= decision_threshold).astype(int)
    confidence = np.abs(probability - decision_threshold)

    # The candidate threshold was already required to improve both error
    # directions on tune.  Calibration now certifies each complete direction;
    # it does not optimize a second confidence threshold on the same labels.
    proposed_delete = (baseline == 1) & (candidate == 0)
    proposed_add = (baseline == 0) & (candidate == 1)
    calibration = index["calibration"]
    certificates = {}
    certified_masks = []
    for direction, proposed in (("delete_positive", proposed_delete), ("add_positive", proposed_add)):
        changed = calibration[proposed[calibration]]
        harms = int(np.sum((baseline[changed] == target[changed]) & (candidate[changed] != target[changed])))
        rescues = int(np.sum((baseline[changed] != target[changed]) & (candidate[changed] == target[changed])))
        upper = upper_harm_probability(harms, len(changed))
        certified = bool(len(changed) >= 5 and upper < 0.5)
        certificates[direction] = {
            "changes": int(len(changed)), "rescues": rescues, "harms": harms,
            "harm_fraction": float(harms / len(changed)) if len(changed) else None,
            "one_sided_95_upper_harm_probability": upper,
            "certified_rescues_dominate_harms": certified,
        }
        if certified:
            certified_masks.append(proposed)
    safe_change = np.logical_or.reduce(certified_masks) if certified_masks else np.zeros(len(target), dtype=bool)
    policy = np.where(safe_change, candidate, baseline)
    test = index["test"]
    result = {
        "status": "blind_test_complete",
        "principle": "intervene only when a held-out exact risk certificate shows rescues dominate harms",
        "experts": NAMES,
        "split_n": {name: int(len(values)) for name, values in index.items()},
        "baseline_selected_on_tune": NAMES[baseline_column],
        "candidate_decision_threshold_selected_on_tune": float(decision_threshold),
        "calibration_certificates": certificates,
        "test": {
            "baseline": _metrics(target[test], baseline[test]),
            "unconstrained_candidate": _metrics(target[test], candidate[test]),
            "safe_policy": _metrics(target[test], policy[test]),
            "change_rate": float(np.mean(policy[test] != baseline[test])),
            "delta_vs_baseline": _bootstrap_delta(target[test], policy[test], baseline[test], clusters[test]),
        },
        "boundary": "CE cached pilot; not yet OE/report efficacy or a distribution-free clinical safety guarantee",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
