#!/usr/bin/env python3
"""Build and analyze a paired IU-Xray observation-complementarity screen.

``build`` selects one balanced binary claim per study and emits identical
view-0/view-1 manifests for ``run_claim_universe_scoring.py``.  ``analyze``
compares the two independently scored acquisitions, fixed mean-margin fusion,
an oracle ceiling, and a wrong-study second-view permutation placebo.

The shared IU report supplies study-level claim truth.  It is not independent
per-view visibility truth; the output therefore concerns study-level answer
complementarity, not localization or view-specific hallucination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


VERSION = "iuxray-observation-complementarity-v1"


def normalized_binary(value: object) -> str | None:
    text = str(value).strip().lower()
    if text == "yes" or text.startswith("yes,") or text.startswith("yes."):
        return "yes"
    if text == "no" or text.startswith("no,") or text.startswith("no."):
        return "no"
    return None


def study_id(row: dict[str, Any]) -> str:
    return str(row["img_name"]).split("/", 1)[0]


def stable(seed: int, label: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{label}|{row['qid']}".encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> None:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    candidates: dict[str, list[dict[str, Any]]] = {"yes": [], "no": []}
    for row in source:
        if str(row.get("question_type")) != "binary":
            continue
        label = normalized_binary(row.get("answer"))
        if label is None:
            continue
        original = args.image_root / str(row["img_name"])
        sibling_name = str(row["img_name"]).rsplit("/", 1)[0] + "/1.png"
        sibling = args.image_root / sibling_name
        if not original.is_file() or not sibling.is_file():
            continue
        item = dict(row)
        item["normalized_answer"] = label
        item["view1_img_name"] = sibling_name
        candidates[label].append(item)

    selected: list[dict[str, Any]] = []
    used_studies: set[str] = set()
    # The order is frozen as part of the protocol; the final sample is balanced
    # and study-disjoint, so it cannot exploit repeated questions from a study.
    for label in ("yes", "no"):
        ordered = sorted(candidates[label], key=lambda row: stable(args.seed, label, row))
        for row in ordered:
            sid = study_id(row)
            if sid in used_studies:
                continue
            selected.append(row)
            used_studies.add(sid)
            if sum(item["normalized_answer"] == label for item in selected) == args.per_label:
                break
        if sum(item["normalized_answer"] == label for item in selected) != args.per_label:
            raise RuntimeError(f"insufficient study-disjoint {label} claims")

    selected.sort(key=lambda row: int(row["qid"]))
    view0, view1, style = [], [], []
    for row in selected:
        common = {
            "qid": int(row["qid"]),
            "question": str(row["question"]),
            "answer": row["normalized_answer"],
        }
        view0.append({**common, "img_name": str(row["img_name"])})
        view1.append({**common, "img_name": str(row["view1_img_name"])})
        style.append(
            {
                "question_id": int(row["qid"]),
                "question": str(row["question"]),
                "answer": row["normalized_answer"],
                "image": str(row["img_name"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "view0.json", view0)
    write_json(args.output_dir / "view1.json", view1)
    write_jsonl(args.output_dir / "style_view0.jsonl", style)
    write_json(
        args.output_dir / "manifest.json",
        {
            "version": VERSION,
            "source": str(args.input.resolve()),
            "image_root": str(args.image_root.resolve()),
            "seed": args.seed,
            "per_label": args.per_label,
            "n": len(selected),
            "n_studies": len(used_studies),
            "labels": {
                label: sum(row["normalized_answer"] == label for row in selected)
                for label in ("yes", "no")
            },
            "truth_scope": "shared study-report claim; not per-view visibility truth",
            "selection": "stable hash; one binary claim per study; balanced Yes/No",
        },
    )
    print(json.dumps({"status": "complete", "n": len(selected), "output": str(args.output_dir)}))


def load_scores(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            raise ValueError(f"non-ok record in {path}: qid={row.get('question_id')}")
        qid = int(row["question_id"])
        if qid in rows:
            raise ValueError(f"duplicate qid {qid} in {path}")
        rows[qid] = row
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def accuracy(margins: list[float], truths: list[int]) -> float:
    return sum((margin > 0) == bool(truth) for margin, truth in zip(margins, truths)) / len(truths)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(args: argparse.Namespace) -> None:
    left, right = load_scores(args.view0_raw), load_scores(args.view1_raw)
    if set(left) != set(right):
        raise ValueError("view score files have unequal qid sets")
    qids = sorted(left)
    truths: list[int] = []
    margin0: list[float] = []
    margin1: list[float] = []
    for qid in qids:
        label0 = normalized_binary(left[qid].get("truth"))
        label1 = normalized_binary(right[qid].get("truth"))
        if label0 is None or label0 != label1:
            raise ValueError(f"invalid or mismatched truth for qid={qid}")
        truths.append(int(label0 == "yes"))
        margin0.append(float(left[qid]["scores"]["original_margin"]))
        margin1.append(float(right[qid]["scores"]["original_margin"]))

    fused = [(a + b) / 2.0 for a, b in zip(margin0, margin1)]
    confident = [a if abs(a) >= abs(b) else b for a, b in zip(margin0, margin1)]
    acc0, acc1 = accuracy(margin0, truths), accuracy(margin1, truths)
    fused_acc = accuracy(fused, truths)
    confident_acc = accuracy(confident, truths)
    oracle = sum(
        ((a > 0) == bool(y)) or ((b > 0) == bool(y))
        for a, b, y in zip(margin0, margin1, truths)
    ) / len(qids)

    # Probability calibration is learned out-of-fold, never on the evaluated
    # item itself.  This makes the Brier comparison invariant to the arbitrary
    # raw-margin scale of each view and supplies the prespecified secondary
    # endpoint for the observation/computation frontier.
    matrix = np.column_stack([margin0, margin1])
    labels = np.asarray(truths, dtype=int)
    folds = np.asarray([
        int(hashlib.sha256(f"{args.seed}|{qid}|calibration".encode()).hexdigest(), 16) % 5
        for qid in qids
    ])
    base_probability = np.zeros(len(qids), dtype=float)
    fused_probability = np.zeros(len(qids), dtype=float)
    for fold in range(5):
        train, test = folds != fold, folds == fold
        for columns, target in (([0], base_probability), ([0, 1], fused_probability)):
            model = LogisticRegression(C=0.1, max_iter=10000, random_state=args.seed)
            model.fit(matrix[train][:, columns], labels[train])
            target[test] = model.predict_proba(matrix[test][:, columns])[:, 1]
    base_brier = float(brier_score_loss(labels, base_probability))
    fused_brier = float(brier_score_loss(labels, fused_probability))
    base_nll = float(log_loss(labels, base_probability, labels=[0, 1]))
    fused_nll = float(log_loss(labels, fused_probability, labels=[0, 1]))

    rng = random.Random(args.seed)
    bootstrap_delta: list[float] = []
    bootstrap_brier_relative: list[float] = []
    for _ in range(args.bootstrap_draws):
        indices = [rng.randrange(len(qids)) for _ in qids]
        boot0 = [margin0[index] for index in indices]
        bootf = [fused[index] for index in indices]
        booty = [truths[index] for index in indices]
        bootstrap_delta.append(accuracy(bootf, booty) - accuracy(boot0, booty))
        base_boot = float(np.mean((labels[indices] - base_probability[indices]) ** 2))
        fused_boot = float(np.mean((labels[indices] - fused_probability[indices]) ** 2))
        bootstrap_brier_relative.append((base_boot - fused_boot) / max(base_boot, 1e-12))

    shuffle_accuracy: list[float] = []
    indices = list(range(len(qids)))
    for _ in range(args.permutations):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        placebo = [(margin0[i] + margin1[shuffled[i]]) / 2.0 for i in indices]
        shuffle_accuracy.append(accuracy(placebo, truths))
    shuffle_mean = sum(shuffle_accuracy) / len(shuffle_accuracy)
    p_value = (1 + sum(value >= fused_acc for value in shuffle_accuracy)) / (
        len(shuffle_accuracy) + 1
    )
    delta_ci = [quantile(bootstrap_delta, 0.025), quantile(bootstrap_delta, 0.975)]
    brier_relative = (base_brier - fused_brier) / max(base_brier, 1e-12)
    brier_relative_ci = [
        quantile(bootstrap_brier_relative, 0.025),
        quantile(bootstrap_brier_relative, 0.975),
    ]
    result = {
        "version": VERSION,
        "status": "complete",
        "n": len(qids),
        "metrics": {
            "view0_accuracy": acc0,
            "view1_accuracy": acc1,
            "mean_margin_fusion_accuracy": fused_acc,
            "max_abs_margin_accuracy": confident_acc,
            "two_view_oracle_accuracy": oracle,
            "view_disagreement_rate": sum((a > 0) != (b > 0) for a, b in zip(margin0, margin1)) / len(qids),
            "fusion_minus_view0": fused_acc - acc0,
            "fusion_minus_view0_ci95": delta_ci,
            "oracle_minus_view0": oracle - acc0,
            "wrong_study_fusion_accuracy_mean": shuffle_mean,
            "real_fusion_minus_wrong_study_mean": fused_acc - shuffle_mean,
            "wrong_study_permutation_p": p_value,
            "view0_crossfit_brier": base_brier,
            "two_view_crossfit_brier": fused_brier,
            "two_view_relative_brier_improvement": brier_relative,
            "two_view_relative_brier_improvement_ci95": brier_relative_ci,
            "view0_crossfit_nll": base_nll,
            "two_view_crossfit_nll": fused_nll,
        },
        "gate": {
            "fusion_gain_at_least_2pp": fused_acc - acc0 >= 0.02,
            "fusion_gain_ci_excludes_zero": delta_ci[0] > 0,
            "real_over_wrong_study_at_least_2pp": fused_acc - shuffle_mean >= 0.02,
            "oracle_headroom_at_least_5pp": oracle - acc0 >= 0.05,
            "relative_brier_improvement_at_least_5pct_ci_positive": (
                brier_relative >= 0.05 and brier_relative_ci[0] > 0
            ),
        },
        "decision": "GO" if (
            fused_acc - acc0 >= 0.02
            and delta_ci[0] > 0
            and fused_acc - shuffle_mean >= 0.02
            and oracle - acc0 >= 0.05
            and p_value <= 0.05
        ) else "NO-GO",
        "truth_scope": "shared study-report claim; not per-view visibility truth",
        "interpretation_boundary": (
            "A GO supports complementary information in a second acquisition. It does not show "
            "that a decoder can acquire that view, localize a lesion, or reduce report hallucinations."
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input", type=Path, required=True)
    build_parser.add_argument("--image-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--per-label", type=int, default=128)
    build_parser.add_argument("--seed", type=int, default=20260812)
    build_parser.set_defaults(function=build)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--view0-raw", type=Path, required=True)
    analyze_parser.add_argument("--view1-raw", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--bootstrap-draws", type=int, default=5000)
    analyze_parser.add_argument("--permutations", type=int, default=2000)
    analyze_parser.add_argument("--seed", type=int, default=42)
    analyze_parser.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
