"""Audit question-conditioned clinical priors in PubMedVision CXR data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.analyze_pubmed_style_prior import (
    LABEL_PATTERNS,
    question_answer,
    select_rows,
    sha256,
)
from anchor.corrected_sgta.analyze_style_lineage_report_probe import (
    positive_mention,
)


VERSION = "pubmed-question-prior-audit-v1"
SEED = 2027


def group_bootstrap_difference(
    cue: np.ndarray,
    positive: np.ndarray,
    groups: np.ndarray,
    draws: int = 1000,
) -> list[float]:
    unique, inverse = np.unique(groups, return_inverse=True)
    totals = np.zeros((len(unique), 4), dtype=float)
    np.add.at(totals[:, 0], inverse, cue * positive)
    np.add.at(totals[:, 1], inverse, cue)
    np.add.at(totals[:, 2], inverse, (1 - cue) * positive)
    np.add.at(totals[:, 3], inverse, 1 - cue)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(draws):
        sampled = totals[
            rng.integers(0, len(unique), size=len(unique))
        ].sum(axis=0)
        cue_rate = sampled[0] / max(sampled[1], 1)
        other_rate = sampled[2] / max(sampled[3], 1)
        differences.append(float(cue_rate - other_rate))
    return [
        float(value)
        for value in np.quantile(differences, [0.025, 0.975])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--max-images", type=int, default=10000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args.manifest, args.max_images)
    questions, answers, groups = [], [], []
    for row in rows:
        question, answer = question_answer(row)
        questions.append(question)
        answers.append(answer)
        groups.append(str(row.get("group_id") or row["id"]))
    groups_array = np.asarray(groups)
    metrics = []
    supported = 0
    for concept, pattern in LABEL_PATTERNS.items():
        cue = np.asarray(
            [bool(re.search(pattern, text, re.IGNORECASE)) for text in questions],
            dtype=int,
        )
        positive = np.asarray(
            [positive_mention(text, pattern) for text in answers],
            dtype=int,
        )
        cue_n = int(cue.sum())
        other_n = int(len(cue) - cue_n)
        cue_rate = float(positive[cue == 1].mean()) if cue_n else float("nan")
        other_rate = (
            float(positive[cue == 0].mean()) if other_n else float("nan")
        )
        difference = cue_rate - other_rate
        interval = group_bootstrap_difference(
            cue, positive, groups_array
        )
        passes = bool(
            concept != "normal"
            and cue_n >= 20
            and interval[0] > 0.25
        )
        supported += int(passes)
        metrics.append(
            {
                "concept": concept,
                "question_cue_n": cue_n,
                "question_no_cue_n": other_n,
                "answer_positive_given_cue": cue_rate,
                "answer_positive_without_cue": other_rate,
                "difference": difference,
                "group_bootstrap_ci95": interval,
                "passes": passes,
            }
        )
    result = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n_unique_strict_cxr": len(rows),
        "unique_groups": len(set(groups)),
        "target_data_accessed": False,
        "definition": (
            "Question cue is a frozen concept regex; positive answer is a "
            "non-negated mention in the complete reference answer."
        ),
        "metrics": metrics,
        "decision": {
            "criterion": (
                "at least four non-normal concepts with >=20 cue examples "
                "and group-bootstrap difference CI lower >.25"
            ),
            "supported_concepts": supported,
            "gate_passed": bool(supported >= 4),
        },
        "claim_ceiling": (
            "PubMedVision CXR questions carry a strong lexical prior for "
            "reference clinical content; this does not show model causality"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2))
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        eligible = [
            item
            for item in metrics
            if item["concept"] != "normal"
            and item["question_cue_n"] >= 20
        ]
        positions = np.arange(len(eligible))
        width = 0.36
        figure, axis = plt.subplots(
            figsize=(8.2, 3.8), constrained_layout=True
        )
        axis.bar(
            positions - width / 2,
            [item["answer_positive_given_cue"] for item in eligible],
            width,
            label="Concept named in question",
            color="#d1495b",
        )
        axis.bar(
            positions + width / 2,
            [item["answer_positive_without_cue"] for item in eligible],
            width,
            label="Concept not named",
            color="#30638e",
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Positive concept rate in reference answer")
        axis.set_xticks(
            positions,
            [item["concept"] for item in eligible],
            rotation=20,
            ha="right",
        )
        axis.set_title(
            "PubMedVision CXR questions predict reference clinical content"
        )
        axis.legend(frameon=False, loc="upper right")
        figure.savefig(args.figure, dpi=220)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
