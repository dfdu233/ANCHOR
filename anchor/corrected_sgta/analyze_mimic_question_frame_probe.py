"""Evaluate paired full-sentence MIMIC question-frame answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anchor.corrected_sgta.evaluate_medheval_answers import (
    rule_pope_prediction,
)


VERSION = "mimic-question-frame-analysis-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def label(text: str) -> str | None:
    value = rule_pope_prediction(text)
    return value if value in {"yes", "no"} else None


def metrics(rows: list[dict]) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "invalid": 0}
    for row in rows:
        truth = label(row["ground_truth"])
        prediction = row["rule_prediction"]
        if prediction not in {"yes", "no"}:
            counts["invalid"] += 1
        elif truth == "yes" and prediction == "yes":
            counts["tp"] += 1
        elif truth == "no" and prediction == "no":
            counts["tn"] += 1
        elif truth == "no":
            counts["fp"] += 1
        else:
            counts["fn"] += 1
    n = len(rows)
    sensitivity = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    specificity = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
    return {
        "n": n,
        "accuracy": (counts["tp"] + counts["tn"]) / n,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "parse_rate": 1 - counts["invalid"] / n,
        **counts,
    }


def cluster_bootstrap(
    original: dict[str, dict],
    neutral: dict[str, dict],
    draws: int = 5000,
) -> list[float]:
    common = sorted(set(original) & set(neutral))
    by_image: dict[str, list[str]] = {}
    for key in common:
        by_image.setdefault(original[key]["image_relative"], []).append(key)
    images = sorted(by_image)
    rng = np.random.default_rng(2027)
    differences = []
    for _ in range(draws):
        sampled = rng.choice(images, size=len(images), replace=True)
        original_rows, neutral_rows = [], []
        for copy_index, image in enumerate(sampled):
            for key in by_image[image]:
                first = dict(original[key])
                second = dict(neutral[key])
                first["question_id"] = f"{key}::{copy_index}"
                second["question_id"] = f"{key}::{copy_index}"
                original_rows.append(first)
                neutral_rows.append(second)
        differences.append(
            metrics(neutral_rows)["accuracy"]
            - metrics(original_rows)["accuracy"]
        )
    return [
        float(value)
        for value in np.quantile(differences, [0.025, 0.975])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--judge", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input)
    rule_rows = [dict(row) for row in rows]
    judge_metadata = None
    if args.judge:
        judged = {
            row["id"]: row for row in read_jsonl(args.judge)
        }
        for row in rows:
            row["rule_prediction"] = judged[row["id"]]["judge_prediction"]
        judge_metadata = {
            "path": str(args.judge.resolve()),
            "sha256": sha256(args.judge),
            "model": next(iter(judged.values()))["model"],
        }
    by_frame = {
        frame: [row for row in rows if row["frame"] == frame]
        for frame in ("original", "neutral")
    }
    original = {row["question_id"]: row for row in by_frame["original"]}
    neutral = {row["question_id"]: row for row in by_frame["neutral"]}
    common = sorted(set(original) & set(neutral))
    rescue = sum(
        original[key]["rule_prediction"] != label(original[key]["ground_truth"])
        and neutral[key]["rule_prediction"] == label(neutral[key]["ground_truth"])
        for key in common
    )
    harm = sum(
        original[key]["rule_prediction"] == label(original[key]["ground_truth"])
        and neutral[key]["rule_prediction"] != label(neutral[key]["ground_truth"])
        for key in common
    )
    frame_metrics = {
        frame: metrics(frame_rows)
        for frame, frame_rows in by_frame.items()
    }
    delta = (
        frame_metrics["neutral"]["accuracy"]
        - frame_metrics["original"]["accuracy"]
    )
    interval = cluster_bootstrap(original, neutral)
    result = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "frames": frame_metrics,
        "evaluator": (
            "blind semantic judge" if args.judge else "normalized RULE parser"
        ),
        "judge": judge_metadata,
        "rule_parser_frames": (
            {
                frame: metrics(
                    [row for row in rule_rows if row["frame"] == frame]
                )
                for frame in ("original", "neutral")
            }
            if args.judge
            else None
        ),
        "neutral_minus_original_accuracy": delta,
        "image_cluster_bootstrap_ci95": interval,
        "rescue": rescue,
        "harm": harm,
        "text_change_rate": float(
            np.mean(
                [
                    original[key]["text"] != neutral[key]["text"]
                    for key in common
                ]
            )
        ),
        "decision": {
            "criterion": (
                "neutral improves accuracy, rescue>harm, and parse rate "
                "does not decrease"
            ),
            "gate_passed": bool(
                delta > 0
                and rescue > harm
                and frame_metrics["neutral"]["parse_rate"]
                >= frame_metrics["original"]["parse_rate"]
            ),
        },
        "claim_ceiling": (
            "question framing changes natural-image MIMIC decisions on an "
            "exposed development subset; not a confirmatory method result"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2))
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            1, 2, figsize=(7.8, 3.5), constrained_layout=True
        )
        frames = ["original", "neutral"]
        axes[0].bar(
            frames,
            [frame_metrics[frame]["accuracy"] for frame in frames],
            color=["#30638e", "#d1495b"],
        )
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("MIMIC-64 complete-sentence answers")
        axes[1].bar(
            ["Rescue", "Harm"],
            [rescue, harm],
            color=["#2a9d8f", "#e76f51"],
        )
        axes[1].set_ylabel("Paired questions")
        axes[1].set_title("Effect of neutral framing")
        figure.savefig(args.figure, dpi=220)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
