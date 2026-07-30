"""Compare prompt-conditioned and style-conditioned answer drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from anchor.corrected_sgta.analyze_style_lineage_report_probe import (
    CONCEPTS,
    positive_mention,
)


VERSION = "prompt-style-factorial-analysis-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path, frame: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    selected = []
    for row in rows:
        row["prompt_frame"] = row.get("prompt_frame", frame)
        if row["prompt_frame"] == frame:
            selected.append(row)
    return selected


def presence_value(row: dict) -> float:
    pattern = CONCEPTS[row["disease"]]
    if re.search(pattern, row["text"], re.IGNORECASE) is None:
        return float("nan")
    return float(positive_mention(row["text"], pattern))


def paired_prompt_range(rows: list[dict]) -> tuple[float, dict]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["prototype_id"], row["disease"])
        grouped.setdefault(key, {})[row["prompt_frame"]] = presence_value(row)
    ranges = []
    complete = {}
    for key, values in grouped.items():
        finite = [value for value in values.values() if np.isfinite(value)]
        if len(finite) == 3:
            value = max(finite) - min(finite)
            ranges.append(value)
            complete["::".join(key)] = value
    return float(np.mean(ranges)), complete


def style_range(rows: list[dict]) -> tuple[float, dict]:
    details = {}
    ranges = []
    for frame in sorted({row["prompt_frame"] for row in rows}):
        for disease in sorted({row["disease"] for row in rows}):
            rates = []
            for cluster in sorted({int(row["cluster"]) for row in rows}):
                values = [
                    presence_value(row)
                    for row in rows
                    if row["prompt_frame"] == frame
                    and row["disease"] == disease
                    and int(row["cluster"]) == cluster
                ]
                finite = np.asarray(values)[np.isfinite(values)]
                rates.append(float(finite.mean()) if len(finite) else float("nan"))
            finite_rates = np.asarray(rates)[np.isfinite(rates)]
            value = (
                float(finite_rates.max() - finite_rates.min())
                if len(finite_rates)
                else float("nan")
            )
            ranges.append(value)
            details[f"{frame}::{disease}"] = {
                "cluster_positive_finding_rates": rates,
                "range": value,
            }
    return float(np.nanmean(ranges)), details


def bootstrap_difference(
    prompt_ranges: dict[str, float],
    rows: list[dict],
    draws: int = 5000,
) -> tuple[float, list[float]]:
    prototypes = sorted({row["prototype_id"] for row in rows})
    rng = np.random.default_rng(2027)
    observed_prompt = float(np.mean(list(prompt_ranges.values())))
    observed_style = style_range(rows)[0]
    samples = []
    for _ in range(draws):
        sampled = rng.choice(prototypes, size=len(prototypes), replace=True)
        sampled_rows = []
        sampled_prompt = []
        for copy_index, prototype in enumerate(sampled):
            for row in rows:
                if row["prototype_id"] == prototype:
                    copied = dict(row)
                    copied["prototype_id"] = f"{prototype}::{copy_index}"
                    sampled_rows.append(copied)
            for key, value in prompt_ranges.items():
                if key.startswith(f"{prototype}::"):
                    sampled_prompt.append(value)
        samples.append(float(np.mean(sampled_prompt) - style_range(sampled_rows)[0]))
    interval = np.quantile(samples, [0.025, 0.975]).tolist()
    return observed_prompt - observed_style, [float(value) for value in interval]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "positive": args.positive,
        "neutral": args.neutral,
        "negative": args.negative,
    }
    rows = [
        row
        for frame, path in paths.items()
        for row in read_jsonl(path, frame)
    ]
    prompt_effect, prompt_details = paired_prompt_range(rows)
    style_effect, style_details = style_range(rows)
    difference, interval = bootstrap_difference(prompt_details, rows)
    frame_rates = {}
    disease_frame_rates = {}
    for frame in paths:
        frame_values = np.asarray(
            [
                presence_value(row)
                for row in rows
                if row["prompt_frame"] == frame
            ]
        )
        frame_rates[frame] = float(
            frame_values[np.isfinite(frame_values)].mean()
        )
        for disease in sorted({row["disease"] for row in rows}):
            values = np.asarray(
                [
                    presence_value(row)
                    for row in rows
                    if row["prompt_frame"] == frame
                    and row["disease"] == disease
                ]
            )
            disease_frame_rates[f"{disease}::{frame}"] = float(
                values[np.isfinite(values)].mean()
            )
    parse_rate = float(
        np.mean([np.isfinite(presence_value(row)) for row in rows])
    )
    result = {
        "version": VERSION,
        "inputs": {
            frame: {"path": str(path.resolve()), "sha256": sha256(path)}
            for frame, path in paths.items()
        },
        "n": len(rows),
        "parse_rate": parse_rate,
        "frame_positive_finding_rates": frame_rates,
        "disease_frame_positive_finding_rates": disease_frame_rates,
        "mean_paired_prompt_range": prompt_effect,
        "mean_style_cluster_range": style_effect,
        "prompt_minus_style_range": difference,
        "prompt_minus_style_ci95": interval,
        "style_details": style_details,
        "decision": {
            "criterion": (
                "prompt range >=.20, prompt-style CI lower >0, "
                "parse rate >=.90"
            ),
            "gate_passed": bool(
                prompt_effect >= 0.20
                and interval[0] > 0
                and parse_rate >= 0.90
            ),
        },
        "claim_ceiling": (
            "question-framing effect exceeds source-style-cluster effect on "
            "shared-content synthetic probes; not target accuracy"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2))
    if args.figure:
        import matplotlib.pyplot as plt

        args.figure.parent.mkdir(parents=True, exist_ok=True)
        frames = ["positive", "neutral", "negative"]
        diseases = sorted({row["disease"] for row in rows})
        matrix = np.asarray(
            [
                [
                    disease_frame_rates[f"{disease}::{frame}"]
                    for frame in frames
                ]
                for disease in diseases
            ]
        )
        figure, axes = plt.subplots(
            1, 2, figsize=(8.8, 3.8), constrained_layout=True
        )
        image = axes[0].imshow(
            matrix, vmin=0, vmax=1, cmap="magma", aspect="auto"
        )
        axes[0].set_title("Clinical finding asserted")
        axes[0].set_xticks(range(3), frames)
        axes[0].set_yticks(range(len(diseases)), diseases)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axes[0].text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if matrix[row, column] < 0.55
                        else "black"
                    ),
                )
        figure.colorbar(image, ax=axes[0], label="Rate", shrink=0.8)
        axes[1].bar(
            ["Question frame", "Style cluster"],
            [prompt_effect, style_effect],
            color=["#d1495b", "#30638e"],
        )
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Mean paired decision range")
        axes[1].set_title("Prompt prior dominates style")
        for index, value in enumerate([prompt_effect, style_effect]):
            axes[1].text(
                index, value + 0.03, f"{value:.2f}", ha="center", fontsize=10
            )
        figure.savefig(args.figure, dpi=220)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
