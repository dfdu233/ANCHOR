"""Test whether PubMedVision style clusters switch open-report clinical priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


VERSION = "style-lineage-report-analysis-v1"
CONCEPTS = {
    "pneumothorax": r"\bpneumothora",
    "effusion": r"\b(?:pleural )?effusion",
    "opacity": r"\b(?:opacity|opacities|consolidation|infiltrate)",
    "cardiomegaly": r"\b(?:cardiomegaly|enlarged cardiac|enlarged heart)",
    "edema": r"\b(?:pulmonary )?edema",
    "device": r"\b(?:catheter|tube|pacemaker|line|devices?)\b",
}
NEGATION = re.compile(
    r"\b(?:no|without|absent|negative for|not|neither)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def positive_mention(text: str, pattern: str) -> int:
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for match in matches:
        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        endings = [
            index
            for index in (
                text.find(".", match.end()),
                text.find(";", match.end()),
                text.find("\n", match.end()),
            )
            if index >= 0
        ]
        sentence_end = min(endings) if endings else len(text)
        sentence = text[sentence_start + 1 : sentence_end]
        if NEGATION.search(sentence) is None:
            return 1
    return 0


def cluster_effect(labels: np.ndarray, clusters: np.ndarray) -> float:
    overall = float(labels.mean())
    total = float(np.square(labels - overall).sum())
    if total == 0:
        return 0.0
    between = 0.0
    for cluster in np.unique(clusters):
        group = labels[clusters == cluster]
        between += len(group) * float((group.mean() - overall) ** 2)
    return between / total


def permutation_p(
    labels: np.ndarray,
    clusters: np.ndarray,
    draws: int = 10000,
) -> tuple[float, float]:
    observed = cluster_effect(labels, clusters)
    rng = np.random.default_rng(2027)
    null = np.asarray(
        [cluster_effect(labels, rng.permutation(clusters)) for _ in range(draws)]
    )
    p_value = float((1 + np.sum(null >= observed)) / (draws + 1))
    return observed, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-analysis", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input)
    clusters = np.asarray([int(row["cluster"]) for row in rows])
    concept_results = {}
    supported = 0
    for concept, pattern in CONCEPTS.items():
        labels = np.asarray(
            [positive_mention(row["text"], pattern) for row in rows],
            dtype=float,
        )
        effect, p_value = permutation_p(labels, clusters)
        rates = [
            float(labels[clusters == cluster].mean())
            for cluster in sorted(np.unique(clusters))
        ]
        rate_range = max(rates) - min(rates)
        passes = bool(rate_range >= 2 / 3 and p_value < 0.05)
        supported += int(passes)
        concept_results[concept] = {
            "positive_rate": float(labels.mean()),
            "cluster_positive_rates": rates,
            "cluster_eta_squared": effect,
            "permutation_p": p_value,
            "rate_range": rate_range,
            "passes": passes,
        }
    texts = [row["text"].strip() for row in rows]
    result = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "n": len(rows),
        "unique_texts": len(set(texts)),
        "mean_words": float(np.mean([len(text.split()) for text in texts])),
        "concepts": concept_results,
        "decision": {
            "criterion": (
                "at least two clinical concepts have cluster mention-rate "
                "range >=2/3 and cluster-label permutation p<.05"
            ),
            "supported_concepts": supported,
            "gate_passed": bool(supported >= 2),
        },
        "claim_ceiling": (
            "style-conditioned clinical mention prior on synthetic "
            "shared-content source prototypes; not diagnostic accuracy"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2))
    if args.figure:
        import matplotlib.pyplot as plt

        args.figure.parent.mkdir(parents=True, exist_ok=True)
        report_names = list(CONCEPTS)
        report_matrix = np.asarray(
            [
                concept_results[name]["cluster_positive_rates"]
                for name in report_names
            ]
        )
        matrices = [report_matrix]
        names = [report_names]
        titles = ["Open reports: positive clinical mentions"]
        if args.binary_analysis:
            binary = json.loads(args.binary_analysis.read_text())
            binary_names = list(binary["medical_by_disease"])
            binary_matrix = np.asarray(
                [
                    binary["medical_by_disease"][name]["cluster_yes_rate"]
                    for name in binary_names
                ]
            )
            matrices.insert(0, binary_matrix)
            names.insert(0, binary_names)
            titles.insert(0, "Binary sentences: affirmative answers")
        figure, axes = plt.subplots(
            1,
            len(matrices),
            figsize=(5.2 * len(matrices), 3.7),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for axis, matrix, row_names, title in zip(
            axes, matrices, names, titles, strict=True
        ):
            image = axis.imshow(
                matrix, vmin=0, vmax=1, cmap="magma", aspect="auto"
            )
            axis.set_title(title, fontsize=11)
            axis.set_xlabel("PubMedVision style cluster")
            axis.set_xticks(range(matrix.shape[1]))
            axis.set_yticks(range(matrix.shape[0]), row_names)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=(
                            "white"
                            if matrix[row, column] < 0.55
                            else "black"
                        ),
                    )
        figure.colorbar(image, ax=axes.tolist(), label="Rate", shrink=0.85)
        figure.savefig(args.figure, dpi=220)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
