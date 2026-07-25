#!/usr/bin/env python3
"""Generate a protocol-aware paper figure from aligned baseline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_LABELS = {
    "context": "Context",
    "cxr_vishal": "CXR",
    "knowledge_ce": "Knowledge",
    "mm_vishal": "MM",
}
METHODS = ["Baseline", "SGTA", "LATA", "SCA-T TIM-KL"]
COLORS = {
    "Baseline": "#4C78A8",
    "SGTA": "#F58518",
    "LATA": "#54A24B",
    "SCA-T TIM-KL": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    ce_rows = [
        row for row in data["rows"]
        if row.get("protocol") == "ce_logits" and row.get("status") == "ok"
    ]
    by_key = {(row["model"], row["task"], row["method"]): row for row in ce_rows}
    tasks = ["context", "cxr_vishal", "knowledge_ce", "mm_vishal"]
    models = ["hulu", "llava"]

    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    width = 0.18
    x = list(range(len(tasks)))
    for ax, model in zip(axes, models):
        for i, method in enumerate(METHODS):
            vals = []
            labels = []
            for task in tasks:
                row = by_key.get((model, task, method))
                vals.append(None if row is None else 100.0 * float(row["accuracy"]))
                labels.append("" if row is None else f"n={row['n']}")
            xpos = [v + (i - 1.5) * width for v in x]
            ax.bar(xpos, [0 if v is None else v for v in vals], width=width, color=COLORS[method], label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=20, ha="right")
        ax.set_ylim(35, 95)
        ax.set_ylabel("Accuracy (%)" if model == "hulu" else "")
        ax.text(0.02, 0.96, "Hulu-Med" if model == "hulu" else "LLaVA-Med",
                transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / "fig_baseline_aligned_ce.pdf"
    png = args.out_dir / "fig_baseline_aligned_ce.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)

    latex = args.out_dir / "fig_baseline_aligned_ce.tex"
    latex.write_text(r"""\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{Figures/fig_baseline_aligned_ce.pdf}
    \caption{Protocol-aligned closed-ended MedHEval results under the CE logit interface. All bars in this figure are computed from the same finite-label evidence caches for each model and task; architecture-specific generative decoding baselines are reported separately because they use a different prompt and parsing protocol.}
    \label{fig:baseline_aligned_ce}
\end{figure}
""")
    print(json.dumps({"pdf": str(pdf), "png": str(png), "latex": str(latex)}, indent=2))


if __name__ == "__main__":
    main()
