#!/usr/bin/env python3
"""Create one protocol-aware LaTeX table for CE and generative baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ["context", "cxr_vishal", "knowledge_ce", "mm_vishal"]
TASK_LABEL = {"context": "Context", "cxr_vishal": "CXR", "knowledge_ce": "Knowledge", "mm_vishal": "MM"}
CE_ORDER = ["Baseline", "FedDG", "Gamma-TTA", "SGTA", "LAME", "LATA", "SCA-T TIM-KL"]
GEN_ORDER = ["greedy", "DoLa", "PAI", "opera", "avisc", "m3id", "VCD", "damro"]


def pct(value):
    return "--" if value is None else f"{100*float(value):.1f}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("corrected_runs/aaai_aligned_baseline_summary_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/AuthorKit27/result_tables/TABLE_tab_unified_protocol_aware_baselines.tex"))
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.summary.read_text())
    rows = data.get("rows", [])
    by = {(r.get("protocol"), r.get("model"), r.get("task"), r.get("method")): r for r in rows}
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{Unified full-model comparison on MedHEval. All optimization methods are reported through the corrected SGTA experiment registry with resumable cached outputs and recorded fingerprints. Accuracy is shown in percent; unavailable or still-running cells are marked by --.}")
    lines.append(r"\label{tab:unified_baselines_and_optimizers}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Method & Context & CXR & Knowledge & MM \\")
    lines.append(r"\midrule")
    for model in ["hulu", "llava"]:
        for method in CE_ORDER:
            vals = [pct((by.get(("ce_logits", model, task, method)) or {}).get("accuracy")) for task in TASKS]
            mname = "Hulu-Med" if model == "hulu" else "LLaVA-Med"
            lines.append(f"{mname} & {method} & " + " & ".join(vals) + r" \\")
        if model == "hulu":
            lines.append(r"\midrule")
    # Architecture-specific mitigation methods are integrated into the same
    # corrected_sgta registry and table. Values are filled as chunks finish.
    for method in GEN_ORDER:
        vals = [pct((by.get(("official_generative", "llava", task, method)) or {}).get("accuracy")) for task in TASKS]
        if any(v != "--" for v in vals):
            lines.append(f"LLaVA-Med & {method} & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
