#!/usr/bin/env python3
"""Analyze ANCHOR-Flow-SGTA gate outputs.

The gate asks whether SGTA/FedDG-style views create useful full-sentence
candidates and whether output-path source energy selects better responses.
It does not use target labels for selection; labels in the raw file are used
only because this is an analysis step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "anchor-flow-sgta-gate-analysis-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(raw: Path) -> dict[str, Any]:
    rows = load_jsonl(raw)
    task = rows[0].get("task", "unknown") if rows else "unknown"
    methods = {
        "greedy": "greedy_score",
        "nll": "nll_score",
        "random": "random_score",
        "anchor_flow": "anchor_flow_score",
        "oracle": "oracle_score",
    }
    out: dict[str, Any] = {"version": VERSION, "raw": str(raw), "task": task, "n": len(rows)}
    greedy_scores = [float(r["greedy_score"]) for r in rows]
    for name, key in methods.items():
        scores = [float(r[key]) for r in rows]
        out[name] = {
            "mean_score": mean(scores),
            "delta_vs_greedy": mean(scores) - mean(greedy_scores),
        }
        if task == "ce":
            out[name]["accuracy"] = mean(scores)
    out["diversity"] = {
        "mean_unique_text_rate": mean([r["unique_text_count"] / max(1, len(r["candidates"])) for r in rows]),
        "view_disagreement_rate": mean([float(bool(r["view_disagreement"])) for r in rows]),
        "oracle_headroom": out["oracle"]["mean_score"] - out["greedy"]["mean_score"],
    }
    out["anchor_flow_rescue_harm"] = {
        "rescue": sum(float(r["anchor_flow_score"]) > float(r["greedy_score"]) for r in rows),
        "harm": sum(float(r["anchor_flow_score"]) < float(r["greedy_score"]) for r in rows),
        "unchanged": sum(float(r["anchor_flow_score"]) == float(r["greedy_score"]) for r in rows),
    }
    out["oracle_rescue_harm"] = {
        "rescue": sum(float(r["oracle_score"]) > float(r["greedy_score"]) for r in rows),
        "harm": sum(float(r["oracle_score"]) < float(r["greedy_score"]) for r in rows),
        "unchanged": sum(float(r["oracle_score"]) == float(r["greedy_score"]) for r in rows),
    }
    if task == "oe":
        def normal(text: str) -> bool:
            v = " ".join(str(text).lower().split())
            return "appears to be normal" in v or "no significant abnormal" in v or "no acute cardiopulmonary" in v
        out["normal_template_rate"] = {
            "greedy": mean([float(normal(r["greedy_text"])) for r in rows]),
            "anchor_flow": mean([float(normal(r["anchor_flow_text"])) for r in rows]),
        }
    out["gate"] = {
        "style_candidate_oracle_pass": bool(out["diversity"]["oracle_headroom"] >= (0.05 if task == "ce" else 0.01)),
        "anchor_flow_beats_nll": bool(out["anchor_flow"]["mean_score"] > out["nll"]["mean_score"]),
        "anchor_flow_beats_random": bool(out["anchor_flow"]["mean_score"] > out["random"]["mean_score"]),
        "continue": False,
    }
    out["gate"]["continue"] = all([
        out["gate"]["style_candidate_oracle_pass"],
        out["gate"]["anchor_flow_beats_nll"],
        out["gate"]["anchor_flow_beats_random"],
    ])
    bad = []
    interesting = []
    for r in rows:
        if r["unique_text_count"] > 1 or float(r["oracle_score"]) > float(r["greedy_score"]) or float(r["anchor_flow_score"]) < float(r["greedy_score"]):
            item = {
                "id": r["id"],
                "greedy_score": r["greedy_score"],
                "anchor_flow_score": r["anchor_flow_score"],
                "oracle_score": r["oracle_score"],
                "selected_view": r["candidates"][r["anchor_flow_selected_index"]]["view_name"],
                "unique_text_count": r["unique_text_count"],
                "texts_by_view": {c["view_name"]: c["text"] for c in r["candidates"]},
            }
            (bad if float(r["anchor_flow_score"]) < float(r["greedy_score"]) else interesting).append(item)
    out["interesting_cases"] = interesting[:20]
    out["harm_cases"] = bad[:20]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"version": VERSION, "analyses": [summarize(path) for path in args.raw]}
    payload["overall_continue"] = any(item["gate"]["continue"] for item in payload["analyses"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "overall_continue": payload["overall_continue"]}, indent=2))


if __name__ == "__main__":
    main()
