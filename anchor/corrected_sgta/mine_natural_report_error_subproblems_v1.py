#!/usr/bin/env python3
"""Label-free lexical screen for recurring natural-report error subproblems.

This is an exploratory substrate screen, not a clinical metric.  It compares model text
with the paired report and only counts conservative high-precision phrase families.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


FAMILIES = {
    "ett": r"\bendotracheal(?: tube)?\b|\bett\b|\bintubat(?:ed|ion)\b",
    "enteric_tube": r"\benteric tube\b|\bnasogastric tube\b|\bng tube\b|\bfeeding tube\b",
    "central_line": r"\bcentral venous catheter\b|\bcentral line\b|\bpicc\b|\bcvc\b|\b(?:jugular|subclavian) (?:venous )?catheter\b",
    "chest_tube": r"\bchest tube\b|\bthoracostomy tube\b|\bpigtail(?: catheter)?\b",
    "pacemaker": r"\bpacemaker\b|\bpacing (?:wire|lead)\b|\bdefibrillator\b|\baicd\b",
    "sternotomy": r"\bsternotomy (?:wire|wires)\b",
    "prosthetic_valve": r"\b(?:aortic|mitral|tricuspid) valve (?:prosthesis|replacement|annuloplasty)\b|\bprosthetic (?:heart )?valve\b",
    "surgical_clips": r"\bsurgical clips?\b|\bclips? (?:are )?(?:seen|noted|projecting)\b",
    "fracture": r"\bfractur(?:e|es|ed)\b",
    "mass_nodule": r"\bmass(?:es)?\b|\bnodules?\b",
    "lymphadenopathy": r"\blymphadenopathy\b|\benlarged lymph nodes?\b",
    "fibrosis": r"\bfibro(?:sis|tic)\b",
    "hyperinflation_copd": r"\bhyperinflation\b|\bhyperexpanded\b|\bcopd\b|\bemphysema\b",
    "vascular_calcification": r"\b(?:aortic|coronary|vascular) calcification(?:s)?\b|\bcalcified aorta\b",
}


def has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    counts = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    by_model = defaultdict(int)
    total = 0
    for path in args.paths:
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                ref = row.get("ground_truth") or row.get("reference") or ""
                pred = row.get("model_answer") or row.get("prediction") or row.get("answer") or ""
                if not ref or not pred:
                    continue
                model = str(row.get("model", path.parts[-4] if len(path.parts) >= 4 else "unknown"))
                total += 1
                by_model[model] += 1
                for family, pattern in FAMILIES.items():
                    r = has(pattern, ref)
                    p = has(pattern, pred)
                    key = "tp" if p and r else "fp" if p else "fn" if r else "tn"
                    counts[model][f"{family}:{key}"] += 1
                    counts["ALL"][f"{family}:{key}"] += 1
                    if p != r and len(examples[f"{model}:{family}:{key}"]) < 5:
                        examples[f"{model}:{family}:{key}"].append({
                            "item_id": row.get("item_id") or row.get("qid"),
                            "path": str(path),
                            "reference": ref,
                            "prediction": pred,
                        })

    summary = {}
    for model, n in {**by_model, "ALL": total}.items():
        summary[model] = {"n": n, "families": {}}
        for family in FAMILIES:
            c = {k: counts[model][f"{family}:{k}"] for k in ("tp", "tn", "fp", "fn")}
            c["mismatch"] = c["fp"] + c["fn"]
            c["mismatch_rate"] = c["mismatch"] / n if n else None
            summary[model]["families"][family] = c

    out = {
        "status": "exploratory_lexical_screen_not_clinical_ground_truth",
        "input_paths": [str(x) for x in args.paths],
        "summary": summary,
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary["ALL"], indent=2))


if __name__ == "__main__":
    main()
