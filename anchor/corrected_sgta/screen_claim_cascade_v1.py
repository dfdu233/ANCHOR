#!/usr/bin/env python3
"""CPU screen for adjacent false-positive clustering in generated reports.

Reference reports are used only as a benchmark proxy.  The permutation null
keeps every report's extracted claim multiset and error count fixed, and tests
whether false positive claims are more adjacent than expected from report-level
difficulty alone.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ALIASES = {
    "atelectasis": ("atelectasis",),
    "cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement"),
    "consolidation": ("consolidation",),
    "edema": ("pulmonary edema", "vascular congestion", "interstitial edema"),
    "fracture": ("fracture",),
    "lung lesion": ("lung lesion", "pulmonary nodule", "lung nodule", "mass lesion"),
    "lung opacity": ("lung opacity", "airspace opacity", "airspace opacities"),
    "pleural effusion": ("pleural effusion", "pleural effusions"),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax", "pneumothoraces"),
}
NEG = re.compile(r"\b(no|not|without|absent|absence of|negative for|free of|clear of)\b")


def positive_claims(text: str) -> list[str]:
    output = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.lower()):
        hits = []
        for finding, aliases in ALIASES.items():
            positions = [sentence.find(alias) for alias in aliases if alias in sentence]
            if not positions:
                continue
            pos = min(x for x in positions if x >= 0)
            if NEG.search(sentence[max(0, pos - 45):pos]):
                continue
            hits.append((pos, finding))
        output.extend(finding for _, finding in sorted(hits))
    return output


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def statistic(sequences: list[np.ndarray]) -> float:
    ff = ft = 0
    for seq in sequences:
        if len(seq) < 2:
            continue
        ff += int(np.sum(seq[:-1] & seq[1:]))
        ft += int(np.sum(seq[:-1]))
    return ff / ft if ft else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--permutations", type=int, default=5000)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    results = {}
    all_sequences = []
    for path in a.inputs:
        sequences = []
        n_claims = n_false = 0
        for row in load_rows(path):
            predicted = positive_claims(str(row.get("text", "")))
            reference = set(positive_claims(str(row.get("gt_ans", ""))))
            seq = np.asarray([claim not in reference for claim in predicted], dtype=bool)
            if len(seq) >= 2:
                sequences.append(seq)
                all_sequences.append(seq)
            n_claims += len(seq)
            n_false += int(seq.sum())
        observed = statistic(sequences)
        null = np.empty(a.permutations)
        for i in range(a.permutations):
            null[i] = statistic([rng.permutation(seq) for seq in sequences])
        results[str(path)] = {
            "n_reports_with_2plus_positive_claims": len(sequences),
            "n_positive_claims": n_claims,
            "proxy_false_positive_rate": n_false / n_claims if n_claims else None,
            "p_next_false_given_false": observed,
            "permuted_mean": float(np.nanmean(null)),
            "adjacency_excess": float(observed - np.nanmean(null)),
            "permutation_ci95": [float(np.nanquantile(null, .025)), float(np.nanquantile(null, .975))],
            "one_sided_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        }
    observed = statistic(all_sequences)
    null = np.empty(a.permutations)
    for i in range(a.permutations):
        null[i] = statistic([rng.permutation(seq) for seq in all_sequences])
    payload = {
        "version": "claim-cascade-screen-v1",
        "warning": "lexical reference matching is a benchmark proxy, not clinical hallucination truth",
        "per_input": results,
        "pooled": {
            "n_reports": len(all_sequences),
            "p_next_false_given_false": observed,
            "permuted_mean": float(np.nanmean(null)),
            "adjacency_excess": float(observed - np.nanmean(null)),
            "permutation_ci95": [float(np.nanquantile(null, .025)), float(np.nanquantile(null, .975))],
            "one_sided_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        },
        "decision": "GO" if observed > np.nanquantile(null, .975) and observed - np.nanmean(null) >= .10 else "NO_GO",
        "gate": "pooled adjacency excess >=0.10 and above 97.5% within-report permutation null",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
