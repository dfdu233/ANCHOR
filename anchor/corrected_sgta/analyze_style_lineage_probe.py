"""Compare style-conditioned answer dispersion in base and medical VLMs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


VERSION = "style-lineage-analysis-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dispersion(rows: list[dict]) -> tuple[float, dict]:
    diseases = sorted({row["disease"] for row in rows})
    clusters = sorted({int(row["cluster"]) for row in rows})
    details = {}
    ranges = []
    for disease in diseases:
        rates = []
        for cluster in clusters:
            subset = [
                row
                for row in rows
                if row["disease"] == disease and int(row["cluster"]) == cluster
            ]
            values = [
                1.0 if row["explicit_prediction"] == "yes" else 0.0
                for row in subset
                if row["explicit_prediction"] in {"yes", "no"}
            ]
            rates.append(float(np.mean(values)) if values else float("nan"))
        finite = np.asarray(rates)[np.isfinite(rates)]
        value = float(finite.max() - finite.min()) if len(finite) else float("nan")
        ranges.append(value)
        details[disease] = {"cluster_yes_rate": rates, "range": value}
    return float(np.nanmean(ranges)), details


def paired_permutation(
    base_rows: list[dict],
    medical_rows: list[dict],
    draws: int = 10000,
) -> tuple[float, float]:
    base = {row["id"]: row for row in base_rows}
    medical = {row["id"]: row for row in medical_rows}
    common = sorted(set(base) & set(medical))
    observed = dispersion([medical[key] for key in common])[0] - dispersion(
        [base[key] for key in common]
    )[0]
    rng = np.random.default_rng(2027)
    null = []
    for _ in range(draws):
        first, second = [], []
        for key in common:
            if rng.random() < 0.5:
                first.append(base[key])
                second.append(medical[key])
            else:
                first.append(medical[key])
                second.append(base[key])
        null.append(dispersion(second)[0] - dispersion(first)[0])
    p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (draws + 1))
    return observed, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path)
    parser.add_argument("--medical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    medical_rows = read_jsonl(args.medical)
    medical_dispersion, medical_details = dispersion(medical_rows)
    medical_parse = np.mean(
        [row["explicit_prediction"] in {"yes", "no"} for row in medical_rows]
    )
    diseases_with_medical_flips = sum(
        value["range"] > 0 for value in medical_details.values()
    )
    medical_switch_observed = bool(
        medical_dispersion >= 0.10
        and diseases_with_medical_flips >= 2
        and medical_parse >= 0.90
    )
    base_rows = read_jsonl(args.base) if args.base else []
    if base_rows:
        base_dispersion, base_details = dispersion(base_rows)
        difference, p_value = paired_permutation(base_rows, medical_rows)
        base_parse = np.mean(
            [row["explicit_prediction"] in {"yes", "no"} for row in base_rows]
        )
        lineage_gate = bool(
            difference >= 0.10
            and p_value < 0.05
            and medical_switch_observed
            and base_parse >= 0.90
        )
    else:
        base_dispersion = None
        base_details = None
        difference = None
        p_value = None
        base_parse = None
        lineage_gate = False
    inputs = {
        "medical": str(args.medical.resolve()),
        "medical_sha256": sha256(args.medical),
    }
    if args.base:
        inputs.update(
            {
                "base": str(args.base.resolve()),
                "base_sha256": sha256(args.base),
            }
        )
    result = {
        "version": VERSION,
        "inputs": inputs,
        "n_base": len(base_rows),
        "n_medical": len(medical_rows),
        "base_parse_rate": (
            float(base_parse) if base_parse is not None else None
        ),
        "medical_parse_rate": float(medical_parse),
        "base_mean_cluster_range": base_dispersion,
        "medical_mean_cluster_range": medical_dispersion,
        "medical_minus_base_dispersion": difference,
        "paired_permutation_p_one_sided": p_value,
        "base_by_disease": base_details,
        "medical_by_disease": medical_details,
        "decision": {
            "medical_style_switch_observed": medical_switch_observed,
            "criterion": (
                "medical-base mean cluster range >=0.10, p<.05, "
                "at least two medical disease flips, both parse rates >=.90"
            ),
            "diseases_with_medical_style_flips": diseases_with_medical_flips,
            "gate_passed": lineage_gate,
            "status": (
                "lineage_comparison_complete"
                if base_rows
                else "base_control_missing"
            ),
        },
        "claim_ceiling": (
            "training-lineage-specific style prior on synthetic shared-content "
            "source prototypes; not target-domain accuracy"
        ),
    }
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
