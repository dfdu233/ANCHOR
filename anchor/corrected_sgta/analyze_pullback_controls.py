"""Paired finite-difference audit of decoder-visible source alignment.

For a question q and pooled visual representation z, the local categorical
Fisher pullback energy is

    delta.T @ J_q.T @ F_q @ J_q @ delta ~= 2 KL(p(z) || p(z + delta)).

The cache provides finite source interventions, so this analyzer reports JS and
symmetric KL as stable finite-difference proxies.  It never interprets feature
closure alone as evidence of correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from corrected_sgta.cache import decode_array


VERSION = "sgta-paired-pullback-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomization-repeats", type=int, default=100_000)
    parser.add_argument("--headroom-gate-pp", type=float, default=5.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max()
    values = np.exp(values)
    return values / np.clip(values.sum(), 1e-12, None)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js(p: np.ndarray, q: np.ndarray) -> float:
    middle = 0.5 * (p + q)
    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def paired_randomization_pvalue(
    matched: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> float:
    """One-sided paired sign-flip test for E[matched-control] > 0."""
    differences = np.asarray(matched, dtype=np.float64) - np.asarray(
        control, dtype=np.float64
    )
    observed = float(differences.mean())
    if not np.any(differences):
        return 1.0
    rng = random.Random(seed)
    exceed = 0
    for _ in range(repeats):
        statistic = float(
            np.mean(
                [
                    value if rng.random() < 0.5 else -value
                    for value in differences
                ]
            )
        )
        exceed += statistic >= observed
    return float((exceed + 1) / (repeats + 1))


def margin(scores: np.ndarray, target: int) -> float:
    other = np.delete(np.asarray(scores, dtype=np.float64), target)
    return float(scores[target] - other.max())


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def analyze_rows(path: Path, *, seed: int, repeats: int, headroom_gate: float) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        raise RuntimeError(f"no valid rows: {path}")

    metrics = {
        role: {
            "twice_js": [],
            "symmetric_kl": [],
            "visual_delta_sq": [],
            "fisher_efficiency": [],
            "true_margin_delta": [],
        }
        for role in ("matched", "wrong_control")
    }
    correct = {"original": 0, "matched": 0, "wrong_control": 0}
    oracle = 0
    disagreements = {"matched": 0, "wrong_control": 0}
    rescues = {"matched": 0, "wrong_control": 0}
    harms = {"matched": 0, "wrong_control": 0}
    fingerprints = set()

    for row in rows:
        if row["style_roles"] != ["original", "matched", "wrong_control"]:
            raise RuntimeError(f"invalid three-arm roles for qid={row.get('qid')}")
        fingerprints.add(row["fingerprint"])
        visual = decode_array(row["style_visual_features"]).astype(np.float64)
        sequence_scores = -np.asarray(row["style_sequence_nll"], dtype=np.float64)
        probabilities = [softmax(values) for values in sequence_scores]
        predictions = np.argmax(sequence_scores, axis=1)
        target = int(row["gt_index"])
        original_correct = int(predictions[0] == target)
        correct["original"] += original_correct

        for index, role in ((1, "matched"), (2, "wrong_control")):
            delta_sq = float(np.sum((visual[index] - visual[0]) ** 2))
            twice_js = 2.0 * js(probabilities[0], probabilities[index])
            symmetric_kl = 0.5 * (
                kl(probabilities[0], probabilities[index])
                + kl(probabilities[index], probabilities[0])
            )
            metrics[role]["twice_js"].append(twice_js)
            metrics[role]["symmetric_kl"].append(symmetric_kl)
            metrics[role]["visual_delta_sq"].append(delta_sq)
            metrics[role]["fisher_efficiency"].append(
                twice_js / max(delta_sq, 1e-12)
            )
            metrics[role]["true_margin_delta"].append(
                margin(sequence_scores[index], target)
                - margin(sequence_scores[0], target)
            )
            arm_correct = int(predictions[index] == target)
            correct[role] += arm_correct
            disagreements[role] += int(predictions[index] != predictions[0])
            rescues[role] += int(not original_correct and arm_correct)
            harms[role] += int(original_correct and not arm_correct)
        oracle += int(
            predictions[0] == target or predictions[1] == target
        )

    if len(fingerprints) != 1:
        raise RuntimeError(f"mixed fingerprints in {path}")
    arrays = {
        role: {name: np.asarray(values) for name, values in payload.items()}
        for role, payload in metrics.items()
    }
    comparisons = {}
    for offset, name in enumerate(metrics["matched"]):
        comparisons[name] = {
            "matched_minus_wrong_mean": float(
                arrays["matched"][name].mean()
                - arrays["wrong_control"][name].mean()
            ),
            "paired_randomization_p_greater": paired_randomization_pvalue(
                arrays["matched"][name],
                arrays["wrong_control"][name],
                seed=seed + offset,
                repeats=repeats,
            ),
        }

    n = len(rows)
    headroom_pp = 100.0 * (oracle - correct["original"]) / n
    checks = {
        "matched_pullback_energy_gt_wrong_p_lt_0.05": (
            comparisons["twice_js"]["matched_minus_wrong_mean"] > 0.0
            and comparisons["twice_js"]["paired_randomization_p_greater"] < 0.05
        ),
        "matched_oracle_headroom_at_least_gate": headroom_pp >= headroom_gate,
        "matched_rescues_not_less_than_harms": (
            rescues["matched"] >= harms["matched"]
        ),
    }
    return {
        "path": str(path.resolve()),
        "rows_sha256": sha256_file(path),
        "fingerprint": next(iter(fingerprints)),
        "n": n,
        "metrics": {
            role: {name: describe(values) for name, values in payload.items()}
            for role, payload in arrays.items()
        },
        "comparisons": comparisons,
        "accuracy": {
            role: {"correct": value, "accuracy": value / n}
            for role, value in correct.items()
        },
        "matched_oracle_headroom_pp": headroom_pp,
        "disagreements": disagreements,
        "rescues": rescues,
        "harms": harms,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    for path in args.rows:
        if not path.is_file():
            raise FileNotFoundError(path)
    runs = [
        analyze_rows(
            path,
            seed=args.seed + 100 * index,
            repeats=args.randomization_repeats,
            headroom_gate=args.headroom_gate_pp,
        )
        for index, path in enumerate(args.rows)
    ]
    config = {
        "version": VERSION,
        "seed": args.seed,
        "randomization_repeats": args.randomization_repeats,
        "headroom_gate_pp": args.headroom_gate_pp,
        "inputs": {str(path): sha256_file(path) for path in args.rows},
        "primary_metric": "2*JS divided by squared pooled visual displacement",
        "interpretation": "finite-difference pullback proxy, not exact infinitesimal Fisher",
    }
    payload = {
        "analysis_version": VERSION,
        "fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "config": config,
        "runs": runs,
        "aggregate": {
            "n_runs": len(runs),
            "n_rows": sum(run["n"] for run in runs),
            "runs_passing": sum(run["pass"] for run in runs),
            "runs_with_pullback_specificity": sum(
                run["checks"]["matched_pullback_energy_gt_wrong_p_lt_0.05"]
                for run in runs
            ),
            "runs_with_headroom": sum(
                run["checks"]["matched_oracle_headroom_at_least_gate"]
                for run in runs
            ),
            "decision": (
                "consider_question_conditioned_alignment_pilot"
                if any(run["pass"] for run in runs)
                else "stop_current_source_paths"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "fingerprint": payload["fingerprint"],
                "aggregate": payload["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
