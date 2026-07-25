"""Aggregate a fixed-dose exact-source alignment probe with paired diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def exact_mcnemar_p(rescues: int, harmful: int) -> float:
    discordant = rescues + harmful
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * 0.5**discordant
        for k in range(min(rescues, harmful) + 1)
    )
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    base_correct: list[int],
    method_correct: list[int],
    *,
    seed: int = 42,
    repeats: int = 10_000,
) -> list[float]:
    rng = random.Random(seed)
    n = len(base_correct)
    deltas = []
    for _ in range(repeats):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(
            100.0
            * sum(method_correct[i] - base_correct[i] for i in indices)
            / n
        )
    deltas.sort()
    return [deltas[int(0.025 * repeats)], deltas[int(0.975 * repeats) - 1]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_eval.read_text())
    baseline_by_qid = {
        int(item["question_id"]): item["prediction"] for item in baseline["details"]
    }
    results = []
    for cache_path in sorted(args.probe_root.glob("l*/llava_cxr.jsonl")):
        rows = [
            json.loads(line)
            for line in cache_path.read_text().splitlines()
            if line.strip()
        ]
        if not rows or any(row.get("status") != "ok" for row in rows):
            raise RuntimeError(f"incomplete cache: {cache_path}")
        base_correct, method_correct, oracle_correct = [], [], []
        rescues, harmful, disagreements, flips = 0, 0, 0, []
        baseline_mismatches = []
        candidates, selected_candidates = [], []
        for row in rows:
            gt = int(row["gt_index"])
            predictions = row["style_decoded_prediction"]
            base_prediction = predictions[0]
            method_prediction = predictions[1] if len(predictions) > 1 else base_prediction
            base_ok = int(base_prediction == gt)
            method_ok = int(method_prediction == gt)
            base_correct.append(base_ok)
            method_correct.append(method_ok)
            oracle_correct.append(max(base_ok, method_ok))
            changed = base_prediction != method_prediction
            disagreements += int(changed)
            rescues += int(changed and not base_ok and method_ok)
            harmful += int(changed and base_ok and not method_ok)
            if changed:
                flips.append(
                    {
                        "qid": row["qid"],
                        "ground_truth_index": gt,
                        "original_prediction_index": base_prediction,
                        "aligned_prediction_index": method_prediction,
                        "rescue": bool(not base_ok and method_ok),
                        "harmful": bool(base_ok and not method_ok),
                    }
                )
            expected = baseline_by_qid.get(int(row["qid"]))
            observed = {0: "yes", 1: "no"}.get(base_prediction)
            if expected != observed:
                baseline_mismatches.append(
                    {"qid": row["qid"], "cached": expected, "probe": observed}
                )
            candidate = row["alignment_candidates"][0]
            candidates.append(candidate)
            if candidate["selected"]:
                selected_candidates.append(candidate)

        def candidate_summary(items: list[dict]) -> dict:
            structure_keys = (
                "psnr",
                "edge_correlation",
                "ssim",
                "central_local_contrast_correlation",
                "central_gradient_magnitude_ratio",
            )
            return {
                "n": len(items),
                "safe": sum(bool(item["safe"]) for item in items),
                "mean_visual_distance_before": mean(
                    [float(item["visual_distance_before"]) for item in items]
                ),
                "mean_visual_distance_after": mean(
                    [float(item["visual_distance_after"]) for item in items]
                ),
                "mean_relative_closure": mean(
                    [float(item["relative_closure"]) for item in items]
                ),
                "positive_closure": sum(
                    float(item["relative_closure"]) > 0 for item in items
                ),
                "mean_structure": {
                    key: mean(
                        [
                            float(item["structure"][key])
                            for item in items
                            if item["structure"].get(key) is not None
                        ]
                    )
                    for key in structure_keys
                },
            }

        n = len(rows)
        rescues_over_harm = rescues / harmful if harmful else None
        result = {
            "dose": float(candidates[0]["low_frequency_ratio"]),
            "cache": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "meta_sha256": sha256_file(
                cache_path.with_suffix(cache_path.suffix + ".meta.json")
            ),
            "fingerprint": rows[0]["fingerprint"],
            "n": n,
            "selected_views": len(selected_candidates),
            "fallbacks": n - len(selected_candidates),
            "baseline_accuracy": sum(base_correct) / n,
            "aligned_accuracy": sum(method_correct) / n,
            "accuracy_delta_pp": 100.0
            * (sum(method_correct) - sum(base_correct))
            / n,
            "style_oracle_accuracy": sum(oracle_correct) / n,
            "style_oracle_headroom_pp": 100.0
            * (sum(oracle_correct) - sum(base_correct))
            / n,
            "disagreements": disagreements,
            "rescues": rescues,
            "harmful_flips": harmful,
            "rescue_harm_ratio": rescues_over_harm,
            "mcnemar_exact_p": exact_mcnemar_p(rescues, harmful),
            "paired_bootstrap_delta_95ci_pp": paired_bootstrap_ci(
                base_correct, method_correct
            ),
            "baseline_cache_mismatches": baseline_mismatches,
            "candidate_summary_all": candidate_summary(candidates),
            "candidate_summary_selected": candidate_summary(selected_candidates),
            "flips": flips,
        }
        results.append(result)

    gate = {
        "scope": "pre-registered strong-signal gate for expansion from n=32 to n=128",
        "min_style_oracle_headroom_pp": 8.0,
        "min_rescues": 3,
        "min_rescue_harm_ratio": 2.0,
        "require_nonnegative_point_delta": True,
    }
    for result in results:
        result["passes_expansion_gate"] = (
            result["style_oracle_headroom_pp"]
            >= gate["min_style_oracle_headroom_pp"]
            and result["rescues"] >= gate["min_rescues"]
            and result["rescue_harm_ratio"] >= gate["min_rescue_harm_ratio"]
            and result["accuracy_delta_pp"] >= 0.0
        )
    output = {
        "analysis_version": "exact-source-fourier-probe-analysis-v1",
        "baseline_eval": str(args.baseline_eval.resolve()),
        "baseline_eval_sha256": sha256_file(args.baseline_eval),
        "baseline_reported_accuracy": baseline["accuracy_invalid_as_error"],
        "expansion_gate": gate,
        "results": results,
        "decision": (
            "proceed_n128"
            if any(item["passes_expansion_gate"] for item in results)
            else "stop_pixel_fourier_route"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
