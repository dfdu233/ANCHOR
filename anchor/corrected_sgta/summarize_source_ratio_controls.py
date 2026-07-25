#!/usr/bin/env python3
"""Summarize matched source-ratio controls from maintained CE caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from corrected_sgta.cache import decode_array
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=JSONL",
        help="Repeat for each cache to include.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--min-edge-correlation", type=float, default=0.90)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def margin(prob: np.ndarray) -> float:
    ordered = np.sort(prob)
    return float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - left @ right / denominator, 0.0, 2.0))


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    eps = 1e-12
    return float(
        0.5 * np.sum(left * np.log((left + eps) / (midpoint + eps)))
        + 0.5 * np.sum(right * np.log((right + eps) / (midpoint + eps)))
    )


def slot_name(metadata: dict, index: int) -> str:
    if index == 0:
        return "original"
    ratio = float(metadata.get("parameters", {}).get("source_ratio", np.nan))
    return f"matched_sr{ratio:g}"


def load_run(name: str, path: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"{path}: protocol mismatch")
    fingerprint = metadata["fingerprint"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    errors = [row for row in rows if row.get("status") != "ok"]
    records = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("fingerprint") == fingerprint
    ]
    if errors or len(records) != len(rows):
        raise RuntimeError(
            f"{path}: expected all rows to be successful and fingerprint-matched; "
            f"rows={len(rows)} records={len(records)} errors={len(errors)}"
        )

    outcomes: list[dict] = []
    for row in records:
        logits = np.asarray(row["style_logits"], dtype=np.float64)
        features = decode_array(row["style_features"]).astype(np.float64)
        style_metadata = row["style_metadata"]
        if len(logits) != len(features) or len(logits) != len(style_metadata):
            raise RuntimeError(f"{path}: style count mismatch at qid={row['qid']}")
        base_prob = probabilities(logits[0])
        base_pred = int(np.argmax(base_prob))
        base_correct = base_pred == int(row["gt_index"])
        for index, (style_logits, style_feature, item) in enumerate(
            zip(logits, features, style_metadata)
        ):
            prob = probabilities(style_logits)
            pred = int(np.argmax(prob))
            correct = pred == int(row["gt_index"])
            structure = item.get("structure") or {}
            psnr = structure.get("psnr")
            edge = structure.get("edge_correlation")
            unsafe = index > 0 and (
                psnr is None
                or float(psnr) < args.min_psnr
                or edge is None
                or float(edge) < args.min_edge_correlation
            )
            center_distance = item.get("center_distance") or {}
            outcomes.append(
                {
                    "run": name,
                    "qid": str(row["qid"]),
                    "domain_id": item.get("domain_id", "original"),
                    "slot": slot_name(item, index),
                    "source_ratio": item.get("parameters", {}).get("source_ratio"),
                    "gt_index": int(row["gt_index"]),
                    "base_pred": base_pred,
                    "style_pred": pred,
                    "base_correct": int(base_correct),
                    "style_correct": int(correct),
                    "rescue": int((not base_correct) and correct),
                    "harm": int(base_correct and (not correct)),
                    "changed": int(pred != base_pred),
                    "base_confidence": float(base_prob.max()),
                    "style_confidence": float(prob.max()),
                    "base_margin": margin(base_prob),
                    "style_margin": margin(prob),
                    "pixel_mse": float(structure.get("pixel_mse", 0.0)),
                    "psnr": None if psnr is None else float(psnr),
                    "edge_correlation": None if edge is None else float(edge),
                    "unsafe": int(unsafe),
                    "center_distance_cosine_before": center_distance.get(
                        "log_amplitude_cosine_distance"
                    ),
                    "center_distance_rrmse_before": center_distance.get(
                        "log_amplitude_relative_rmse"
                    ),
                    "feature_cosine_distance": cosine_distance(features[0], style_feature),
                    "logit_js_divergence": js_divergence(base_prob, prob),
                }
            )
    run_info = {
        "name": name,
        "cache": str(path),
        "cache_sha256": file_sha256(path),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "fingerprint": fingerprint,
        "n_ok": len(records),
        "config": metadata["config"],
    }
    return run_info, outcomes


def mean_or_none(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def paired_summary(
    rows: list[dict], args: argparse.Namespace, bootstrap_seed: int
) -> dict:
    n = len(rows)
    baseline = np.asarray([row["base_correct"] for row in rows], dtype=np.float64)
    styled = np.asarray([row["style_correct"] for row in rows], dtype=np.float64)
    paired_delta = styled - baseline
    rescue = int(sum(row["rescue"] for row in rows))
    harm = int(sum(row["harm"] for row in rows))
    rng = np.random.default_rng(bootstrap_seed)
    if n and args.bootstrap:
        indices = rng.integers(0, n, size=(args.bootstrap, n))
        distribution = paired_delta[indices].mean(axis=1)
        ci025, ci500, ci975 = np.quantile(distribution, [0.025, 0.5, 0.975])
        p_gt_zero = float(np.mean(distribution > 0))
    else:
        ci025 = ci500 = ci975 = p_gt_zero = None
    discordant = rescue + harm
    return {
        "n": n,
        "baseline_accuracy": float(baseline.mean()) if n else None,
        "accuracy": float(styled.mean()) if n else None,
        "delta_vs_baseline": float(paired_delta.mean()) if n else None,
        "rescue_rate": rescue / n if n else None,
        "harm_rate": harm / n if n else None,
        "changed_rate": float(np.mean([row["changed"] for row in rows])) if n else None,
        "rescued_count": rescue,
        "harmed_count": harm,
        "changed_count": int(sum(row["changed"] for row in rows)),
        "mcnemar_exact_p": float(binomtest(rescue, discordant, 0.5).pvalue)
        if discordant
        else 1.0,
        "bootstrap_ci025": None if ci025 is None else float(ci025),
        "bootstrap_ci500": None if ci500 is None else float(ci500),
        "bootstrap_ci975": None if ci975 is None else float(ci975),
        "bootstrap_p_gt_zero": p_gt_zero,
        "unsafe_rate": float(np.mean([row["unsafe"] for row in rows])) if n else None,
        "mean_psnr": mean_or_none([row["psnr"] for row in rows]),
        "mean_edge_correlation": mean_or_none(
            [row["edge_correlation"] for row in rows]
        ),
        "mean_feature_cosine_distance": mean_or_none(
            [row["feature_cosine_distance"] for row in rows]
        ),
        "mean_logit_js_divergence": mean_or_none(
            [row["logit_js_divergence"] for row in rows]
        ),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"invalid --run value: {spec}")
        name, path = spec.split("=", 1)
        run_specs.append((name, Path(path)))

    runs = []
    outcomes = []
    for name, path in run_specs:
        run_info, run_outcomes = load_run(name, path, args)
        runs.append(run_info)
        outcomes.extend(run_outcomes)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    domain_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in outcomes:
        grouped[(row["run"], row["slot"])].append(row)
        domain_grouped[(row["run"], row["domain_id"], row["slot"])].append(row)

    slot_rows = []
    for index, ((run, slot), rows) in enumerate(sorted(grouped.items())):
        slot_rows.append(
            {
                "run": run,
                "slot": slot,
                **paired_summary(rows, args, args.seed + index),
            }
        )
    domain_rows = []
    for index, ((run, domain, slot), rows) in enumerate(sorted(domain_grouped.items())):
        domain_rows.append(
            {
                "run": run,
                "domain_id": domain,
                "slot": slot,
                **paired_summary(rows, args, args.seed + 10_000 + index),
            }
        )

    transformed_slots = sorted(
        {row["slot"] for row in outcomes if row["slot"] != "original"}
    )
    aggregate = []
    for slot in transformed_slots:
        cells = [row for row in slot_rows if row["slot"] == slot]
        aggregate.append(
            {
                "slot": slot,
                "n_tasks": len(cells),
                "macro_delta": float(np.mean([row["delta_vs_baseline"] for row in cells])),
                "weighted_delta": float(
                    np.average(
                        [row["delta_vs_baseline"] for row in cells],
                        weights=[row["n"] for row in cells],
                    )
                ),
                "positive_tasks": sum(row["delta_vs_baseline"] > 0 for row in cells),
                "negative_tasks": sum(row["delta_vs_baseline"] < 0 for row in cells),
                "unchanged_tasks": sum(row["delta_vs_baseline"] == 0 for row in cells),
                "total_rescued": sum(row["rescued_count"] for row in cells),
                "total_harmed": sum(row["harmed_count"] for row in cells),
                "total_changed": sum(row["changed_count"] for row in cells),
            }
        )

    write_tsv(args.output_dir / "sample_outcomes_long_v1.tsv", outcomes)
    write_tsv(args.output_dir / "slot_summary_v1.tsv", slot_rows)
    write_tsv(args.output_dir / "domain_slot_summary_v1.tsv", domain_rows)
    summary = {
        "version": "source-ratio-control-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Cross-model visual-independent controls for matched source-ratio "
            "transfer, including paired accuracy, rescue/harm, structure, feature, "
            "and logit-shift diagnostics."
        ),
        "settings": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "min_psnr": args.min_psnr,
            "min_edge_correlation": args.min_edge_correlation,
        },
        "runs": runs,
        "slot_summary": slot_rows,
        "domain_slot_summary": domain_rows,
        "aggregate": aggregate,
        "measurement_note": (
            "center_distance_*_before is the original-to-center distance cached for "
            "each transform. It is not an after-transfer distance and must not be used "
            "as direct evidence that the transformed image became closer."
        ),
        "files": {
            "sample_outcomes": str(args.output_dir / "sample_outcomes_long_v1.tsv"),
            "slot_summary": str(args.output_dir / "slot_summary_v1.tsv"),
            "domain_slot_summary": str(args.output_dir / "domain_slot_summary_v1.tsv"),
        },
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
