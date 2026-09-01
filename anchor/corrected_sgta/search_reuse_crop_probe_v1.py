#!/usr/bin/env python3
"""Build and analyze an end-to-end selection--reuse inflation probe.

Build selects a claim and image region using frozen patch responses on VinDr
images that are reader-unanimous negative for every searched claim.  It emits
selected, same-area random, and full-image variants for ordinary VLM scoring.
Analyze asks whether selection, relative to the random-placebo crop, raises the
final decoder margin and false-positive rate increasingly with search size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil, sha256_file


VERSION = "selection-reuse-crop-probe-v1"
SEED = 20260812
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
CSV_COLUMNS = {
    "aortic_enlargement": "Aortic enlargement",
    "cardiomegaly": "Cardiomegaly",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule/Mass",
    "pleural_effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
}


def stable_seed(*values: object) -> int:
    return int(hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:16], 16)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def hidden_ids(directory: Path) -> set[str]:
    return {json.loads(line)["image_id"] for line in (directory / "metadata.jsonl").read_text().splitlines()}


def global_null_ids(path: Path) -> set[str]:
    votes: dict[str, dict[str, list[int]]] = {}
    readers: dict[str, set[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image = row["image_id"]
            votes.setdefault(image, {name: [] for name in FINDINGS})
            readers.setdefault(image, set()).add(row["rad_id"])
            for finding, column in CSV_COLUMNS.items():
                votes[image][finding].append(int(row[column]))
    return {
        image for image, values in votes.items()
        if len(readers[image]) == 3
        and all(len(values[name]) == 3 and sum(values[name]) == 0 for name in FINDINGS)
    }


def patch_artifact(directory: Path) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    rows = [json.loads(line) for line in (directory / "metadata.jsonl").read_text().splitlines()]
    scores = np.asarray(np.load(directory / "patch_scores.npz")["patch_scores"], dtype=float)
    index = {row["image_id"]: i for i, row in enumerate(rows)}
    geometry = {(int(row["grid_groups"]), int(row["grid_side"])) for row in rows}
    if len(rows) != scores.shape[0] or len(index) != len(rows) or len(geometry) != 1:
        raise ValueError("invalid patch artifact")
    groups, side = next(iter(geometry))
    if groups != 1:
        raise ValueError("v1 crop mapping requires one native visual grid")
    return scores, index, {"groups": groups, "side": side}


def position_null(ids: list[str], scores: np.ndarray, index: dict[str, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for column, finding in enumerate(FINDINGS):
        values = np.stack([scores[index[image], :, column] for image in ids])
        mean, std = values.mean(0), values.std(0)
        positive = std[std > 0]
        floor = float(np.quantile(positive, 0.1)) if len(positive) else 1.0
        output[finding] = (mean, np.maximum(std, floor))
    return output


def window_scores(values: np.ndarray, side: int, window: int) -> np.ndarray:
    grid = values.reshape(side, side)
    integral = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return sums / np.sqrt(window * window)


def crop_for_grid(image: Image.Image, row: int, col: int, side: int, window: int) -> Image.Image:
    width, height = image.size
    x0, x1 = int(round(col / side * width)), int(round((col + window) / side * width))
    y0, y1 = int(round(row / side * height)), int(round((row + window) / side * height))
    # Add half a selected-window of anatomical context on every side.
    pad_x, pad_y = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    target_width = min(width, (x1 - x0) + 2 * pad_x)
    target_height = min(height, (y1 - y0) + 2 * pad_y)
    left = max(0, min(x0 - pad_x, width - target_width))
    top = max(0, min(y0 - pad_y, height - target_height))
    box = (left, top, left + target_width, top + target_height)
    return image.crop(box)


def question(finding: str) -> str:
    return f"Does this chest X-ray show {finding.replace('_', ' ')}?"


def build(args: argparse.Namespace) -> None:
    scores, index, geometry = patch_artifact(args.patch_scores)
    receipt_path = args.output_dir / "receipt.json"
    expected_inputs = {
        "patch_scores_sha256": sha256_file(args.patch_scores / "patch_scores.npz"),
        "patch_metadata_sha256": sha256_file(args.patch_scores / "metadata.jsonl"),
        "reader_labels_sha256": sha256_file(args.reader_labels_csv),
    }
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            prior.get("version") != VERSION
            or prior.get("inputs") != expected_inputs
            or prior.get("claim_counts") != args.claim_counts
            or prior.get("region_counts") != args.region_counts
            or int(prior.get("window_side", -1)) != int(args.window_side)
        ):
            raise ValueError("completed build configuration drift")
        print(json.dumps({"status": "already_complete", "manifest_n": prior["manifest_n"]}))
        return
    null_ids = global_null_ids(args.reader_labels_csv)
    dev_ids = sorted(hidden_ids(args.development_hidden) & null_ids & index.keys())
    test_ids = sorted(hidden_ids(args.confirmation_hidden) & null_ids & index.keys())
    if len(dev_ids) < 100 or len(test_ids) < 50:
        raise ValueError(f"underpowered global null: dev={len(dev_ids)} test={len(test_ids)}")
    null = position_null(dev_ids, scores, index)
    side = int(geometry["side"])
    window = int(args.window_side)
    if not 1 <= window <= side:
        raise ValueError("window side outside visual grid")
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest, selections = [], []
    qid = 1
    for image_id in test_ids:
        source = args.dicom_root / f"{image_id}.dicom"
        image = dicom_to_pil(source)
        order = np.random.default_rng(stable_seed(SEED, image_id, "claim-order")).permutation(len(FINDINGS))
        position_count = (side - window + 1) ** 2
        if any(count > position_count for count in args.region_counts):
            raise ValueError(f"region count exceeds {position_count} possible windows")
        region_order = np.random.default_rng(stable_seed(SEED, image_id, "region-order")).permutation(position_count)
        configurations = [(1, count) for count in args.region_counts]
        configurations += [(count, position_count) for count in args.claim_counts if count != 1]
        for claim_count, region_count in configurations:
            candidates = []
            for column in order[:claim_count]:
                finding = FINDINGS[int(column)]
                mean, std = null[finding]
                z = (scores[index[image_id], :, column] - mean) / std
                surface = window_scores(z, side, window)
                allowed = region_order[:region_count]
                flat = int(allowed[int(np.argmax(surface.ravel()[allowed]))])
                row, col = np.unravel_index(flat, surface.shape)
                candidates.append((float(surface[row, col]), finding, int(row), int(col)))
            selected_score, finding, row, col = max(candidates)
            random_rng = np.random.default_rng(stable_seed(SEED, image_id, claim_count, "random-window"))
            random_row = int(random_rng.integers(0, side - window + 1))
            random_col = int(random_rng.integers(0, side - window + 1))
            if (random_row, random_col) == (row, col):
                random_col = (random_col + 1) % (side - window + 1)
            variants = {
                "selected": crop_for_grid(image, row, col, side, window),
                "random": crop_for_grid(image, random_row, random_col, side, window),
                "full": image,
            }
            group = f"{image_id}-k{claim_count}-r{region_count}"
            for variant, value in variants.items():
                relative = f"{group}-{variant}.png"
                value.save(image_dir / relative)
                manifest.append({
                    "qid": qid,
                    "img_name": relative,
                    "question": question(finding),
                    "answer": "no",
                })
                selections.append({
                    "qid": qid,
                    "group": group,
                    "image_id": image_id,
                    "claim_count": int(claim_count),
                    "region_count": int(region_count),
                    "search_size": int(claim_count * region_count),
                    "finding": finding,
                    "variant": variant,
                    "selected_internal_score": selected_score,
                    "selected_window": [row, col, window],
                    "random_window": [random_row, random_col, window],
                })
                qid += 1
    atomic_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "selections.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selections))
    atomic_json(args.output_dir / "receipt.json", {
        "version": VERSION,
        "status": "complete",
        "development_global_null_n": len(dev_ids),
        "confirmation_global_null_n": len(test_ids),
        "manifest_n": len(manifest),
        "claim_counts": args.claim_counts,
        "region_counts": args.region_counts,
        "window_side": window,
        "visual_grid_side": side,
        "truth": "all searched findings have exactly 0/3 reader votes",
        "inputs": expected_inputs,
        "command": " ".join(sys.argv),
        "source_sha256": sha256_file(Path(__file__)),
    })
    print(json.dumps({"status": "complete", "test_images": len(test_ids), "manifest_n": len(manifest)}))


def quantile_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[float]:
    samples = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(draws)]
    return np.quantile(samples, [0.025, 0.975]).tolist()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def analyze(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    selections = {int(row["qid"]): row for row in map(json.loads, args.selections.read_text().splitlines())}
    scored = {}
    for row in map(json.loads, args.raw.read_text().splitlines()):
        if row.get("status") == "ok":
            scored[int(row["question_id"])] = float(row["scores"]["original_margin"])
    if set(selections) != set(scored):
        raise ValueError(f"score coverage mismatch: selections={len(selections)} scores={len(scored)}")
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for qid, row in selections.items():
        key = (row["image_id"], int(row["claim_count"]), int(row["region_count"]))
        group = groups.setdefault(key, {**row, "margins": {}})
        group["margins"][row["variant"]] = scored[qid]
    configs = sorted({(key[1], key[2]) for key in groups}, key=lambda value: value[0] * value[1])
    rng = np.random.default_rng(args.seed)
    by_config = {}
    gap_by_image: dict[tuple[int, int], dict[str, float]] = {config: {} for config in configs}
    for k, r in configs:
        rows = [row for (image, claim_count, region_count), row in groups.items() if (claim_count, region_count) == (k, r)]
        selected = np.asarray([row["margins"]["selected"] for row in rows])
        random = np.asarray([row["margins"]["random"] for row in rows])
        full = np.asarray([row["margins"]["full"] for row in rows])
        internal = np.asarray([row["selected_internal_score"] for row in rows])
        gap = selected - random
        fp_gap = (selected > 0).astype(float) - (random > 0).astype(float)
        for row, value in zip(rows, gap):
            gap_by_image[(k, r)][row["image_id"]] = float(value)
        key = f"k{k}_r{r}"
        by_config[key] = {
            "n": len(rows),
            "selected_mean_margin": float(selected.mean()),
            "random_mean_margin": float(random.mean()),
            "full_mean_margin": float(full.mean()),
            "selected_minus_random_margin": float(gap.mean()),
            "selected_minus_random_ci95": quantile_interval(gap, rng, args.bootstrap_draws),
            "spearman_selector_vs_selected_margin": safe_spearman(internal, selected),
            "spearman_selector_vs_selected_random_gap": safe_spearman(internal, gap),
            "selected_fp_rate": float(np.mean(selected > 0)),
            "random_fp_rate": float(np.mean(random > 0)),
            "full_fp_rate": float(np.mean(full > 0)),
            "selected_minus_random_fp_rate": float(np.mean(selected > 0) - np.mean(random > 0)),
            "selected_minus_random_fp_rate_ci95": quantile_interval(fp_gap, rng, args.bootstrap_draws),
        }
    region_low = (1, min(r for k, r in configs if k == 1))
    region_high = (1, max(r for k, r in configs if k == 1))
    overall_high = max(configs, key=lambda value: value[0] * value[1])
    common = sorted(set(gap_by_image[region_low]) & set(gap_by_image[region_high]))
    growth = np.asarray([gap_by_image[region_high][image] - gap_by_image[region_low][image] for image in common])
    high_key = f"k{overall_high[0]}_r{overall_high[1]}"
    high_ci = by_config[high_key]["selected_minus_random_ci95"]
    growth_ci = quantile_interval(growth, rng, args.bootstrap_draws)
    high_fp_delta = by_config[high_key]["selected_minus_random_fp_rate"]
    high_fp_ci = by_config[high_key]["selected_minus_random_fp_rate_ci95"]
    result = {
        "version": VERSION,
        "status": "complete",
        "metrics_by_search_configuration": by_config,
        "primary": {
            "region_only_selected_random_gap_growth": float(growth.mean()),
            "region_only_gap_growth_ci95": growth_ci,
            "region_only_comparison": {"low": list(region_low), "high": list(region_high)},
            "largest_search_configuration": list(overall_high),
            "largest_search_selected_random_fp_delta": high_fp_delta,
            "largest_search_selected_random_fp_delta_ci95": high_fp_ci,
            "gate": bool(high_ci[0] > 0 and growth_ci[0] > 0 and high_fp_ci[0] > 0),
            "gate_rule": "largest-search selected-random margin CI>0; same-claim region-count gap-growth CI>0; largest-search selected-random FP-rate CI>0",
        },
        "configuration": {
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
            "raw_sha256": sha256_file(args.raw),
            "selections_sha256": sha256_file(args.selections),
            "source_sha256": sha256_file(Path(__file__)),
            "command": " ".join(sys.argv),
        },
        "boundary": "A pass shows an end-to-end selected-crop inflation mechanism on clear negatives; it is not itself a mitigation result.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result["primary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--patch-scores", type=Path, required=True)
    builder.add_argument("--development-hidden", type=Path, required=True)
    builder.add_argument("--confirmation-hidden", type=Path, required=True)
    builder.add_argument("--reader-labels-csv", type=Path, required=True)
    builder.add_argument("--dicom-root", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    builder.add_argument("--claim-counts", type=int, nargs="+", default=[1, 7])
    builder.add_argument("--region-counts", type=int, nargs="+", default=[16, 64, 361])
    builder.add_argument("--window-side", type=int, default=6)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--selections", type=Path, required=True)
    analysis.add_argument("--raw", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--bootstrap-draws", type=int, default=5000)
    analysis.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
