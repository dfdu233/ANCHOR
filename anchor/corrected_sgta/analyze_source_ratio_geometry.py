#!/usr/bin/env python3
"""Reconstruct source-ratio views and measure before/after center geometry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from corrected_sgta.infer_ce import (
    CENTER_ROOT,
    _center_distance,
    load_center,
    resize_image,
)
from corrected_sgta.methods import feddg_frequency_interpolation
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION, resolve_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, metavar="NAME=JSONL"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def mean(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


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
    raw_rows: list[dict] = []
    run_info = []

    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"invalid --run value: {spec}")
        name, path_text = spec.split("=", 1)
        path = Path(path_text)
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"{path}: protocol mismatch")
        fingerprint = metadata["fingerprint"]
        max_image_side = int(metadata["config"]["max_image_side"])
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if any(
            row.get("status") != "ok" or row.get("fingerprint") != fingerprint
            for row in records
        ):
            raise RuntimeError(f"{path}: unsuccessful or fingerprint-mismatched row")
        run_info.append(
            {
                "name": name,
                "cache": str(path),
                "fingerprint": fingerprint,
                "n_ok": len(records),
            }
        )

        for row in records:
            image_path = resolve_image(row.get("img_name", ""))
            if image_path is None:
                raise RuntimeError(f"missing image for qid={row['qid']}")
            with Image.open(image_path) as source:
                original = resize_image(source, max_image_side)
            logits = np.asarray(row["style_logits"], dtype=np.float64)
            base_prob = probabilities(logits[0])
            base_pred = int(np.argmax(base_prob))
            base_correct = base_pred == int(row["gt_index"])

            for index, item in enumerate(row["style_metadata"]):
                if item.get("family") != "feddg":
                    continue
                parameters = item["parameters"]
                source_ratio = float(parameters["source_ratio"])
                low_frequency_ratio = float(parameters["low_frequency_ratio"])
                center = load_center(str(CENTER_ROOT / item["center_file"]))
                transformed = feddg_frequency_interpolation(
                    original,
                    center,
                    low_frequency_ratio=low_frequency_ratio,
                    source_ratio=source_ratio,
                )
                before = _center_distance(original, center, low_frequency_ratio)
                after = _center_distance(transformed, center, low_frequency_ratio)
                prob = probabilities(logits[index])
                pred = int(np.argmax(prob))
                correct = pred == int(row["gt_index"])
                before_rrmse = float(before["log_amplitude_relative_rmse"])
                after_rrmse = float(after["log_amplitude_relative_rmse"])
                before_cos = float(before["log_amplitude_cosine_distance"])
                after_cos = float(after["log_amplitude_cosine_distance"])
                structure = item["structure"]
                raw_rows.append(
                    {
                        "run": name,
                        "qid": str(row["qid"]),
                        "domain_id": item["domain_id"],
                        "slot": f"matched_sr{source_ratio:g}",
                        "source_ratio": source_ratio,
                        "expected_linear_closure": 1.0 - source_ratio,
                        "rrmse_before": before_rrmse,
                        "rrmse_after": after_rrmse,
                        "rrmse_absolute_closure": before_rrmse - after_rrmse,
                        "rrmse_relative_closure": (before_rrmse - after_rrmse)
                        / max(before_rrmse, 1e-12),
                        "cosine_before": before_cos,
                        "cosine_after": after_cos,
                        "cosine_absolute_closure": before_cos - after_cos,
                        "cosine_relative_closure": (before_cos - after_cos)
                        / max(before_cos, 1e-12),
                        "pixel_mse": float(structure["pixel_mse"]),
                        "psnr": None
                        if structure["psnr"] is None
                        else float(structure["psnr"]),
                        "edge_correlation": float(structure["edge_correlation"]),
                        "base_correct": int(base_correct),
                        "style_correct": int(correct),
                        "rescue": int((not base_correct) and correct),
                        "harm": int(base_correct and (not correct)),
                        "changed": int(pred != base_pred),
                        "base_confidence": float(base_prob.max()),
                        "style_confidence": float(prob.max()),
                    }
                )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["run"], row["domain_id"], row["slot"])].append(row)
    summary_rows = []
    for (run, domain, slot), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "run": run,
                "domain_id": domain,
                "slot": slot,
                "n": len(rows),
                "mean_expected_linear_closure": mean(
                    [row["expected_linear_closure"] for row in rows]
                ),
                "mean_rrmse_before": mean([row["rrmse_before"] for row in rows]),
                "mean_rrmse_after": mean([row["rrmse_after"] for row in rows]),
                "mean_rrmse_relative_closure": mean(
                    [row["rrmse_relative_closure"] for row in rows]
                ),
                "fraction_rrmse_closer": mean(
                    [float(row["rrmse_after"] < row["rrmse_before"]) for row in rows]
                ),
                "mean_cosine_before": mean([row["cosine_before"] for row in rows]),
                "mean_cosine_after": mean([row["cosine_after"] for row in rows]),
                "mean_cosine_relative_closure": mean(
                    [row["cosine_relative_closure"] for row in rows]
                ),
                "fraction_cosine_closer": mean(
                    [float(row["cosine_after"] < row["cosine_before"]) for row in rows]
                ),
                "mean_psnr": mean([row["psnr"] for row in rows]),
                "mean_edge_correlation": mean(
                    [row["edge_correlation"] for row in rows]
                ),
                "changed_count": sum(row["changed"] for row in rows),
                "rescued_count": sum(row["rescue"] for row in rows),
                "harmed_count": sum(row["harm"] for row in rows),
            }
        )

    slot_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        slot_grouped[(row["run"], row["slot"])].append(row)
    run_slot_summary = []
    for (run, slot), rows in sorted(slot_grouped.items()):
        run_slot_summary.append(
            {
                "run": run,
                "slot": slot,
                "n": len(rows),
                "mean_rrmse_relative_closure": mean(
                    [row["rrmse_relative_closure"] for row in rows]
                ),
                "fraction_rrmse_closer": mean(
                    [float(row["rrmse_after"] < row["rrmse_before"]) for row in rows]
                ),
                "mean_psnr": mean([row["psnr"] for row in rows]),
                "changed_count": sum(row["changed"] for row in rows),
                "rescued_count": sum(row["rescue"] for row in rows),
                "harmed_count": sum(row["harm"] for row in rows),
            }
        )

    raw_path = args.output_dir / "geometry_raw_v1.tsv"
    domain_path = args.output_dir / "geometry_domain_summary_v1.tsv"
    run_slot_path = args.output_dir / "geometry_run_slot_summary_v1.tsv"
    write_tsv(raw_path, raw_rows)
    write_tsv(domain_path, summary_rows)
    write_tsv(run_slot_path, run_slot_summary)
    summary = {
        "version": "source-ratio-geometry-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Directly verify whether reconstructed source-ratio images move closer "
            "to their matched FFT-amplitude center while measuring structure loss."
        ),
        "distance_definition": (
            "Relative RMSE and cosine distance between log1p FFT amplitudes inside "
            "the transferred low-frequency window. RRMSE is primary because tiny "
            "windows can make cosine distance numerically degenerate."
        ),
        "runs": run_info,
        "run_slot_summary": run_slot_summary,
        "domain_summary": summary_rows,
        "files": {
            "raw": str(raw_path),
            "domain_summary": str(domain_path),
            "run_slot_summary": str(run_slot_path),
        },
    }
    output = args.output_dir / "geometry_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "run_slot_summary": run_slot_summary}, indent=2))


if __name__ == "__main__":
    main()
