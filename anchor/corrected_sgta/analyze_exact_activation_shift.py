"""Audit a fixed-beta exact-source activation-shift CE pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions(row: dict, interface: str) -> list[int | None]:
    if interface == "decoded":
        return row["style_decoded_prediction"]
    if interface == "sequence_nll":
        return [int(np.argmin(values)) for values in row["style_sequence_nll"]]
    if interface == "surface":
        return [int(np.argmax(values)) for values in row["style_logits"]]
    raise ValueError(interface)


def interface_metrics(rows: list[dict], interface: str) -> dict:
    paired = [(row, predictions(row, interface)) for row in rows]
    n = len(paired)
    parsed = [
        sum(pred[j] is not None and pred[j] >= 0 for _, pred in paired)
        for j in (0, 1)
    ]
    correct = [
        sum(pred[j] == row["gt_index"] for row, pred in paired)
        for j in (0, 1)
    ]
    rescue = sum(
        pred[0] != row["gt_index"] and pred[1] == row["gt_index"]
        for row, pred in paired
    )
    harmful = sum(
        pred[0] == row["gt_index"] and pred[1] != row["gt_index"]
        for row, pred in paired
    )
    oracle = sum(
        pred[0] == row["gt_index"] or pred[1] == row["gt_index"]
        for row, pred in paired
    )
    return {
        "n": n,
        "parsed": parsed,
        "parse_rate": [value / n for value in parsed],
        "correct": correct,
        "accuracy": [value / n for value in correct],
        "point_delta_pp": 100.0 * (correct[1] - correct[0]) / n,
        "disagreements": sum(pred[0] != pred[1] for _, pred in paired),
        "rescues": rescue,
        "harmful_flips": harmful,
        "rescue_to_harm_ratio": None if harmful == 0 else rescue / harmful,
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / n,
        "oracle_headroom_pp": 100.0 * (oracle - correct[0]) / n,
    }


def main() -> None:
    args = parse_args()
    meta_path = args.input.with_suffix(args.input.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    raw_rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in raw_rows if row.get("status") == "ok"]
    fingerprints = {row.get("fingerprint") for row in rows}
    qids = [str(row["qid"]) for row in rows]
    closures = [
        float(row["alignment_candidates"][0]["relative_closure"]) for row in rows
    ]
    invariants = {
        "all_rows_successful": len(rows) == len(raw_rows),
        "unique_qids": len(qids) == len(set(qids)),
        "single_matching_fingerprint": fingerprints == {metadata["fingerprint"]},
        "beta_is_frozen_at_one": metadata["config"]["beta"] == 1.0,
        "pixels_identical": all(
            row["alignment_candidates"][0]["structure"]["pixel_identity"]
            for row in rows
        ),
        "token_residuals_declared_invariant": all(
            row["alignment_candidates"][0]["structure"]["token_residual_identity"]
            for row in rows
        ),
        "projected_mean_closure_at_least_99_999_percent": all(
            value >= 0.99999 for value in closures
        ),
    }
    metrics = {
        name: interface_metrics(rows, name)
        for name in ("decoded", "sequence_nll", "surface")
    }
    primary = metrics["decoded"]
    gate_checks = {
        "at_least_three_rescues": primary["rescues"] >= 3,
        "oracle_headroom_at_least_8pp": primary["oracle_headroom_pp"] >= 8.0,
        "rescue_to_harm_ratio_at_least_two": (
            primary["rescues"] > 0
            if primary["harmful_flips"] == 0
            else primary["rescue_to_harm_ratio"] >= 2.0
        ),
        "nonnegative_point_delta": primary["point_delta_pp"] >= 0.0,
    }
    changed_cases = []
    for row in rows:
        pred = predictions(row, "decoded")
        if pred[0] != pred[1]:
            changed_cases.append(
                {
                    "qid": row["qid"],
                    "gt_index": row["gt_index"],
                    "labels": row["labels"],
                    "decoded_prediction": pred,
                    "decoded_text": row["style_decoded_text"],
                    "sequence_nll_prediction": predictions(row, "sequence_nll"),
                    "surface_prediction": predictions(row, "surface"),
                }
            )
    payload = {
        "analysis_version": "exact-source-activation-shift-audit-v1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "metadata": str(meta_path.resolve()),
        "metadata_sha256": sha256_file(meta_path),
        "fingerprint": metadata["fingerprint"],
        "n_rows": len(raw_rows),
        "n_successful": len(rows),
        "invariants": invariants,
        "invariants_pass": all(invariants.values()),
        "relative_closure": {
            "min": min(closures),
            "mean": float(np.mean(closures)),
            "max": max(closures),
        },
        "metrics": metrics,
        "expansion_gate": {
            "checks": gate_checks,
            "pass": all(gate_checks.values()),
            "decision": (
                "expand_to_n128"
                if all(gate_checks.values())
                else "stop_without_tuning_or_expansion"
            ),
        },
        "changed_cases": changed_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
