"""Paired audit for Source-Nuisance Projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def predictions(row: dict, interface: str) -> list[int | None]:
    if interface == "decoded":
        return row["style_decoded_prediction"]
    if interface == "sequence_nll":
        return [int(np.argmin(values)) for values in row["style_sequence_nll"]]
    if interface == "surface":
        return [int(np.argmax(values)) for values in row["style_logits"]]
    raise ValueError(interface)


def metrics(rows: list[dict], interface: str) -> dict:
    pairs = [(row, predictions(row, interface)) for row in rows]
    base = np.asarray([pred[0] == row["gt_index"] for row, pred in pairs])
    method = np.asarray([pred[1] == row["gt_index"] for row, pred in pairs])
    rescue = int(np.sum(~base & method))
    harm = int(np.sum(base & ~method))
    discordant = rescue + harm
    return {
        "n": len(rows),
        "original_correct": int(base.sum()),
        "method_correct": int(method.sum()),
        "original_accuracy": float(base.mean()),
        "method_accuracy": float(method.mean()),
        "delta_pp": float(100.0 * (method.mean() - base.mean())),
        "prediction_changes": sum(
            pred[0] != pred[1] for _, pred in pairs
        ),
        "rescues": rescue,
        "harms": harm,
        "mcnemar_exact_p": (
            1.0
            if discordant == 0
            else float(binomtest(min(rescue, harm), discordant, 0.5).pvalue)
        ),
    }


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    ok = [row for row in rows if row.get("status") == "ok"]
    reports = {
        interface: metrics(ok, interface)
        for interface in ("decoded", "sequence_nll", "surface")
    }
    primary = reports["decoded"]
    gate = {
        "n_is_128": len(ok) == 128,
        "delta_at_least_3pp": primary["delta_pp"] >= 3.0,
        "rescues_not_less_than_harms": primary["rescues"] >= primary["harms"],
        "no_runtime_errors": len(ok) == len(rows),
    }
    payload = {
        "version": "source-nuisance-projection-analysis-v1",
        "input": str(args.input.resolve()),
        "n_rows": len(rows),
        "n_successful": len(ok),
        "metrics": reports,
        "expansion_gate": {
            "checks": gate,
            "pass": all(gate.values()),
            "decision": "advance" if all(gate.values()) else "stop_without_tuning",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

