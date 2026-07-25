"""Frozen Wave-A report with invalid decoded labels counted as errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corrected_sgta import freeze_alignment_report_v3 as implementation
from corrected_sgta.cache import iter_successes


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def majority(values):
    valid = [int(value) for value in values if value is not None]
    if not valid:
        return None
    counts = np.bincount(valid)
    return int(np.flatnonzero(counts == counts.max())[0])


def decoded_summary(predictions, labels) -> dict:
    valid = [value is not None for value in predictions]
    correct = [value is not None and int(value) == int(gt) for value, gt in zip(predictions, labels)]
    return {
        "n_total": len(labels),
        "n_parseable": int(sum(valid)),
        "parse_rate": float(np.mean(valid)),
        "accuracy_invalid_as_error": float(np.mean(correct)),
        "accuracy_parseable_only": None
        if not any(valid)
        else float(np.mean([ok for ok, keep in zip(correct, valid) if keep])),
    }


def main() -> None:
    implementation.main()
    output = Path(argument("--output"))
    cache = Path(argument("--cache"))
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    rows = list(iter_successes(cache, metadata["fingerprint"]))
    labels = [int(row["gt_index"]) for row in rows]
    originals = []
    consensus = []
    decode_executed = []
    disagreement = []
    for row in rows:
        roles = row["style_roles"]
        matched = [0] + [index for index, role in enumerate(roles) if role == "matched"]
        text = row.get("style_decoded_text")
        decoded = row.get("style_decoded_prediction")
        decode_executed.append(text is not None and len(text) == len(roles))
        if decoded is None:
            originals.append(None)
            consensus.append(None)
            continue
        originals.append(decoded[0])
        values = [decoded[index] for index in matched]
        consensus.append(majority(values))
        parseable = [value for value in values if value is not None]
        if parseable:
            disagreement.append(len(set(parseable)) > 1)
    report = json.loads(output.read_text())
    report["version"] = "sgta-alignment-frozen-report-release2-v1"
    report["evidence_channels"]["actual_decode_original"] = decoded_summary(originals, labels)
    report["evidence_channels"]["actual_decode_matched_majority"] = decoded_summary(consensus, labels)
    report["evidence_channels"]["actual_decode_cross_matched_disagreement_rate"] = (
        None if not disagreement else float(np.mean(disagreement))
    )
    report["gate"]["checks"]["actual_decode_present"] = bool(rows) and all(decode_executed)
    report["gate"]["pass"] = report["n"] == 256 and all(report["gate"]["checks"].values())
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
