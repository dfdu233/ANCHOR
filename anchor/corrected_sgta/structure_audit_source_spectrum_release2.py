"""Complete independent reconstruction audit for source-spectrum release 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from corrected_sgta import structure_audit_v2 as implementation
from corrected_sgta.frequency_alignment_source_spectrum_release2 import source_spectrum_alignment_release2
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts
from corrected_sgta.structure_audit_wave_a import structure_proxy as base_structure_proxy


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def complete_structure_proxy(left, right):
    payload = base_structure_proxy(left, right)
    a0 = np.asarray(left, dtype=np.float64) / 255.0
    b0 = np.asarray(right, dtype=np.float64) / 255.0
    a = np.hypot(*np.gradient(a0)); b = np.hypot(*np.gradient(b0))
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    payload["edge_correlation"] = float(np.clip(a @ b / denominator, -1.0, 1.0)) if denominator > 1e-12 else 1.0
    return payload


def main() -> None:
    cache = Path(argument("--cache")); source_bank = Path(argument("--source-bank"))
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    expected = metadata["config"]["source_bank_sha256"]; actual = sha256_file(source_bank)
    if actual != expected:
        raise RuntimeError(f"structure audit/source-bank mismatch: {actual} != {expected}")
    verify_source_artifacts(load_manifest(source_bank))
    implementation.feddg_frequency_interpolation_v2 = source_spectrum_alignment_release2
    implementation.structure_proxy = complete_structure_proxy
    implementation.main()
    output = Path(argument("--output")); report = json.loads(output.read_text())
    for record in report["records"]:
        record["pass"] = bool(record["pass"] and record["edge_correlation"] >= 0.90)
    for role, key in (("matched", "matched"), ("wrong_control", "wrong_control")):
        values = [x for x in report["records"] if x["role"] == role]
        report[key]["pass_rate"] = None if not values else float(np.mean([x["pass"] for x in values]))
        report[key]["edge_correlation_median"] = None if not values else float(np.median([x["edge_correlation"] for x in values]))
    matched = [x for x in report["records"] if x["role"] == "matched"]
    report["thresholds"]["edge_correlation_min"] = 0.90
    report["formal_matched_structure_pass"] = bool(matched) and all(x["pass"] for x in matched)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(output)


if __name__ == "__main__":
    main()

