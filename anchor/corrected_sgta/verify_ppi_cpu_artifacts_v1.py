#!/usr/bin/env python3
"""Independent fail-closed verifier for PPI source assignment and power artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


VERSION = "ppi-cpu-artifact-verifier-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-dir", type=Path, default=Path("corrected_runs/ppi_source_assignment_v1"))
    parser.add_argument("--power-dir", type=Path, default=Path("corrected_runs/ppi_mechanism_power_v1"))
    parser.add_argument("--output", type=Path, default=Path("corrected_runs/ppi_cpu_verification_v1/verification.json"))
    args = parser.parse_args()

    audit_path = args.assignment_dir / "audit.json"
    assignments_path = args.assignment_dir / "assignments.jsonl"
    power_path = args.power_dir / "power_audit.json"
    audit = json.loads(audit_path.read_text())
    power = json.loads(power_path.read_text())
    require(sha256(assignments_path) == audit["assignments_sha256"], "assignment hash mismatch")
    require(audit["unit_count"] == 772, "unexpected source unit count")
    require(audit["eligible_claims"] == ["consolidation", "pleural_effusion"], "claim set drift")
    require(audit["decision"] == "CPU_FEASIBLE_ONLY", "assignment decision drift")
    for flag in ("gpu_authorized", "human_extractor_admitted", "vin_dr_consumed", "model_consumed", "gpu_consumed"):
        require(audit[flag] is False, f"unsafe provenance flag: {flag}")
    require(len(audit["runs"]) == 18, "expected 3 seeds x 2 experiments x 3 arms")
    for run in audit["runs"]:
        metrics = run["metrics"]
        require(metrics["combo_counts"] == {"0": 193, "1": 193, "2": 193, "3": 193}, "unbalanced cells")
        words = metrics["word_mass_by_combo"]
        mean_words = sum(words) / 4
        require(max(abs(value - mean_words) for value in words) <= 0.011 * mean_words, "word-mass imbalance")
        claims = metrics["claims"]
        if run["arm"] == "plus":
            pairing = run["solver"]["pairing"]
            signals = []
            for index, claim in enumerate(audit["eligible_claims"]):
                signals.extend(
                    [
                        pairing["r_mu"][index] * claims[claim]["u_mu"],
                        pairing["r_kappa"][index] * claims[claim]["v_kappa"],
                    ]
                )
                require(abs(claims[claim]["v_mu"]) <= 0.05 + 1e-9, "mean-to-precision leakage")
                require(abs(claims[claim]["u_kappa"]) <= 0.05 + 1e-9, "precision-to-mean leakage")
            require(min(signals) >= 0.20, "weak target assignment contrast")
        elif run["arm"] == "zero":
            require(max(abs(value) for row in claims.values() for value in row.values()) <= 0.05, "zero arm not null")

    indexed = defaultdict(dict)
    with assignments_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["seed"], row["experiment"], row["response_unit_id"])
            require(row["arm"] not in indexed[key], "duplicate assignment row")
            indexed[key][row["arm"]] = row
    require(len(indexed) == 3 * 2 * 772, "assignment closure mismatch")
    for arms in indexed.values():
        require(set(arms) == {"plus", "minus", "zero"}, "arm closure mismatch")
        require(arms["plus"]["combo"] + arms["minus"]["combo"] == 3, "minus is not exact complement")
        require(arms["plus"]["response_sha256"] == arms["minus"]["response_sha256"], "text drift across arms")

    require(power["assignment_audit_sha256"] == sha256(audit_path), "power input binding mismatch")
    require(power["decision"] == "POWER_GATE_PASS", "power gate failed")
    require(power["gpu_authorized"] is False, "power artifact authorized GPU")
    planned = [row for row in power["rows"] if row["n_per_claim_reader_bucket_per_seed"] == 100]
    gated = next(row for row in planned if row["mechanism"] == "evidence_gated")
    artifacts = [row for row in planned if row["mechanism"] != "evidence_gated"]
    require(gated["mechanism_admission_rate"] >= 0.80, "planned power below 0.80")
    require(all(row["mechanism_admission_rate"] <= 0.05 for row in artifacts), "artifact false admission above 0.05")

    result = {
        "version": VERSION,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "decision": "VERIFIED_CPU_ONLY",
        "gpu_authorized": False,
        "assignment_audit_sha256": sha256(audit_path),
        "assignments_sha256": sha256(assignments_path),
        "power_audit_sha256": sha256(power_path),
        "verified_assignment_rows": 3 * len(indexed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
