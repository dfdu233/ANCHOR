#!/usr/bin/env python3
"""Bind a blinded physician bundle to a label-free paired analysis contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.prepare_vqa_rad_internal_control_dev_v1 import atomic_json


VERSION = "anchor-physician-oe-multiarm-prereg-v1"


def build(
    *, template: Path, mapping: Path, delivery: Path, contract: Path,
    baseline: str, candidates: list[str], root: Path,
) -> dict:
    spec = json.loads(contract.read_text())
    if baseline in candidates or not baseline or len(candidates) != len(set(candidates)):
        raise ValueError("baseline/candidate identities are invalid")
    if spec.get("baseline") != baseline:
        raise ValueError("CLI baseline differs from the frozen clinical contract")
    if spec.get("candidate_arms") != candidates:
        raise ValueError("CLI candidates differ from the frozen clinical contract")
    result = {
        "protocol_version": VERSION,
        "frozen_before_physician_labels": True,
        "clinical_labels_inspected": False,
        "scope": "held-out physician claim-level T3 screen; not full efficacy evidence",
        "baseline": baseline,
        "candidate_methods": candidates,
        "unit_of_inference": spec["statistics"]["cluster_unit"],
        "bootstrap_iterations": int(spec["statistics"]["bootstrap_iterations"]),
        "bootstrap_seed": int(spec["statistics"]["bootstrap_seed"]),
        "primary_endpoint": spec["primary_endpoint"],
        "multiplicity": spec["statistics"]["multiplicity"],
        "promotion_gates": spec["no_exchange_gates"],
        "machine_gate_spec": spec["machine_gate_spec"],
        "analysis_module": "anchor.medeval.analyze_physician_oe_multiarm_v2",
        "all_gates_required": True,
        "automatic_metrics_define_clinical_truth": False,
        "provenance": {
            "review_template_sha256": sha256_file(template),
            "private_mapping_sha256": sha256_file(mapping),
            "delivery_manifest_sha256": sha256_file(delivery),
            "clinical_contract_sha256": sha256_file(contract),
            "prepare_adjudication_source_sha256": sha256_file(root / "anchor/medeval/prepare_physician_oe_adjudication.py"),
            "finalize_consensus_source_sha256": sha256_file(root / "anchor/medeval/finalize_physician_oe_consensus.py"),
            "analysis_source_sha256": sha256_file(root / "anchor/medeval/analyze_physician_oe_multiarm.py"),
            "analysis_wrapper_source_sha256": sha256_file(root / "anchor/medeval/analyze_physician_oe_multiarm_v2.py"),
        },
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        template=args.template, mapping=args.mapping, delivery=args.delivery,
        contract=args.contract, baseline=args.baseline,
        candidates=args.candidate, root=args.root.resolve(),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
