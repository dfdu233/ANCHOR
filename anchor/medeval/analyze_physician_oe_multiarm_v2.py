#!/usr/bin/env python3
"""T3-scoped wrapper binding the frozen physician analysis to exact gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_physician_oe_multiarm import (
    _validate_consensus_provenance,
    analyze_multiarm,
)
from .hashing import sha256_file
from .store import atomic_write_json
from .validate_physician_oe_review import load_jsonl, validate_completed


VERSION = "anchor-physician-oe-multiarm-analysis-v2"
EXPECTED_GATE_SPEC = {
    "primary_ci_high_below": 0.0,
    "holm_adjusted_one_sided_p_below": 0.05,
    "matched_coverage_minimum_groups": 12,
    "matched_coverage_ci_high_below": 0.0,
    "required_recall_delta_ci_low_at_least": -0.05,
    "direct_correctness_delta_ci_low_at_least": -0.05,
    "harm_delta_ci_high_at_most": 0.05,
    "refusal_delta_ci_high_at_most": 0.01,
    "length_ratio_at_least": 0.9,
    "evaluated_visual_claim_ratio_at_least": 0.9,
}
EXPECTED_GATE_NAMES = {
    "primary_error_reduction",
    "holm_adjusted_primary_p_below_0p05",
    "matched_coverage_error_reduction",
    "required_recall_noninferior_0p05",
    "direct_correctness_noninferior_0p05",
    "harm_not_increased_0p05",
    "refusal_not_increased_0p01",
    "length_at_least_90pct",
    "visual_claims_at_least_90pct",
}


def bind_t3(result: dict, prereg: dict) -> dict:
    if prereg.get("machine_gate_spec") != EXPECTED_GATE_SPEC:
        raise ValueError("preregistered machine gate spec differs from audited implementation")
    for method, contrast in result["contrasts"].items():
        gates = contrast.get("promotion_gates", {})
        if set(gates) != EXPECTED_GATE_NAMES:
            raise ValueError(f"promotion gate closure mismatch: {method}")
        if contrast.get("t3_promotion_authorized") is not all(gates.values()):
            raise ValueError(f"promotion decision does not equal all gates: {method}")
    result["protocol_version"] = VERSION
    result["evidence_stage"] = "T3"
    result["machine_gate_spec"] = EXPECTED_GATE_SPEC
    result["claim_boundary"] = (
        "Held-out physician T3 pilot only; promotion requires paired visual-error "
        "reduction with Holm control and no omission, coverage, correctness, harm, "
        "refusal, length, or evaluated-claim exchange. Full efficacy is not established."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--consensus-provenance", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    if (
        prereg.get("analysis_module") != "anchor.medeval.analyze_physician_oe_multiarm_v2"
        or prereg.get("baseline") != args.baseline
        or int(prereg.get("bootstrap_seed", -1)) != args.seed
        or int(prereg.get("bootstrap_iterations", 0)) != args.bootstrap_iterations
    ):
        raise ValueError("CLI analysis parameters differ from frozen preregistration")
    template = load_jsonl(args.template)
    consensus = load_jsonl(args.consensus)
    validation = validate_completed(template, consensus)
    provenance = json.loads(args.consensus_provenance.read_text())
    _validate_consensus_provenance(provenance, args.consensus, str(template[0]["bundle_id"]))
    result = bind_t3(
        analyze_multiarm(
            consensus,
            load_jsonl(args.mapping),
            baseline=args.baseline,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ),
        prereg,
    )
    result["validation"] = validation
    result["provenance"] = {
        "template": str(args.template.resolve()),
        "template_sha256": sha256_file(args.template),
        "consensus": str(args.consensus.resolve()),
        "consensus_sha256": sha256_file(args.consensus),
        "consensus_provenance": str(args.consensus_provenance.resolve()),
        "consensus_provenance_sha256": sha256_file(args.consensus_provenance),
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "prereg": str(args.prereg.resolve()),
        "prereg_sha256": sha256_file(args.prereg),
    }
    if args.output.exists():
        raise FileExistsError("physician multi-arm analysis is write-once")
    atomic_write_json(args.output, result)
    print(json.dumps({"promoted_methods": result["promoted_methods"]}, indent=2))


if __name__ == "__main__":
    main()
