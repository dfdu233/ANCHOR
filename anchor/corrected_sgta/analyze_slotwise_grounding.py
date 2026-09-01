#!/usr/bin/env python3
"""Quantify the addressable error mass of slotwise claim backoff.

This deliberately calls non-exact modifier values *candidates*, not errors:
single reports are incomplete and anatomy strings can be synonymous.  The
audit is a screening gate for whether a slotwise method deserves a formal
experiment, never a source of clinical truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "slotwise-grounding-addressable-mass-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_claims(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        claim
        for claim in report["claims"]
        if claim.get("provenance") == "image_grounded"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-comparable-modifiers", type=int, default=30)
    parser.add_argument("--minimum-different-values", type=int, default=10)
    args = parser.parse_args()

    generated_payload = json.loads(args.generated.read_text(encoding="utf-8"))
    reference_payload = json.loads(args.reference.read_text(encoding="utf-8"))
    generated = generated_payload["reports"]
    reference = reference_payload["reports"]
    if len(generated) != len(reference):
        raise ValueError("generated/reference report counts differ")

    counts: Counter[str] = Counter()
    candidate_examples: list[dict[str, Any]] = []
    per_report: list[dict[str, Any]] = []
    for generated_report, reference_report in zip(generated, reference):
        if generated_report["id"] != reference_report["id"]:
            raise ValueError("generated/reference report order differs")
        local: Counter[str] = Counter()
        references = image_claims(reference_report)
        for claim in image_claims(generated_report):
            counts["generated_claims"] += 1
            local["generated_claims"] += 1
            parent_matches = [
                item
                for item in references
                if item["finding"] == claim["finding"]
                and item["polarity"] == claim["polarity"]
            ]
            if not parent_matches:
                counts["parent_unmatched"] += 1
                local["parent_unmatched"] += 1
                continue
            counts["parent_matched"] += 1
            local["parent_matched"] += 1
            for slot in ("anatomy", "attributes"):
                value = claim.get(slot)
                if not value:
                    continue
                counts[f"{slot}_generated"] += 1
                reference_values = [
                    item.get(slot) for item in parent_matches if item.get(slot)
                ]
                if not reference_values:
                    counts[f"{slot}_reference_unspecified"] += 1
                elif value in reference_values:
                    counts[f"{slot}_exact"] += 1
                    counts["comparable_modifiers"] += 1
                else:
                    counts[f"{slot}_different_value_candidate"] += 1
                    counts["comparable_modifiers"] += 1
                    counts["different_value_candidates"] += 1
                    if len(candidate_examples) < 20:
                        candidate_examples.append(
                            {
                                "id": generated_report["id"],
                                "slot": slot,
                                "generated_claim": claim,
                                "reference_parent_matches": parent_matches,
                                "warning": (
                                    "Candidate only: may be synonymy or reference omission, "
                                    "not a physician-verified attribute error."
                                ),
                            }
                        )
        per_report.append({"id": generated_report["id"], "counts": dict(local)})

    comparable = counts["comparable_modifiers"]
    different = counts["different_value_candidates"]
    sufficient = (
        comparable >= args.minimum_comparable_modifiers
        and different >= args.minimum_different_values
    )
    summary = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "generated": str(args.generated.resolve()),
            "generated_sha256": sha256_file(args.generated),
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
        },
        "n_reports": len(generated),
        "counts": dict(counts),
        "rates": {
            "parent_match_rate": (
                counts["parent_matched"] / counts["generated_claims"]
                if counts["generated_claims"]
                else None
            ),
            "different_value_candidate_rate_among_comparable_modifiers": (
                different / comparable if comparable else None
            ),
            "maximum_addressable_fraction_of_generated_claims": (
                different / counts["generated_claims"]
                if counts["generated_claims"]
                else None
            ),
        },
        "screening_gate": {
            "minimum_comparable_modifiers": args.minimum_comparable_modifiers,
            "minimum_different_values": args.minimum_different_values,
            "passed": sufficient,
            "decision": (
                "retain_for_physician_verified_pilot"
                if sufficient
                else "insufficient_addressable_error_mass_for_main_method"
            ),
        },
        "candidate_examples": candidate_examples,
        "per_report": per_report,
        "claim_boundary": (
            "RadGraph structure plus a single reference report is a grade-C "
            "screening proxy. Different strings are not clinical errors; only "
            "reader/physician labels can promote them to truth."
        ),
        "code_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
