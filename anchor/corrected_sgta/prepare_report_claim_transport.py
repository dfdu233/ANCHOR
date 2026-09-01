#!/usr/bin/env python3
"""Build an unknown-aware report claim universe for fixed-K transport audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corrected_sgta.analyze_no_free_grounding import sha256_file
from corrected_sgta.radgraph_claims import load_ontology_aliases


VERSION = "report-claim-transport-manifest-v1"
STATES = ("supported", "refuted", "undetermined", "unverified")


def _positive_draft_findings(report: dict[str, Any]) -> list[str]:
    return sorted({
        str(claim["finding"])
        for claim in report.get("claims", [])
        if claim.get("polarity") == "present"
        and claim.get("provenance") == "image_grounded"
    })


def _reference_state(claims: list[dict[str, Any]]) -> str:
    image_claims = [claim for claim in claims if claim.get("provenance") == "image_grounded"]
    definite_present = any(
        claim.get("polarity") == "present" and claim.get("uncertainty") == "definite"
        for claim in image_claims
    )
    definite_absent = any(
        claim.get("polarity") == "absent" and claim.get("uncertainty") == "definite"
        for claim in image_claims
    )
    uncertain = any(claim.get("uncertainty") == "uncertain" for claim in image_claims)
    if uncertain or (definite_present and definite_absent):
        return "undetermined"
    if definite_present:
        return "supported"
    if definite_absent:
        return "refuted"
    return "unverified"


def build(
    draft_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    input_rows: list[dict[str, Any]],
    findings: list[str],
) -> dict[str, Any]:
    drafts = {str(report["id"]): report for report in draft_payload["reports"]}
    references = {str(report["id"]): report for report in reference_payload["reports"]}
    inputs = {str(row.get("id", row.get("qid"))): row for row in input_rows}
    if set(drafts) != set(references) or set(drafts) != set(inputs):
        raise ValueError("draft, reference, and generation input IDs are not aligned")
    images = []
    score_rows = []
    qid = 0
    for image_id in sorted(drafts):
        selected = _positive_draft_findings(drafts[image_id])
        if not selected:
            continue
        by_finding: dict[str, list[dict[str, Any]]] = {finding: [] for finding in findings}
        for claim in references[image_id].get("claims", []):
            finding = str(claim.get("finding"))
            if finding in by_finding:
                by_finding[finding].append(claim)
        states = {finding: _reference_state(by_finding[finding]) for finding in findings}
        if not any(state in {"supported", "refuted"} for state in states.values()):
            continue
        image_name = str(inputs[image_id]["img_name"])
        split_value = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) % 2
        split = "dev" if split_value == 0 else "holdout"
        images.append({
            "image_id": image_id,
            "image": image_name,
            "split": split,
            "draft_selected": selected,
            "k": len(selected),
            "reference_states": states,
        })
        for finding in findings:
            score_rows.append({
                "qid": qid,
                "img_name": image_name,
                "image_id": image_id,
                "finding": finding,
                "question": (
                    f"Does this chest X-ray show {finding.replace('_', ' ')}?"
                ),
                "answer": states[finding],
                "split": split,
            })
            qid += 1
    return {
        "version": VERSION,
        "states": list(STATES),
        "findings": findings,
        "images": images,
        "score_rows": score_rows,
        "n_images": len(images),
        "n_score_rows": len(score_rows),
        "n_by_split": {
            split: sum(image["split"] == split for image in images)
            for split in ("dev", "holdout")
        },
        "truth_contract": (
            "Unmentioned or knowledge-derived reference findings are unverified, not "
            "refuted. Conflicts and uncertain mentions are undetermined."
        ),
        "evidence_grade": "C: single-report RadGraph extraction; screening only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-claims", type=Path, required=True)
    parser.add_argument("--reference-claims", type=Path, required=True)
    parser.add_argument("--generation-input", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-questions", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.score_questions.exists():
        raise FileExistsError("output paths must not already exist")
    findings = sorted(load_ontology_aliases(args.ontology))
    result = build(
        json.loads(args.draft_claims.read_text()),
        json.loads(args.reference_claims.read_text()),
        json.loads(args.generation_input.read_text()),
        findings,
    )
    payload = {
        "config": {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "draft_claims_sha256": sha256_file(args.draft_claims),
            "reference_claims_sha256": sha256_file(args.reference_claims),
            "generation_input_sha256": sha256_file(args.generation_input),
            "ontology_sha256": sha256_file(args.ontology),
            "code_sha256": sha256_file(Path(__file__)),
        },
        "result": {key: value for key, value in result.items() if key != "score_rows"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.score_questions.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.score_questions.write_text(json.dumps(result["score_rows"], indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **{
        key: result[key] for key in ("n_images", "n_score_rows", "n_by_split")
    }}, indent=2))


if __name__ == "__main__":
    main()
