#!/usr/bin/env python3
"""Fail-closed CPU feasibility audit for same-report twins over all 108 pairs.

The two original donor contexts are anonymized by context hash before semantic
screening.  Their legacy present/absent arm names are used only to identify the
two original (rather than derived-control) rows and are never treated as a
polarity label.  Polarity is reparsed independently from the report text.

This stage intentionally stops without writing generation manifests unless
all four core findings have at least eight eligible pairs.  It never relaxes a
semantic gate to satisfy that threshold and never runs a GPU model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.build_same_report_polarity_twin_pilot_v1 import (
    assess_report,
    sha256_file,
    stable_hash,
)


SOURCE_ROWS = Path("corrected_runs/matched_retrieval_polarity_canary_v1/canary.jsonl")
SOURCE_PAIRS = Path("corrected_runs/matched_retrieval_polarity_canary_v1/matched_pairs.jsonl")
OUT_DIR = Path("corrected_runs/same_report_polarity_twin_full108_feasibility_v1")
PAIR_AUDIT = OUT_DIR / "pair_feasibility.jsonl"
RESULT = OUT_DIR / "result.json"
PROTOCOL = "same-report-polarity-twin-full108-feasibility-v1"
CORE_FINDINGS = ("pleural_effusion", "cardiomegaly", "pneumothorax", "lung_opacity")
MINIMUM_PER_CORE_FINDING = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def anonymous_donors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard the old polarity role, then assign a stable text-only donor id."""
    donors = [
        {
            "context": row["context"],
            "context_sha256": hashlib.sha256(row["context"].encode("utf-8")).hexdigest(),
        }
        for row in rows
    ]
    donors.sort(key=lambda row: stable_hash(f"{PROTOCOL}:anonymous:{row['context_sha256']}"))
    for index, donor in enumerate(donors):
        donor["anonymous_donor_id"] = f"donor_{index}"
    return donors


def main() -> None:
    source_rows = read_jsonl(SOURCE_ROWS)
    source_pairs = read_jsonl(SOURCE_PAIRS)
    pair_metadata = {row["pair_id"]: row for row in source_pairs}
    raw_originals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        # Storage-role filter only.  The value is discarded before assessment.
        if row["arm"] in {"present", "absent"}:
            raw_originals[row["pair_id"]].append(row)
    if len(source_pairs) != 108 or set(raw_originals) != set(pair_metadata):
        raise RuntimeError("source is not the frozen complete 108-pair canary")
    if any(len(rows) != 2 for rows in raw_originals.values()):
        raise RuntimeError("every pair must expose exactly two original donor contexts")

    audits: list[dict[str, Any]] = []
    total_by_finding = Counter()
    eligible_by_finding = Counter()
    eligible_polarity_by_finding: dict[str, Counter[str]] = defaultdict(Counter)
    failure_by_finding: dict[str, Counter[str]] = defaultdict(Counter)

    for pair_id in sorted(raw_originals):
        meta = pair_metadata[pair_id]
        finding = meta["finding"]
        total_by_finding[finding] += 1
        donors = anonymous_donors(raw_originals[pair_id])
        donor_audits: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for donor in donors:
            candidate, reasons = assess_report(donor["context"], finding)
            item: dict[str, Any] = {
                "anonymous_donor_id": donor["anonymous_donor_id"],
                "context_sha256": donor["context_sha256"],
                "eligible": candidate is not None,
                "exclusion_reasons": reasons,
            }
            if candidate is not None:
                item.update({
                    "independently_parsed_polarity": candidate["direct_polarity"],
                    "finding_sentence": candidate["sentence"],
                    "finding_sentence_span": [candidate["sentence_begin"], candidate["sentence_end"]],
                })
                candidates.append({**candidate, **donor})
            donor_audits.append(item)

        eligible = bool(candidates)
        pair_audit: dict[str, Any] = {
            "pair_id": pair_id,
            "finding": finding,
            "cohort": meta["cohort"],
            "query_image_id": meta["source_id"],
            "query_patient_hash": stable_hash(str(meta["query_patient"])),
            "legacy_arm_polarity_consulted": False,
            "eligible": eligible,
            "donors": donor_audits,
        }
        if eligible:
            selected = min(
                candidates,
                key=lambda row: stable_hash(
                    f"{PROTOCOL}:candidate:{pair_id}:{row['context_sha256']}:{row['sentence']}"
                ),
            )
            eligible_by_finding[finding] += 1
            eligible_polarity_by_finding[finding][selected["direct_polarity"]] += 1
            pair_audit["selected_anonymous_donor_id"] = selected["anonymous_donor_id"]
            pair_audit["selected_context_sha256"] = selected["context_sha256"]
            pair_audit["selected_independently_parsed_polarity"] = selected["direct_polarity"]
        else:
            pair_reasons = sorted(
                set(reason for donor in donor_audits for reason in donor["exclusion_reasons"])
            )
            pair_audit["pair_exclusion_reason_union"] = pair_reasons
            failure_by_finding[finding].update(pair_reasons)
        audits.append(pair_audit)

    core_counts = {finding: eligible_by_finding[finding] for finding in CORE_FINDINGS}
    insufficient = {
        finding: {"eligible": count, "required": MINIMUM_PER_CORE_FINDING}
        for finding, count in core_counts.items()
        if count < MINIMUM_PER_CORE_FINDING
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_AUDIT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audits),
        encoding="utf-8",
    )

    # The current frozen source is expected to stop here.  Keeping the stop
    # explicit prevents a future caller from silently constructing an
    # underpowered or semantically relaxed substrate.
    if insufficient:
        decision = "stopped_core_finding_minimum_not_met"
        manifests = {
            "generated": False,
            "reason": "at least one core finding has fewer than eight strictly eligible pairs",
            "discovery_confirmation_split": "not_attempted",
            "exact_token_manifest_audit": "not_attempted",
        }
    else:
        raise RuntimeError(
            "all core findings unexpectedly passed; implement and separately audit the "
            "patient/image-disjoint discovery-confirmation manifest stage before proceeding"
        )

    finding_summary = {}
    for finding in sorted(total_by_finding):
        eligible = eligible_by_finding[finding]
        total = total_by_finding[finding]
        finding_summary[finding] = {
            "total_pairs": total,
            "eligible_pairs": eligible,
            "excluded_pairs": total - eligible,
            "exclusion_rate": (total - eligible) / total,
            "selected_candidate_independent_polarity": dict(
                sorted(eligible_polarity_by_finding[finding].items())
            ),
            "excluded_pair_reason_incidence": dict(
                sorted(failure_by_finding[finding].items())
            ),
        }

    result = {
        "status": decision,
        "protocol": PROTOCOL,
        "gpu_execution": "not_run",
        "source": {
            "canary_jsonl": str(SOURCE_ROWS),
            "canary_sha256": sha256_file(SOURCE_ROWS),
            "matched_pairs_jsonl": str(SOURCE_PAIRS),
            "matched_pairs_sha256": sha256_file(SOURCE_PAIRS),
            "pair_count": len(source_pairs),
        },
        "semantic_contract": {
            "same_as_primary_32_pair_substrate": True,
            "legacy_arm_polarity_consulted": False,
            "donors_anonymized_by_context_hash_before_assessment": True,
            "fail_closed": True,
            "no_gate_relaxed": True,
            "outcome_or_model_output_read": False,
        },
        "core_finding_gate": {
            "findings": list(CORE_FINDINGS),
            "minimum_eligible_pairs_each": MINIMUM_PER_CORE_FINDING,
            "observed": core_counts,
            "insufficient": insufficient,
            "passed": not insufficient,
        },
        "finding_summary": finding_summary,
        "manifest_stage": manifests,
        "mechanism_substrate_verdict": (
            "insufficient_strict_claim-isolated coverage for a balanced four-finding "
            "same-report polarity mechanism experiment"
        ),
        "artifacts": {
            "pair_feasibility_jsonl": str(PAIR_AUDIT),
            "pair_feasibility_sha256": sha256_file(PAIR_AUDIT),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
