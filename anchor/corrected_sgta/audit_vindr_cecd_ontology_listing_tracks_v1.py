#!/usr/bin/env python3
"""Outcome-blind comparison of the VinDr 8- and 14-finding listing tracks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.prepare_vindr_reader_manifest import sha256_file


VERSION = "vindr-cecd-ontology-listing-track-audit-v1"
SPLITS = ("pilot", "dev", "confirmation")


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_balanced_v2(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    findings = tuple(sorted(str(value) for value in summary.get("eligible_findings", [])))
    require(len(findings) == 8, "balanced v2 must freeze exactly eight findings")
    require(int(summary.get("oe_claim_rows", -1)) == len(rows), "v2 row count drift")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    image_split: dict[str, str] = {}
    for row in rows:
        image_id = str(row.get("image_id", ""))
        split = str(row.get("experiment_split", ""))
        finding = str(row.get("finding", ""))
        require(image_id and split in SPLITS and finding in findings, "invalid v2 row")
        require(row.get("reader_count") == 3, "v2 row lacks three readers")
        require(row.get("reader_panel") == ["R8", "R9", "R10"], "v2 panel drift")
        require(
            row.get("reference_relevance") in {"required", "optional", "out_of_scope"},
            "v2 relevance missing",
        )
        if image_id in image_split:
            require(image_split[image_id] == split, "v2 image crosses splits")
        image_split[image_id] = split
        grouped[image_id].append(row)
    require(
        all({str(row["finding"]) for row in image_rows} == set(findings) for image_rows in grouped.values()),
        "v2 does not contain complete image x eight-finding universes",
    )
    require(all(len(image_rows) == 8 for image_rows in grouped.values()), "v2 duplicate finding")

    by_split_images = Counter(image_split.values())
    required_k: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    per_finding: dict[str, dict[str, Counter[int]]] = {
        split: {finding: Counter() for finding in findings} for split in SPLITS
    }
    multi_by_split = Counter()
    for image_id, image_rows in grouped.items():
        split = image_split[image_id]
        required = sum(int(row["positive_votes"]) == 3 for row in image_rows)
        required_k[split][required] += 1
        multi_by_split[split] += int(required >= 2)
        for row in image_rows:
            per_finding[split][str(row["finding"])][int(row["positive_votes"])] += 1
    dev_ready = all(
        per_finding["dev"][finding][vote] >= 20
        for finding in findings
        for vote in range(4)
    )
    confirmation_ready = all(
        per_finding["confirmation"][finding][vote] >= 60
        for finding in findings
        for vote in range(4)
    )
    return {
        "finding_count": len(findings),
        "findings": list(findings),
        "claim_rows": len(rows),
        "images": len(grouped),
        "images_by_split": dict(by_split_images),
        "required_K_histogram_by_split": {
            split: {str(key): value for key, value in sorted(counts.items())}
            for split, counts in required_k.items()
        },
        "true_multiclaim_images_by_split": dict(multi_by_split),
        "true_multiclaim_images": sum(multi_by_split.values()),
        "complete_fixed_claim_universe": True,
        "image_disjoint": True,
        "patient_disjoint_verifiable": False,
        "all_eight_dev_vote_bins_at_least_20": dev_ready,
        "all_eight_confirmation_vote_bins_at_least_60": confirmation_ready,
        "per_finding_vote_bins": {
            split: {
                finding: {f"{vote}/3": counts[vote] for vote in range(4)}
                for finding, counts in values.items()
            }
            for split, values in per_finding.items()
        },
    }


def summarize_breadth(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    task = manifest.get("task_contract", {})
    require(
        task.get("formal_task_type") == "ontology_constrained_open_cardinality_listing",
        "breadth task contract drift",
    )
    require(task.get("free_form_oe") is False, "breadth track mislabeled free OE")
    require(len(task.get("target_finding_ids", [])) == 14, "breadth ontology is not 14")
    require(
        len(rows) == int(manifest["reference_contract"]["reference_rows"]),
        "breadth row count drift",
    )
    image_ids = [str(row["image_id"]) for row in rows]
    require(len(set(image_ids)) == len(image_ids), "breadth image duplicated")
    split_counts = Counter(str(row["experiment_split"]) for row in rows)
    multi = sum(
        str(row["sampling_stratum"]) == "multiple_unanimous_target_findings"
        for row in rows
    )
    selected_required = Counter(len(row["required_finding_ids"]) for row in rows)
    census = manifest["source_census"]
    return {
        "finding_count": 14,
        "images": len(rows),
        "claim_rows": len(rows) * 14,
        "images_by_split": dict(split_counts),
        "selected_required_K_histogram": {
            str(key): value for key, value in sorted(selected_required.items())
        },
        "selected_true_multiclaim_images": multi,
        "fixed_panel_population_images": census["fixed_panel_images"],
        "population_true_multiclaim_images": census[
            "images_with_at_least_two_unanimous_target_findings"
        ],
        "population_any_outside_target": census[
            "images_with_any_reader_positive_outside_target_ontology"
        ],
        "population_unanimous_outside_target": census[
            "images_with_unanimous_positive_outside_target_ontology"
        ],
        "complete_fixed_claim_universe": True,
        "image_disjoint": True,
        "patient_disjoint_verifiable": False,
        "free_oe_authorized": False,
    }


def audit_tracks(v2_dir: Path, breadth_manifest_path: Path) -> dict[str, Any]:
    summary_path = v2_dir / "summary_v2.json"
    v2_reference_path = v2_dir / "oe_listing_reference_v2.jsonl"
    breadth_manifest = load_json(breadth_manifest_path)
    breadth_reference_path = breadth_manifest_path.parent / str(
        breadth_manifest["reference_contract"]["reference_file"]
    )
    v2 = summarize_balanced_v2(load_json(summary_path), load_jsonl(v2_reference_path))
    breadth = summarize_breadth(
        breadth_manifest, load_jsonl(breadth_reference_path)
    )
    require(v2["true_multiclaim_images"] >= 500, "v2 lacks enough multiclaim images")
    require(v2["all_eight_dev_vote_bins_at_least_20"], "v2 dev per-finding gate fails")
    require(
        v2["all_eight_confirmation_vote_bins_at_least_60"],
        "v2 confirmation per-finding gate fails",
    )
    require(
        breadth["population_true_multiclaim_images"] >= 1000,
        "breadth population lacks multiclaim substrate",
    )
    return {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "conditional_go_two_track_closed_ontology_listing",
        "outcome_blind": True,
        "model_outputs_or_scores_read": False,
        "gpu_used": False,
        "balanced_8_finding_mechanism_track": v2,
        "natural_14_finding_breadth_track": breadth,
        "decision": {
            "primary_near_term_track": "balanced_8_finding_mechanism_track",
            "breadth_confirmation_track": "natural_14_finding_breadth_track",
            "unrestricted_native_oe": "strict_no_go_without_independent_out_of_ontology_truth",
            "patient_level_generalization": "not_authorized",
            "model_or_gpu_authorized": False,
        },
        "provenance": {
            "v2_summary": str(summary_path.resolve()),
            "v2_summary_sha256": sha256_file(summary_path),
            "v2_reference": str(v2_reference_path.resolve()),
            "v2_reference_sha256": sha256_file(v2_reference_path),
            "breadth_manifest": str(breadth_manifest_path.resolve()),
            "breadth_manifest_sha256": sha256_file(breadth_manifest_path),
            "breadth_reference": str(breadth_reference_path.resolve()),
            "breadth_reference_sha256": sha256_file(breadth_reference_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--breadth-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_tracks(args.v2_dir, args.breadth_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
