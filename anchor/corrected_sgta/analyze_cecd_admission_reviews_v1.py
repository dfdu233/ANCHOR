#!/usr/bin/env python3
"""Fail-closed analysis of the blinded human-admission gate for CECD."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from corrected_sgta.run_cecd_factorial_v1 import (
    BASELINE_VIEW,
    IDENTITY_RENDER_NAME,
    PROMPT_TEMPLATES,
    SCIENCE_RENDER_NAMES,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import atomic_json, sha256_file


VERSION = "cecd-human-admission-analysis-v2-exact-science-grid"
YES_NO_UNABLE = {"yes", "no", "unable"}
VISIBILITY = {"unchanged", "a_clearer", "b_clearer", "unable"}
EXPECTED_NONBASELINE_RENDERS = tuple(
    name for name in SCIENCE_RENDER_NAMES if name != BASELINE_VIEW
)
EXPECTED_CANDIDATE_PROMPTS = tuple(name for name, _ in PROMPT_TEMPLATES[1:])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: str(value).strip().lower() for key, value in row.items()} for row in csv.DictReader(handle)]


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 1.0


def _kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    pairs = [(a, b) for a, b in zip(left, right) if a != "unable" and b != "unable"]
    if not pairs:
        return None
    agree = sum(a == b for a, b in pairs) / len(pairs)
    labels = ("yes", "no")
    expected = sum(
        (sum(a == label for a, _ in pairs) / len(pairs))
        * (sum(b == label for _, b in pairs) / len(pairs))
        for label in labels
    )
    return 1.0 if expected == 1.0 and agree == 1.0 else (agree - expected) / max(1 - expected, 1e-12)


def analyze(
    *,
    mapping: dict[str, Any],
    clinical_reviews: Sequence[list[dict[str, str]]],
    template_review: list[dict[str, str]],
    language_review: list[dict[str, str]],
    maximum_change_rate: float = 0.05,
    maximum_unable_rate: float = 0.10,
) -> dict[str, Any]:
    if len(clinical_reviews) != 2:
        raise ValueError("exactly two independent clinical reviews are required")
    clinical_map = {row["pair_id"]: row for row in mapping["clinical_pairs"]}
    expected_pairs = set(clinical_map)
    normalized_reviews = []
    for index, rows in enumerate(clinical_reviews, 1):
        by_id = {row.get("pair_id", ""): row for row in rows}
        if set(by_id) != expected_pairs or len(rows) != len(by_id):
            raise ValueError(f"clinical reviewer {index} pair IDs are incomplete or duplicated")
        for pair_id, row in by_id.items():
            if row.get("support_state_same_supported_refuted_undetermined") not in YES_NO_UNABLE:
                raise ValueError(f"invalid support-state response: reviewer={index} pair={pair_id}")
            if row.get("lesion_visibility") not in VISIBILITY:
                raise ValueError(f"invalid visibility response: reviewer={index} pair={pair_id}")
            if row.get("clinically_interchangeable") not in YES_NO_UNABLE:
                raise ValueError(f"invalid interchangeability response: reviewer={index} pair={pair_id}")
            if row.get("unable_to_judge") not in {"yes", "no"}:
                raise ValueError(f"invalid unable response: reviewer={index} pair={pair_id}")
        normalized_reviews.append(by_id)

    transforms = sorted({str(row["transform"]) for row in clinical_map.values()})
    clinical_results = []
    for transform in transforms:
        pair_ids = sorted(
            pair_id for pair_id, row in clinical_map.items() if str(row["transform"]) == transform
        )
        reviewer_rows = []
        for reviewer in normalized_reviews:
            values = [reviewer[pair_id] for pair_id in pair_ids]
            unable = [
                row["unable_to_judge"] == "yes"
                or "unable" in {
                    row["support_state_same_supported_refuted_undetermined"],
                    row["lesion_visibility"],
                    row["clinically_interchangeable"],
                }
                for row in values
            ]
            judgeable = [row for row, flag in zip(values, unable) if not flag]
            support_change = _rate(
                [row["support_state_same_supported_refuted_undetermined"] == "no" for row in judgeable]
            )
            visibility_change = _rate([row["lesion_visibility"] != "unchanged" for row in judgeable])
            noninterchangeable = _rate([row["clinically_interchangeable"] == "no" for row in judgeable])
            unable_rate = _rate(unable)
            reviewer_rows.append(
                {
                    "n": len(values),
                    "judgeable": len(judgeable),
                    "support_state_change_rate": support_change,
                    "visibility_change_rate": visibility_change,
                    "noninterchangeable_rate": noninterchangeable,
                    "unable_rate": unable_rate,
                    "passed": bool(
                        support_change <= maximum_change_rate
                        and visibility_change <= maximum_change_rate
                        and noninterchangeable <= maximum_change_rate
                        and unable_rate <= maximum_unable_rate
                    ),
                }
            )
        support_left = [
            normalized_reviews[0][pair_id]["support_state_same_supported_refuted_undetermined"]
            for pair_id in pair_ids
        ]
        support_right = [
            normalized_reviews[1][pair_id]["support_state_same_supported_refuted_undetermined"]
            for pair_id in pair_ids
        ]
        clinical_results.append(
            {
                "transform": transform,
                "identity_control": transform == "identity_lossless_duplicate",
                "reviewers": reviewer_rows,
                "support_decision_kappa": _kappa(support_left, support_right),
                "passed": all(row["passed"] for row in reviewer_rows),
            }
        )

    language_map = {row["item_id"]: row for row in mapping["language_items"]}
    expected_language = set(language_map)
    decision_fields = (
        "same_clinical_proposition",
        "same_speech_act",
        "same_certainty_demand",
        "same_answer_space",
    )
    language_inputs = []
    for role, rows in (("clinical_template", template_review), ("language", language_review)):
        by_id = {row.get("item_id", ""): row for row in rows}
        if set(by_id) != expected_language or len(rows) != len(by_id):
            raise ValueError(f"{role} review IDs are incomplete or duplicated")
        for item_id, row in by_id.items():
            for field in decision_fields:
                if row.get(field) not in YES_NO_UNABLE:
                    raise ValueError(f"invalid {role} response: item={item_id} field={field}")
        language_inputs.append((role, by_id))

    prompt_results = []
    for candidate in sorted({row["candidate_prompt"] for row in language_map.values()}):
        item_ids = sorted(
            item_id for item_id, row in language_map.items() if row["candidate_prompt"] == candidate
        )
        roles = []
        for role, review in language_inputs:
            fields = {
                field: all(review[item_id][field] == "yes" for item_id in item_ids)
                for field in decision_fields
            }
            roles.append({"role": role, "fields": fields, "passed": all(fields.values())})
        prompt_results.append(
            {"candidate_prompt": candidate, "reviewers": roles, "passed": all(row["passed"] for row in roles)}
        )

    primary = [row for row in clinical_results if not row["identity_control"]]
    identity = [row for row in clinical_results if row["identity_control"]]
    admitted_renders = [row["transform"] for row in primary if row["passed"]]
    admitted_prompts = [row["candidate_prompt"] for row in prompt_results if row["passed"]]
    render_set_exact = bool(
        set(transforms)
        == set(EXPECTED_NONBASELINE_RENDERS) | {IDENTITY_RENDER_NAME}
        and set(admitted_renders) == set(EXPECTED_NONBASELINE_RENDERS)
        and len(admitted_renders) == len(EXPECTED_NONBASELINE_RENDERS)
    )
    prompt_set_exact = bool(
        {row["candidate_prompt"] for row in prompt_results}
        == set(EXPECTED_CANDIDATE_PROMPTS)
        and set(admitted_prompts) == set(EXPECTED_CANDIDATE_PROMPTS)
        and len(admitted_prompts) == len(EXPECTED_CANDIDATE_PROMPTS)
    )
    passed = bool(
        render_set_exact
        and prompt_set_exact
        and len(identity) == 1
        and all(row["passed"] for row in identity)
    )
    return {
        "version": VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "maximum_change_rate": maximum_change_rate,
        "maximum_unable_rate": maximum_unable_rate,
        "clinical_render_families": clinical_results,
        "language_prompt_families": prompt_results,
        "admitted_nonbaseline_renders": admitted_renders,
        "admitted_candidate_prompts": admitted_prompts,
        "science_grid_contract": {
            "baseline_render": BASELINE_VIEW,
            "required_nonbaseline_renders": list(EXPECTED_NONBASELINE_RENDERS),
            "required_candidate_prompts": list(EXPECTED_CANDIDATE_PROMPTS),
            "identity_render": IDENTITY_RENDER_NAME,
            "render_set_exact": render_set_exact,
            "prompt_set_exact": prompt_set_exact,
            "all_scored_cells_human_admitted": bool(
                render_set_exact and prompt_set_exact
            ),
        },
        "cecd_model_scoring_authorized": passed,
        "authorization_basis": (
            "four independent blinded role returns over clinical equivalence and language "
            "equivalence; pixel similarity is prohibited as clinical-admission evidence"
        ),
        "claim_boundary": "human admission authorizes an engineering factorial only, not a mechanism or paper claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--clinical-review", type=Path, action="append", required=True)
    parser.add_argument("--clinical-template-review", type=Path, required=True)
    parser.add_argument("--language-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping_path = args.pack_dir / "sealed_mapping.json"
    result = analyze(
        mapping=json.loads(mapping_path.read_text()),
        clinical_reviews=[read_csv(path) for path in args.clinical_review],
        template_review=read_csv(args.clinical_template_review),
        language_review=read_csv(args.language_review),
    )
    result["provenance"] = {
        "sealed_mapping": str(mapping_path.resolve()),
        "sealed_mapping_sha256": sha256_file(mapping_path),
        "clinical_reviews": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.clinical_review
        ],
        "clinical_template_review": {
            "path": str(args.clinical_template_review.resolve()),
            "sha256": sha256_file(args.clinical_template_review),
        },
        "language_review": {
            "path": str(args.language_review.resolve()),
            "sha256": sha256_file(args.language_review),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
