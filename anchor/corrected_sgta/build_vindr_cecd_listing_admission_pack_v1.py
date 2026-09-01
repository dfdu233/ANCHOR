#!/usr/bin/env python3
"""Build the independent blinded admission pack for VinDr ontology listing.

The frozen 60-image pilot is taken verbatim from the 14-finding breadth
manifest.  This builder reuses the CECD DICOM renderer implementation but not
the prior polar-question admission.  It reads no model outputs and grants no
model/GPU authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.prepare_vindr_cecd_ontology_listing_v1 import (
    IDENTITY_RENDER_ID,
    NONE_TOKEN,
    SCIENCE_PROMPT_IDS,
    SCIENCE_RENDER_IDS,
    STRATA,
    TARGET_FINDINGS,
    canonical_hash,
    prompt_texts,
)
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    build_render_views,
    read_dicom_pixels,
)
from corrected_sgta.validate_vindr_cecd_ontology_listing_v1 import validate_pack


VERSION = "vindr-cecd-listing-blinded-admission-pack-v1"
RETURN_SCHEMA_VERSION = "vindr-cecd-listing-admission-return-schema-v1"
ROLES = (
    "clinical_reviewer_1",
    "clinical_reviewer_2",
    "clinical_template_reviewer",
    "language_reviewer",
)
PROFESSIONAL_ROLE = {
    "clinical_reviewer_1": "physician",
    "clinical_reviewer_2": "physician",
    "clinical_template_reviewer": "physician",
    "language_reviewer": "language_expert",
}
CLINICAL_DECISION_FIELDS = (
    "same_support_state_for_all_14",
    "visibility_change",
    "listing_interchangeable",
    "changed_finding_ids",
    "unable_to_judge",
    "comments",
)
PROMPT_DECISION_FIELDS = (
    "same_target_ontology",
    "same_inclusion_obligation",
    "same_speech_act",
    "same_certainty_demand",
    "same_answer_space",
    "same_output_grammar",
    "unable_to_judge",
    "comments",
)
ALLOWED = {
    "same_support_state_for_all_14": ["yes", "no", "unable"],
    "visibility_change": ["unchanged", "A_clearer", "B_clearer", "mixed", "unable"],
    "listing_interchangeable": ["yes", "no", "unable"],
    "same_target_ontology": ["yes", "no", "unable"],
    "same_inclusion_obligation": ["yes", "no", "unable"],
    "same_speech_act": ["yes", "no", "unable"],
    "same_certainty_demand": ["yes", "no", "unable"],
    "same_answer_space": ["yes", "no", "unable"],
    "same_output_grammar": ["yes", "no", "unable"],
    "unable_to_judge": ["yes", "no"],
}


class PackError(RuntimeError):
    pass


def _rank(seed: int, *parts: object) -> str:
    return hashlib.sha256(
        ":".join(map(str, (VERSION, seed, *parts))).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_png(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=1)
    return sha256_file(path)


def select_frozen_pilot(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # This validates the full source pack and proves that no model result was
    # involved before taking the already-frozen pilot partition.
    validate_pack(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_path = manifest_path.parent / manifest["reference_contract"]["reference_file"]
    rows = [
        row for row in load_jsonl(reference_path) if row["experiment_split"] == "pilot"
    ]
    if len(rows) != 60:
        raise PackError(f"frozen breadth pilot must contain 60 images, found {len(rows)}")
    counts = Counter(str(row["sampling_stratum"]) for row in rows)
    if counts != {stratum: 20 for stratum in STRATA}:
        raise PackError(f"pilot stratum contract drift: {dict(counts)}")
    if len({str(row["image_id"]) for row in rows}) != 60:
        raise PackError("pilot images are not unique")
    return manifest, sorted(rows, key=lambda row: str(row["image_id"]))


def read_all_boxes(path: Path, selected: set[str]) -> dict[str, list[dict[str, float]]]:
    boxes: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "x_min", "y_min", "x_max", "y_max"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise PackError("bbox CSV lacks required columns")
        for row in reader:
            image_id = str(row["image_id"])
            if image_id not in selected:
                continue
            raw = [str(row[name]).strip() for name in ("x_min", "y_min", "x_max", "y_max")]
            if not all(raw):
                continue
            x0, y0, x1, y1 = map(float, raw)
            if x1 > x0 and y1 > y0:
                boxes[image_id].append(
                    {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}
                )
    return dict(boxes)


def return_schema() -> dict[str, Any]:
    finding_ids = [finding for finding, _ in TARGET_FINDINGS]
    return {
        "schema_version": RETURN_SCHEMA_VERSION,
        "protocol_id": VERSION,
        "roles": {
            role: {
                "kind": "clinical" if role.startswith("clinical_reviewer_") else "prompt",
                "professional_role": PROFESSIONAL_ROLE[role],
                "template_filename": f"{role}.csv",
                "completed_filename": f"{role}.completed.csv",
                "attestation_filename": f"{role}.attestation.json",
                "decision_fields": list(
                    CLINICAL_DECISION_FIELDS
                    if role.startswith("clinical_reviewer_")
                    else PROMPT_DECISION_FIELDS
                ),
            }
            for role in ROLES
        },
        "allowed_values": ALLOWED,
        "changed_finding_ids": {
            "grammar": "empty or semicolon-separated unique canonical finding IDs",
            "allowed_ids": finding_ids,
            "required_when_same_support_state_is_no": True,
            "must_be_empty_when_same_support_state_is_yes": True,
        },
        "attestation": {
            "top_level_keys": ["protocol_id", "review_role", "reviewer"],
            "reviewer_keys": [
                "reviewer_id",
                "professional_role",
                "independent_review",
                "blinded_to_sealed_mapping",
                "completed_at_utc",
            ],
            "four_distinct_reviewer_ids_required": True,
        },
    }


def _instructions(kind: str) -> str:
    ontology = ", ".join(finding for finding, _ in TARGET_FINDINGS)
    if kind == "clinical":
        return f"""# Independent blinded VinDr listing-render review

Review every image pair independently. The task is whether the two displays
would support the same closed-ontology abnormality list, not whether they look
pixel-identical. The 14 canonical finding IDs are:

{ontology}

For every row fill:
- `same_support_state_for_all_14`: `yes`, `no`, or `unable`.
- `visibility_change`: `unchanged`, `A_clearer`, `B_clearer`, `mixed`, or `unable`.
- `listing_interchangeable`: `yes`, `no`, or `unable`.
- `changed_finding_ids`: empty unless support state differs; when it differs,
  enter unique canonical IDs separated by semicolons.
- `unable_to_judge`: `yes` iff any primary judgment is `unable`; otherwise `no`.
- `comments`: optional, with no patient identifiers.

Do not infer the source image, transform, baseline side, reader votes, sampling
stratum, or model behavior. Do not consult another reviewer. Return the
completed CSV and your own attestation under the exact names in RETURN_SCHEMA.json.
"""
    return f"""# Independent blinded VinDr listing-prompt review

Compare each wording pair without answering the medical task. Judge separately
whether the two wordings preserve the exact target ontology, obligation to
include every visible member, speech act, certainty demand, answer space, and
comma-separated exact-label output grammar. The empty-set serialization is
`{NONE_TOKEN}` and is not a fifteenth clinical claim.

Fill every `same_*` field with `yes`, `no`, or `unable`; set
`unable_to_judge=yes` iff any primary field is `unable`. Work independently and
return the completed CSV and your own attestation under the exact names in
RETURN_SCHEMA.json.
"""


def build_pack(
    *,
    breadth_manifest_path: Path,
    bbox_csv: Path,
    image_root: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("admission pack output must be a new empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, selected = select_frozen_pilot(breadth_manifest_path)
    selected_ids = {str(row["image_id"]) for row in selected}
    boxes = read_all_boxes(bbox_csv, selected_ids)
    nonbaseline = [name for name in SCIENCE_RENDER_IDS if name != SCIENCE_RENDER_IDS[0]]
    identity_ids = {
        str(row["image_id"])
        for row in sorted(
            selected, key=lambda row: _rank(seed, "identity", row["image_id"])
        )[:12]
    }

    clinical_rows: list[dict[str, Any]] = []
    sealed_pairs: list[dict[str, Any]] = []
    for row in selected:
        image_id = str(row["image_id"])
        dicom = image_root / f"{image_id}.dicom"
        if not dicom.is_file():
            raise FileNotFoundError(dicom)
        views = build_render_views(read_dicom_pixels(dicom), [], boxes.get(image_id, []))
        by_name = {str(view["name"]): view for view in views}
        comparison = list(nonbaseline)
        if image_id in identity_ids:
            comparison.append(IDENTITY_RENDER_ID)
        for transform in comparison:
            pair_id = _rank(seed, "render-pair", image_id, transform)[:20]
            baseline_left = int(_rank(seed, "render-side", pair_id), 16) % 2 == 0
            left_name, right_name = (
                (SCIENCE_RENDER_IDS[0], transform)
                if baseline_left
                else (transform, SCIENCE_RENDER_IDS[0])
            )
            left_rel = Path("images") / f"{pair_id}_A.png"
            right_rel = Path("images") / f"{pair_id}_B.png"
            left_hash = _save_png(by_name[left_name]["image"], output_dir / left_rel)
            right_hash = _save_png(by_name[right_name]["image"], output_dir / right_rel)
            clinical_rows.append(
                {
                    "pair_id": pair_id,
                    "image_A": str(left_rel),
                    "image_B": str(right_rel),
                    "same_support_state_for_all_14": "",
                    "visibility_change": "",
                    "listing_interchangeable": "",
                    "changed_finding_ids": "",
                    "unable_to_judge": "",
                    "comments": "",
                }
            )
            sealed_pairs.append(
                {
                    "pair_id": pair_id,
                    "image_id": image_id,
                    "sampling_stratum": row["sampling_stratum"],
                    "transform": transform,
                    "baseline_side": "A" if baseline_left else "B",
                    "left_render": left_name,
                    "right_render": right_name,
                    "left_png_sha256": left_hash,
                    "right_png_sha256": right_hash,
                    "transform_guard": by_name[transform]["audit"],
                }
            )
    clinical_rows.sort(key=lambda row: _rank(seed, "clinical-order", row["pair_id"]))
    for reviewer in (1, 2):
        _write_csv(
            output_dir / f"clinical_reviewer_{reviewer}.csv",
            list(clinical_rows[0]),
            clinical_rows,
        )

    prompts = prompt_texts()
    baseline_prompt = SCIENCE_PROMPT_IDS[0]
    prompt_pairs: list[dict[str, Any]] = []
    sealed_prompts: list[dict[str, Any]] = []
    candidates = list(SCIENCE_PROMPT_IDS[1:]) + ["exact_duplicate_control"]
    for candidate in candidates:
        candidate_prompt = baseline_prompt if candidate == "exact_duplicate_control" else candidate
        item_id = _rank(seed, "prompt-pair", candidate)[:20]
        baseline_left = int(_rank(seed, "prompt-side", item_id), 16) % 2 == 0
        left_id, right_id = (
            (baseline_prompt, candidate_prompt)
            if baseline_left
            else (candidate_prompt, baseline_prompt)
        )
        prompt_pairs.append(
            {
                "item_id": item_id,
                "wording_A": prompts[left_id],
                "wording_B": prompts[right_id],
                "same_target_ontology": "",
                "same_inclusion_obligation": "",
                "same_speech_act": "",
                "same_certainty_demand": "",
                "same_answer_space": "",
                "same_output_grammar": "",
                "unable_to_judge": "",
                "comments": "",
            }
        )
        sealed_prompts.append(
            {
                "item_id": item_id,
                "candidate_prompt_id": candidate,
                "reference_side": "A" if baseline_left else "B",
                "left_prompt_id": left_id,
                "right_prompt_id": right_id,
                "exact_duplicate_control": candidate == "exact_duplicate_control",
            }
        )
    prompt_pairs.sort(key=lambda row: _rank(seed, "prompt-order", row["item_id"]))
    for filename in ("clinical_template_reviewer.csv", "language_reviewer.csv"):
        _write_csv(output_dir / filename, list(prompt_pairs[0]), prompt_pairs)

    schema = return_schema()
    atomic_json(output_dir / "RETURN_SCHEMA.json", schema)
    (output_dir / "CLINICAL_INSTRUCTIONS.md").write_text(
        _instructions("clinical"), encoding="utf-8"
    )
    (output_dir / "PROMPT_INSTRUCTIONS.md").write_text(
        _instructions("prompt"), encoding="utf-8"
    )
    for role in ROLES:
        atomic_json(
            output_dir / f"{role}.attestation.template.json",
            {
                "protocol_id": VERSION,
                "review_role": role,
                "reviewer": {
                    "reviewer_id": "",
                    "professional_role": PROFESSIONAL_ROLE[role],
                    "independent_review": None,
                    "blinded_to_sealed_mapping": None,
                    "completed_at_utc": "",
                },
            },
        )
    atomic_json(
        output_dir / "sealed_mapping.json",
        {
            "version": VERSION,
            "warning": "Never deliver this file to a reviewer.",
            "clinical_pairs": sealed_pairs,
            "prompt_pairs": sealed_prompts,
        },
    )
    selected_path = output_dir / "selected_images.sealed.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    images = sorted((output_dir / "images").glob("*.png"))
    image_inventory = [
        {"name": f"images/{path.name}", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in images
    ]
    image_inventory_path = output_dir / "IMAGE_INVENTORY.json"
    atomic_json(image_inventory_path, image_inventory)
    guard_failures = [
        row["pair_id"]
        for row in sealed_pairs
        if not bool(row["transform_guard"].get("clinical_guard_pass"))
    ]
    fixed_files = [
        output_dir / "clinical_reviewer_1.csv",
        output_dir / "clinical_reviewer_2.csv",
        output_dir / "clinical_template_reviewer.csv",
        output_dir / "language_reviewer.csv",
        output_dir / "RETURN_SCHEMA.json",
        output_dir / "CLINICAL_INSTRUCTIONS.md",
        output_dir / "PROMPT_INSTRUCTIONS.md",
        output_dir / "sealed_mapping.json",
        selected_path,
        image_inventory_path,
        *(output_dir / f"{role}.attestation.template.json" for role in ROLES),
    ]
    manifest = {
        "version": VERSION,
        "status": "awaiting_four_independent_human_returns",
        "outcome_blind": True,
        "model_outputs_read": False,
        "model_scores_read": False,
        "gpu_used": False,
        "seed": seed,
        "source": {
            "breadth_manifest": str(breadth_manifest_path.resolve()),
            "breadth_manifest_sha256": sha256_file(breadth_manifest_path),
            "breadth_fingerprint": source_manifest["fingerprint"],
            "reference_sha256": source_manifest["reference_contract"]["reference_file_sha256"],
            "bbox_csv": str(bbox_csv.resolve()),
            "bbox_csv_sha256": sha256_file(bbox_csv),
            "image_root": str(image_root.resolve()),
            "renderer_source_sha256": sha256_file(
                Path(__file__).with_name("run_huatuo_dicom_render_pilot_v1.py")
            ),
            "prompt_source_sha256": sha256_file(
                Path(__file__).with_name("prepare_vindr_cecd_ontology_listing_v1.py")
            ),
            "builder_sha256": sha256_file(Path(__file__)),
        },
        "selection": {
            "policy": "entire pre-frozen breadth pilot; no second-stage sampling",
            "images": 60,
            "split": "pilot",
            "strata": {stratum: 20 for stratum in STRATA},
            "selection_sha256": canonical_hash(
                [(row["image_id"], row["sampling_stratum"]) for row in selected]
            ),
        },
        "clinical_review": {
            "reviewers": 2,
            "primary_render_pairs": 60 * len(nonbaseline),
            "identity_control_pairs": len(identity_ids),
            "pairs_total": len(clinical_rows),
            "all_14_findings_judged_jointly": True,
            "source_pixels_may_match_prior_pack": True,
            "prior_task_semantic_decisions_reused": False,
            "computational_guard_pass_pairs": len(sealed_pairs) - len(guard_failures),
            "computational_guard_fail_pairs": len(guard_failures),
            "computational_guard_failure_pair_ids_sha256": canonical_hash(
                sorted(guard_failures)
            ),
            "guard_failures_retained_for_blinded_clinical_review": True,
            "guard_failure_policy": (
                "engineering-invalid cells remain visible to reviewers but are "
                "ineligible for future complete-orbit model scoring; no threshold "
                "is relaxed after observing the failure"
            ),
        },
        "prompt_review": {
            "clinical_template_reviewers": 1,
            "language_reviewers": 1,
            "science_prompts": list(SCIENCE_PROMPT_IDS),
            "science_comparisons_to_baseline": 2,
            "exact_duplicate_controls": 1,
            "pairs_total": len(prompt_pairs),
            "prior_polar_prompt_admission_reused": False,
        },
        "roles": list(ROLES),
        "return_schema_version": RETURN_SCHEMA_VERSION,
        "images": {
            "pngs": len(images),
            "inventory": image_inventory_path.name,
            "inventory_sha256": sha256_file(image_inventory_path),
        },
        "artifact_sha256": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in fixed_files
        },
        "scientific_authorization": {
            "listing_render_equivalence_admitted": False,
            "listing_prompt_equivalence_admitted": False,
            "model_scoring_authorized": False,
            "gpu_authorized": False,
            "efficacy_claim_authorized": False,
        },
        "repository_policy": "restricted derived images and sealed truth remain outside Git/Git-LFS",
    }
    manifest["fingerprint"] = canonical_hash(manifest)
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breadth-manifest", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    result = build_pack(
        breadth_manifest_path=args.breadth_manifest,
        bbox_csv=args.bbox_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
