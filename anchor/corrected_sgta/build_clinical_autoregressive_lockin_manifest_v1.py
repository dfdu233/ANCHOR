#!/usr/bin/env python3
"""REJECTED F6 legacy builder for Clinical Autoregressive Lock-in v4.

The completed 600-answer Huatuo run is used only to document the two frozen
surface attractors.  Scientific rows come exclusively from the pre-existing
global ``dev`` hash split and are never selected using a development answer or
model score.  Each independent block contains a 0/3 anchor, a 3/3 anchor, and
one unused same-support swap for each anchor.  Consequently every DICOM occurs
in exactly one block and block bootstrap remains well defined.

The prefix+fixed-continuation construct was later rejected because several
serialized stimuli are not natural complete utterances.  The public builder
entry now fails before writing.  The remaining code is forensic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from corrected_sgta.clinical_claims import normalize_term
from corrected_sgta.prepare_vindr_reader_manifest import (
    build_records,
    load_ontology_findings,
    read_votes,
    select_ontology_columns,
    sha256_file,
    stable_key,
)
from corrected_sgta.prepare_vindr_reader_manifest_v2 import (
    fixed_panel_records,
    three_way_split,
)


PROTOCOL_ID = "clinical-autoregressive-lockin-manifest-v4-claim-specific-prompt"
F6_REJECTION_ID = "clinical-lockin-f6-unnatural-prefix-continuation-rejection-v1"
PILOT_PROTOCOL_ID = "clinical-presupposition-generation-only-v1"
FROZEN_PANEL = ("R8", "R9", "R10")
FROZEN_SEED = 42
FROZEN_SPLIT = "dev"
DEFAULT_BLOCKS_PER_FINDING = 12

# These are discovery facts frozen from the completed pilot, not fitted on dev.
# Every prefix stops before the embedded clinical claim.
FROZEN_CLAIMS: dict[str, dict[str, Any]] = {
    "pleural_effusion": {
        "pilot_condition": "existential",
        "display_name": "pleural effusion",
        "embedded_claim": "pleural effusion",
        "embedded_polarity": "present",
        "polarity_scope_note": "positive object of 'shows a right-sided'",
        "pilot_surface": "The chest X-ray shows a right-sided pleural effusion",
        "prefix_ladder": [
            "",
            "The chest X-ray ",
            "The chest X-ray shows ",
            "The chest X-ray shows a ",
            "The chest X-ray shows a right-sided ",
        ],
        "modifier_phase": ["a", "right-sided"],
    },
    "lung_opacity": {
        "pilot_condition": "negative_obligation",
        "display_name": "lung opacity",
        "embedded_claim": "opacity",
        "embedded_polarity": "present",
        "polarity_scope_note": (
            "positive existential 'However, there is a subtle opacity'; the earlier no-common-"
            "abnormalities clause is closed before However and is not the opacity polarity"
        ),
        "pilot_surface": "there is a subtle opacity",
        "prefix_ladder": [
            "",
            "This chest X-ray ",
            "This chest X-ray shows no common abnormalities ",
            (
                "This chest X-ray shows no common abnormalities such as consolidation, "
                "effusion or pneumothorax. "
            ),
            (
                "This chest X-ray shows no common abnormalities such as consolidation, "
                "effusion or pneumothorax. However, there is a subtle "
            ),
        ],
        "modifier_phase": ["However", "subtle"],
    },
}


class ManifestError(RuntimeError):
    """A frozen scientific input is missing, stale, or internally inconsistent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ManifestError(f"refusing to overwrite a different artifact: {path}")
        return
    _atomic_write(path, payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ManifestError(f"missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_pilot(pilot_dir: Path) -> dict[str, Any]:
    """Verify completed pilot provenance without assigning clinical truth."""

    config_path = pilot_dir / "generation_config.json"
    generations_path = pilot_dir / "generations.jsonl"
    summary_path = pilot_dir / "generation_summary.json"
    for path in (config_path, generations_path, summary_path):
        if not path.is_file():
            raise ManifestError(f"completed pilot artifact is missing: {path}")
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    rows = _load_jsonl(generations_path)
    if config.get("version") != PILOT_PROTOCOL_ID:
        raise ManifestError("pilot protocol is not the frozen generation-only v1")
    if config.get("split") != "pilot" or int(config.get("seed", -1)) != FROZEN_SEED:
        raise ManifestError("pilot split/seed differs from the frozen discovery run")
    if tuple(config.get("reader_panel", [])) != FROZEN_PANEL:
        raise ManifestError("pilot reader panel differs from exact R8/R9/R10")
    if len(rows) != 600 or summary.get("generations") != 600:
        raise ManifestError("pilot must be the completed 200-image x 3-condition run")
    identities = Counter((row.get("item_id"), row.get("prompt_condition")) for row in rows)
    if len(identities) != 600 or set(identities.values()) != {1}:
        raise ManifestError("pilot generation identities are incomplete or duplicated")
    if any(row.get("clinical_claim_evaluation_status") != "pending_shared_audit" for row in rows):
        raise ManifestError("pilot unexpectedly contains automatic clinical truth")
    expected_hash = summary.get("generations_sha256")
    if expected_hash != sha256_file(generations_path):
        raise ManifestError("pilot generation aggregate hash drifted")
    prompts: dict[str, str] = {}
    for condition in {str(specification["pilot_condition"]) for specification in FROZEN_CLAIMS.values()}:
        condition_rows = [row for row in rows if row["prompt_condition"] == condition]
        prompt_values = {str(row["prompt"]) for row in condition_rows}
        if len(condition_rows) != 200 or len(prompt_values) != 1:
            raise ManifestError(f"pilot {condition} prompt is incomplete or not fixed")
        prompts[condition] = prompt_values.pop()
    discovery_counts = {}
    discovery_ids = {}
    exact_top_surfaces = {}
    for finding, specification in FROZEN_CLAIMS.items():
        condition_rows = [
            row
            for row in rows
            if row["prompt_condition"] == specification["pilot_condition"]
        ]
        matches = [
            row
            for row in condition_rows
            if str(specification["pilot_surface"]).lower() in str(row["text"]).lower()
        ]
        discovery_counts[finding] = len(matches)
        discovery_ids[finding] = [str(row["image_id"]) for row in matches]
        top_text, top_count = Counter(str(row["text"]) for row in matches).most_common(1)[0]
        exact_top_surfaces[finding] = {
            "text": top_text,
            "count": top_count,
            "text_sha256": _sha(top_text.encode()),
            "embedded_polarity": specification["embedded_polarity"],
            "polarity_scope_note": specification["polarity_scope_note"],
        }
    if any(value <= 0 for value in discovery_counts.values()):
        raise ManifestError("a preregistered pilot surface is absent from the pilot")
    pilot_ids = {str(row["item_id"]) for row in rows}
    return {
        "pilot_dir": str(pilot_dir.resolve()),
        "generation_config_sha256": sha256_file(config_path),
        "generations_sha256": expected_hash,
        "generation_summary_sha256": sha256_file(summary_path),
        "pilot_fingerprint": config.get("fingerprint"),
        "pilot_image_ids": pilot_ids,
        "prompts_by_condition": prompts,
        "surface_discovery_counts": discovery_counts,
        "surface_discovery_image_ids": discovery_ids,
        "exact_top_surface_by_claim": exact_top_surfaces,
        "clinical_truth_used": False,
    }


def _fixed_candidates(labels_csv: Path, ontology: Path) -> list[dict[str, Any]]:
    votes, source_findings, _, _ = read_votes(labels_csv)
    selected, _ = select_ontology_columns(source_findings, load_ontology_findings(ontology))
    requested = set(FROZEN_CLAIMS)
    selected = [name for name in selected if normalize_term(name) in requested]
    if {normalize_term(name) for name in selected} != requested:
        raise ManifestError("VinDr/ontology does not contain both frozen findings")
    records, _ = build_records(votes, selected, "local-only")
    return fixed_panel_records(records, FROZEN_PANEL)


def _dev_candidates(labels_csv: Path, ontology: Path) -> list[dict[str, Any]]:
    fixed = _fixed_candidates(labels_csv, ontology)
    output = []
    for source in fixed:
        if three_way_split(str(source["image_id"]), FROZEN_SEED) != FROZEN_SPLIT:
            continue
        if int(source["positive_votes"]) not in {0, 3}:
            continue
        row = dict(source)
        row["experiment_split"] = FROZEN_SPLIT
        row["split_assignment"] = "global_image_sha256_20_20_60"
        row.pop("dicom_url", None)
        output.append(row)
    return output


def _select_blocks(
    candidates: Iterable[Mapping[str, Any]],
    *,
    image_root: Path,
    blocks_per_finding: int,
    pilot_ids: set[str],
) -> list[dict[str, Any]]:
    if blocks_per_finding < 4:
        raise ManifestError("at least four independent blocks per finding are required")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for source in candidates:
        image_id = str(source["image_id"])
        if image_id in pilot_ids:
            raise ManifestError("global split leakage: a dev image occurs in pilot")
        path = image_root / str(source["dicom_relpath"])
        if path.is_file():
            grouped[(str(source["finding"]), int(source["positive_votes"]))].append(
                dict(source)
            )
    used: set[str] = set()
    selected: dict[tuple[str, int], list[dict[str, Any]]] = {}
    # Rare groups choose first.  This order is frozen and uses only labels and
    # image IDs, never a development model output.
    keys = sorted(grouped, key=lambda key: (len(grouped[key]), key))
    needed = 2 * blocks_per_finding  # one anchor plus one same-support swap
    for key in keys:
        ordered = sorted(
            grouped[key],
            key=lambda row: stable_key(
                FROZEN_SEED,
                "clinical-autoregressive-lockin-dev-v1",
                key[0],
                str(key[1]),
                str(row["image_id"]),
            ),
        )
        available = [row for row in ordered if str(row["image_id"]) not in used]
        if len(available) < needed:
            raise ManifestError(
                f"insufficient unique dev DICOMs for {key}: need {needed}, found {len(available)}"
            )
        selected[key] = available[:needed]
        used.update(str(row["image_id"]) for row in selected[key])

    rows: list[dict[str, Any]] = []
    for finding in sorted(FROZEN_CLAIMS):
        negative = selected[(finding, 0)]
        positive = selected[(finding, 3)]
        negative_anchor, negative_same = negative[:blocks_per_finding], negative[blocks_per_finding:]
        positive_anchor, positive_same = positive[:blocks_per_finding], positive[blocks_per_finding:]
        claim = FROZEN_CLAIMS[finding]
        ladder = [
            {
                "step": index,
                "prefix": prefix,
                "prefix_utf8_sha256": _sha(prefix.encode()),
                "phase": (
                    "empty"
                    if index == 0
                    else "common_clinical_prefix"
                    if index <= 2
                    else "claim_specific_modifier"
                ),
                "claim_begins_after_prefix": True,
            }
            for index, prefix in enumerate(claim["prefix_ladder"])
        ]
        for index in range(blocks_per_finding):
            block_id = f"CAL-{finding}-{index:03d}"
            probe_role = "probe_fit" if index < blocks_per_finding // 2 else "probe_eval"
            for support, anchors, same, opposites in (
                (0, negative_anchor, negative_same, positive_anchor),
                (3, positive_anchor, positive_same, negative_anchor),
            ):
                anchor = anchors[index]
                same_row = same[index]
                opposite = opposites[index]
                sample_id = f"{block_id}-v{support}"
                polarity_prefix = (
                    f"Regarding {claim['display_name']} on this chest X-ray, "
                    "the finding is"
                )
                rows.append(
                    {
                        "manifest_protocol_id": PROTOCOL_ID,
                        "sample_id": sample_id,
                        "block_id": block_id,
                        "prompt_end_probe_role": probe_role,
                        "split": FROZEN_SPLIT,
                        "finding": finding,
                        "display_name": claim["display_name"],
                        "positive_votes": support,
                        "reader_count": 3,
                        "reader_panel": list(FROZEN_PANEL),
                        "reader_votes": anchor["reader_votes"],
                        "prompt_condition": claim["pilot_condition"],
                        "embedded_claim": claim["embedded_claim"],
                        "embedded_polarity": claim["embedded_polarity"],
                        "polarity_scope_note": claim["polarity_scope_note"],
                        "embedded_claim_utf8_sha256": _sha(claim["embedded_claim"].encode()),
                        "prefix_ladder": ladder,
                        "non_attractor_preclaim_template_control": {
                            "prefix": polarity_prefix,
                            "prefix_utf8_sha256": _sha(polarity_prefix.encode()),
                            "present_continuation": " present.",
                            "absent_continuation": " absent.",
                            "same_user_prompt": True,
                            "role": (
                                "claim_identity_preserving_non_attractor_teacher_forced_control; "
                                "never evidence for pre-response hidden polarity"
                            ),
                        },
                        "image_conditions": {
                            "original": _image_reference(anchor, image_root, support),
                            "same_support_swap": _image_reference(same_row, image_root, support),
                            "opposite_support_swap": _image_reference(opposite, image_root, 3 - support),
                        },
                        "selection_uses_dev_model_outputs": False,
                        "confirmation_split_touched": False,
                    }
                )
    rows.sort(key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate sample ID")
    referenced = [
        condition["image_id"]
        for row in rows
        for condition in row["image_conditions"].values()
    ]
    counts = Counter(referenced)
    # An anchor is original once and opposite once.  A same-support control is
    # referenced only once.  No DICOM may enter a second independent block.
    if set(counts.values()) != {1, 2}:
        raise AssertionError("unexpected DICOM reuse in independent blocks")
    image_blocks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for condition in row["image_conditions"].values():
            image_blocks[condition["image_id"]].add(row["block_id"])
    if any(len(blocks) != 1 for blocks in image_blocks.values()):
        raise AssertionError("a DICOM crosses independent block IDs")
    return rows


def _image_reference(source: Mapping[str, Any], image_root: Path, support: int) -> dict[str, Any]:
    image_id = str(source["image_id"])
    relative = str(source["dicom_relpath"])
    path = image_root / relative
    if not path.is_file():
        raise ManifestError(f"missing DICOM: {path}")
    actual_support = int(source["positive_votes"])
    if actual_support != support:
        raise AssertionError("pair construction changed reader support")
    return {
        "image_id": image_id,
        "dicom_relpath": relative,
        "dicom_sha256": sha256_file(path),
        "positive_votes": actual_support,
        "reader_count": 3,
    }


def build_manifest(
    *,
    labels_csv: Path,
    ontology: Path,
    image_root: Path,
    pilot_dir: Path,
    output_dir: Path,
    blocks_per_finding: int = DEFAULT_BLOCKS_PER_FINDING,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    raise ManifestError(
        f"{F6_REJECTION_ID}: v4 prefix+fixed-continuation stimuli are rejected; "
        "no manifest rebuild or GPU use is authorized"
    )
    # Unreachable historical implementation retained only so the exact v4
    # artifact can be reconstructed conceptually during failure analysis.
    pilot = audit_pilot(pilot_dir)
    discovery_ids = pilot.pop("surface_discovery_image_ids")
    vote_lookup = {
        (str(row["image_id"]), str(row["finding"])): int(row["positive_votes"])
        for row in _fixed_candidates(labels_csv, ontology)
    }
    pilot["surface_reader_vote_bins"] = {
        finding: dict(
            sorted(
                Counter(
                    f"{vote_lookup[(image_id, finding)]}/3"
                    for image_id in image_ids
                ).items()
            )
        )
        for finding, image_ids in discovery_ids.items()
    }
    candidates = _dev_candidates(labels_csv, ontology)
    rows = _select_blocks(
        candidates,
        image_root=image_root,
        blocks_per_finding=blocks_per_finding,
        pilot_ids=set(pilot.pop("pilot_image_ids")),
    )
    prompts = pilot["prompts_by_condition"]
    for row in rows:
        prompt = prompts[row["prompt_condition"]]
        row["prompt"] = prompt
        row["prompt_utf8_sha256"] = _sha(prompt.encode())
    manifest_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    metadata = {
        "manifest_protocol_id": PROTOCOL_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "exact_command": list(command or []),
        "status": "dev_frozen_gpu_not_run",
        "labels_csv": str(labels_csv.resolve()),
        "labels_csv_sha256": sha256_file(labels_csv),
        "ontology": str(ontology.resolve()),
        "ontology_sha256": sha256_file(ontology),
        "image_root": str(image_root.resolve()),
        "reader_panel": list(FROZEN_PANEL),
        "global_split_seed": FROZEN_SEED,
        "split": FROZEN_SPLIT,
        "pilot_discovery_only": pilot,
        "claims_frozen_before_dev_scoring": list(sorted(FROZEN_CLAIMS)),
        "blocks_per_finding": blocks_per_finding,
        "independent_blocks": len({row["block_id"] for row in rows}),
        "prompt_end_probe_block_split": {
            "unit": "independent_block",
            "fit_blocks_per_finding": blocks_per_finding // 2,
            "eval_blocks_per_finding": blocks_per_finding - blocks_per_finding // 2,
            "frozen_before_any_dev_activation": True,
        },
        "anchor_rows": len(rows),
        "unique_dicoms": len(
            {
                condition["image_id"]
                for row in rows
                for condition in row["image_conditions"].values()
            }
        ),
        "dev_model_output_used_for_selection": False,
        "confirmation_split_locked": True,
        "manifest_sha256": _sha(manifest_bytes),
        "manifest_filename": "dev_manifest.jsonl",
    }
    metadata["metadata_fingerprint"] = _sha(_canonical(metadata))
    _write_once(output_dir / "dev_manifest.jsonl", manifest_bytes)
    _write_once(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks-per-finding", type=int, default=DEFAULT_BLOCKS_PER_FINDING)
    args = parser.parse_args()
    result = build_manifest(
        labels_csv=args.labels_csv,
        ontology=args.ontology,
        image_root=args.image_root,
        pilot_dir=args.pilot_dir,
        output_dir=args.output_dir,
        blocks_per_finding=args.blocks_per_finding,
        command=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
