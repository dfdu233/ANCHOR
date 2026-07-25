#!/usr/bin/env python3
"""Build exact-question, opposite-label source pairs with frozen visual features."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.models import LLAVA_PATH
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_dg_adapter_fingerprint_v3 import tree_identity
from corrected_sgta.rule_source_exact_pair import (
    PAIR_MANIFEST_VERSION,
    canonical_label,
    canonical_question,
    study_id,
)
from corrected_sgta.rule_source_preference import (
    file_sha256,
    stable_json_sha256,
)
from corrected_sgta.train_rule_source_group_adapter import load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for raw in load_rows(path):
        conversations = raw.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError(f"invalid conversations for {raw.get('id')!r}")
        image = Path(str(raw.get("image", ""))).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        row = {
            "id": str(raw.get("id", "")),
            "image": str(image),
            "source_domain": str(raw.get("source_domain", "")),
            "source_id": str(raw.get("source_id", "")),
            "image_sha256": str(raw.get("image_sha256", "")),
            "image_blob_sha256": str(raw.get("image_blob_sha256", "")),
            "question": str(conversations[0].get("value", "")).replace(
                "<image>", ""
            ).strip(),
            "canonical_question": canonical_question(
                conversations[0].get("value", "")
            ),
            "label": canonical_label(conversations[1].get("value", "")),
        }
        if not row["id"] or not row["source_domain"]:
            raise ValueError("source row lacks id/domain")
        actual = file_sha256(image)
        # ``image_sha256`` hashes decoded RGB pixels and deliberately differs
        # from the encoded-file hash for sources such as VQA-RAD JPEGs.
        expected_blob = row["image_blob_sha256"]
        if expected_blob and actual != expected_blob:
            raise ValueError(f"encoded image SHA mismatch for {image}")
        row["file_sha256"] = actual
        row["study_id"] = study_id(row)
        output.append(row)
    return output


def eligible_groups(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_question"]].append(row)
    output = {}
    for question, values in grouped.items():
        by_image: dict[str, set[str]] = defaultdict(set)
        for row in values:
            by_image[row["image_sha256"]].add(row["label"])
        conflicts = {
            image: labels for image, labels in by_image.items() if len(labels) > 1
        }
        if conflicts:
            raise ValueError(
                f"conflicting labels for canonical question {question!r}: "
                f"{sorted(conflicts)[:3]}"
            )
        deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(values, key=lambda item: item["id"]):
            deduplicated.setdefault(
                (row["image_sha256"], row["label"]), row
            )
        yes = [
            row for (_, label), row in deduplicated.items() if label == "yes"
        ]
        no = [
            row for (_, label), row in deduplicated.items() if label == "no"
        ]
        if yes and no:
            output[question] = {
                "yes": sorted(yes, key=lambda item: item["id"]),
                "no": sorted(no, key=lambda item: item["id"]),
            }
    if not output:
        raise ValueError("no exact-question opposite-label groups")
    return output


@torch.inference_mode()
def preprojector_feature(
    adapter: LlavaMedAlignmentAdapter, image: Image.Image
) -> np.ndarray:
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach())

    projector = adapter.model.get_model().mm_projector
    handle = projector.register_forward_pre_hook(hook)
    try:
        tensor = adapter._process_images([image])
        if isinstance(tensor, list):
            tensor = tensor[0].unsqueeze(0)
        adapter.model.encode_images(
            tensor.to(adapter.model.device, dtype=adapter.model.dtype)
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one projector input, captured {len(captured)}")
    feature = captured[0].float().mean(dim=1)[0]
    feature = torch.nn.functional.normalize(feature, dim=0)
    return feature.cpu().numpy()


def make_pairs(
    groups: dict[str, dict[str, list[dict[str, Any]]]],
    features: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    pairs = []
    for question in sorted(groups):
        yes = groups[question]["yes"]
        no = groups[question]["no"]
        similarity = np.full((len(yes), len(no)), -np.inf, dtype=np.float64)
        for i, positive in enumerate(yes):
            for j, negative in enumerate(no):
                same_rgb = (
                    positive["image_sha256"] == negative["image_sha256"]
                )
                same_study = (
                    positive["study_id"] is not None
                    and positive["study_id"] == negative["study_id"]
                    and positive["source_domain"] == negative["source_domain"]
                )
                if same_rgb or same_study:
                    continue
                similarity[i, j] = float(
                    features[positive["image_sha256"]]
                    @ features[negative["image_sha256"]]
                )
        valid_rows = np.isfinite(similarity).any(axis=1)
        valid_cols = np.isfinite(similarity).any(axis=0)
        if not valid_rows.any() or not valid_cols.any():
            continue
        row_map = np.flatnonzero(valid_rows)
        col_map = np.flatnonzero(valid_cols)
        reduced = similarity[np.ix_(row_map, col_map)]
        cost = np.where(np.isfinite(reduced), -reduced, 1e6)
        rr, cc = linear_sum_assignment(cost)
        for r_index, c_index in zip(rr.tolist(), cc.tolist()):
            i, j = int(row_map[r_index]), int(col_map[c_index])
            if not np.isfinite(similarity[i, j]):
                continue
            positive, negative = yes[i], no[j]
            pairs.append(
                {
                    "pair_id": stable_json_sha256(
                        {
                            "question": question,
                            "positive": positive["id"],
                            "negative": negative["id"],
                        }
                    )[:24],
                    "canonical_question": question,
                    "question": positive["question"],
                    "positive": positive,
                    "negative": negative,
                    "cosine_similarity": float(similarity[i, j]),
                }
            )
    if not pairs:
        raise ValueError("Hungarian matching produced no valid pairs")
    return sorted(pairs, key=lambda item: item["pair_id"])


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.source_manifest, args.train_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(args.source_manifest.read_text())
    outputs = source_manifest.get("outputs", {}).get("train", {})
    if outputs.get("json_sha256") != file_sha256(args.train_json):
        raise ValueError("train JSON does not match source manifest")
    rows = normalized_rows(args.train_json)
    groups = eligible_groups(rows)
    image_rows: dict[str, dict[str, Any]] = {}
    for values in groups.values():
        for label in ("yes", "no"):
            for row in values[label]:
                image_rows.setdefault(row["image_sha256"], row)

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    features: dict[str, np.ndarray] = {}
    try:
        for sha, row in tqdm(
            sorted(image_rows.items()), desc="preprojector-pair-features"
        ):
            with Image.open(row["image"]) as handle:
                features[sha] = preprojector_feature(
                    adapter, handle.convert("RGB")
                )
    finally:
        adapter.close()

    pairs = make_pairs(groups, features)
    feature_manifest = [
        {
            "image_sha256": sha,
            "feature_sha256": stable_json_sha256(
                np.asarray(value, dtype=np.float32).tolist()
            ),
            "dimension": int(value.shape[0]),
        }
        for sha, value in sorted(features.items())
    ]
    payload_without_fingerprint = {
        "version": PAIR_MANIFEST_VERSION,
        "canonicalizer": {
            "qualified_name": (
                "corrected_sgta.rule_source_exact_pair.canonical_question"
            ),
            "source_sha256": file_sha256(
                Path(__file__).with_name("rule_source_exact_pair.py")
            ),
        },
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": file_sha256(args.source_manifest),
            "fingerprint": source_manifest["fingerprint"],
        },
        "train_json": {
            "path": str(args.train_json.resolve()),
            "sha256": file_sha256(args.train_json),
            "rows": len(rows),
        },
        "frozen_visual_model": tree_identity(LLAVA_PATH),
        "feature_interface": (
            "mean_l2_normalized_input_to_mm_projector_after_frozen_vision_tower"
        ),
        "matching": (
            "per-canonical-question maximum-cosine one-to-one Hungarian; "
            "same-RGB and same-parsable-study excluded"
        ),
        "eligible": {
            "question_templates": len(groups),
            "rows": sum(
                len(values["yes"]) + len(values["no"])
                for values in groups.values()
            ),
            "unique_images": len(image_rows),
        },
        "features": feature_manifest,
        "pairs": pairs,
        "target_labels_accessed": False,
    }
    fingerprint = stable_json_sha256(payload_without_fingerprint)
    atomic_json(
        args.output,
        {"fingerprint": fingerprint, **payload_without_fingerprint},
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fingerprint": fingerprint,
                "eligible": payload_without_fingerprint["eligible"],
                "pairs": len(pairs),
                "pair_domains": {
                    "cross_domain": sum(
                        pair["positive"]["source_domain"]
                        != pair["negative"]["source_domain"]
                        for pair in pairs
                    ),
                    "same_domain": sum(
                        pair["positive"]["source_domain"]
                        == pair["negative"]["source_domain"]
                        for pair in pairs
                    ),
                },
                "cosine": {
                    "minimum": min(p["cosine_similarity"] for p in pairs),
                    "mean": sum(p["cosine_similarity"] for p in pairs)
                    / len(pairs),
                    "maximum": max(p["cosine_similarity"] for p in pairs),
                },
                "target_labels_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
