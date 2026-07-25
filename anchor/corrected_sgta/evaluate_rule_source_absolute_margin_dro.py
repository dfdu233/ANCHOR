#!/usr/bin/env python3
"""Evaluate the frozen shared absolute-margin source-DRO adapter on source dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.evaluate_rule_source_preference_barycenter import (
    _module_from_state,
    _predict,
    summarize_predictions,
    validate_dev_contract,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_absolute_margin import (
    DEV_IMAGES_TOTAL,
    VERSION as TRAIN_VERSION,
    source_dro_dev_gate,
)
from corrected_sgta.rule_source_preference import (
    canonical_binary_answer,
    file_sha256,
    rule_mimic_prompt,
    stable_json_sha256,
)
from corrected_sgta.train_rule_dg_adapter import canonical_answer
from corrected_sgta.train_rule_source_group_adapter import (
    normalize_source_rows,
    parse_named_paths,
)


VERSION = "rule-source-absolute-margin-dro-dev-eval-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dev-json", action="append", required=True, metavar="DOMAIN=PATH"
    )
    parser.add_argument(
        "--dev-image-root",
        action="append",
        required=True,
        metavar="DOMAIN=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_source_absolute_margin.py"),
        Path(__file__).with_name("train_rule_source_absolute_margin_dro.py"),
        Path(__file__).with_name("rule_source_preference.py"),
        Path(__file__).with_name("evaluate_rule_source_preference_barycenter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dev_jsons = parse_named_paths(args.dev_json, "--dev-json")
    dev_roots = parse_named_paths(args.dev_image_root, "--dev-image-root")
    if set(dev_jsons) != set(dev_roots) or len(dev_jsons) != 3:
        raise ValueError("exactly three matching source-dev domains required")
    for path in [args.source_manifest, args.checkpoint, *dev_jsons.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in dev_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != TRAIN_VERSION:
        raise ValueError("unsupported absolute-margin checkpoint")
    if checkpoint.get("target_labels_accessed") is not False:
        raise ValueError("checkpoint does not seal target labels")
    contract = validate_dev_contract(
        args.source_manifest.resolve(), checkpoint, dev_jsons
    )
    selected = {
        domain: normalize_source_rows(
            domain, dev_jsons[domain], dev_roots[domain], 0, 42
        )
        for domain in sorted(dev_jsons)
    }
    if sum(map(len, selected.values())) != DEV_IMAGES_TOTAL:
        raise ValueError("frozen source-dev split must contain exactly 85 examples")

    fingerprint_payload = {
        "version": VERSION,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "dev_contract": contract,
        "dev_image_roots": {
            name: str(path) for name, path in sorted(dev_roots.items())
        },
        "selected": {
            name: [
                {
                    "id": row["id"],
                    "image": row["image"],
                    "image_sha256": file_sha256(Path(row["image"])),
                }
                for row in rows
            ]
            for name, rows in sorted(selected.items())
        },
        "prompt_protocol": "rule_mimic",
        "prediction_interface": "argmax_complete_yes_no_sequence_log_probability",
        "code_sha256": code_hashes(),
        "target_test_labels_accessed": False,
    }
    fingerprint = stable_json_sha256(fingerprint_payload)

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = _module_from_state(
        checkpoint["state_dict"],
        int(checkpoint["width"]),
        int(checkpoint["rank"]),
        float(checkpoint["max_relative_update"]),
        adapter.model.device,
    )
    records: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in selected
    }
    try:
        with torch.no_grad():
            for domain in sorted(selected):
                for row in tqdm(selected[domain], desc=f"absolute-dro-dev:{domain}"):
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_mimic_prompt(row["question"])
                    ground_truth = canonical_binary_answer(
                        canonical_answer(row["answer"])
                    )
                    identity, identity_scores = _predict(
                        adapter, image, prompt, None
                    )
                    source_dro, source_dro_scores = _predict(
                        adapter, image, prompt, module
                    )
                    records[domain].append(
                        {
                            "id": row["id"],
                            "image": row["image"],
                            "ground_truth": ground_truth,
                            "prompt": prompt,
                            "predictions": {
                                "identity": identity,
                                "source_dro": source_dro,
                            },
                            "sequence_log_probabilities": {
                                "identity": identity_scores,
                                "source_dro": source_dro_scores,
                            },
                        }
                    )
    finally:
        adapter.close()

    summary = summarize_predictions(records, ["source_dro"])
    gate = source_dro_dev_gate(summary)
    atomic_json(
        args.output,
        {
            "version": VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": file_sha256(args.checkpoint),
                "version": checkpoint["version"],
                "fingerprint": checkpoint["fingerprint"],
            },
            "summary": summary,
            "source_dev_gate": gate,
            "records": records,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fingerprint": fingerprint,
                "summary": summary,
                "source_dev_gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
