#!/usr/bin/env python3
"""Evaluate pooled and source-barycenter preference adapters on source dev.

The evaluator uses the exact RULE MIMIC prompt and compares the complete
teacher-forced sequences ``Yes.`` and ``No.``.  It reports a common identity
baseline, the pooled control, the full source-function barycenter, and a
domain-wise leave-one-domain-out (LODO) barycenter.  It never reads target-test
labels or uses source-dev labels to modify a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_preference_gate import source_dev_gate
from corrected_sgta.rule_source_preference import (
    VERSION as TRAIN_VERSION,
    LinearLowRankResidual,
    SourceBarycenterResidual,
    canonical_binary_answer,
    file_sha256,
    rule_mimic_prompt,
    sequence_log_probability,
    stable_json_sha256,
    target_ids_from_labels,
)
from corrected_sgta.train_rule_dg_adapter import (
    build_teacher_forcing,
    canonical_answer,
    sequence_forward,
)
from corrected_sgta.train_rule_source_group_adapter import (
    normalize_source_rows,
    parse_named_paths,
)


VERSION = "rule-source-preference-barycenter-dev-eval-v1"


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
    parser.add_argument("--max-images-per-domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_source_preference.py"),
        Path(__file__).with_name("train_rule_source_preference_barycenter.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("train_rule_source_group_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def validate_dev_contract(
    manifest_path: Path,
    checkpoint: dict[str, Any],
    dev_jsons: dict[str, Path],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "rule-source-manifest-v2":
        raise ValueError("unsupported source manifest")
    checkpoint_contract = checkpoint.get("manifest_contract")
    if not isinstance(checkpoint_contract, dict):
        raise ValueError("checkpoint lacks source-manifest contract")
    if (
        checkpoint_contract.get("manifest_fingerprint")
        != manifest.get("fingerprint")
        or checkpoint_contract.get("manifest_sha256")
        != file_sha256(manifest_path)
    ):
        raise ValueError("checkpoint and source manifest differ")
    declared = manifest.get("outputs", {}).get("by_domain", {})
    if set(dev_jsons) != set(declared):
        raise ValueError("dev domains do not match source manifest")
    hashes = {}
    for domain, path in sorted(dev_jsons.items()):
        actual = file_sha256(path)
        if actual != declared[domain]["dev"]["json_sha256"]:
            raise ValueError(f"source-dev JSON hash mismatch for {domain}")
        hashes[domain] = actual
    if manifest.get("locked_test", {}).get("labels_read_for_selection") is not False:
        raise ValueError("source manifest does not seal target labels")
    return {
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest_sha256": file_sha256(manifest_path),
        "dev_json_sha256": hashes,
        "target_labels_read_for_selection": False,
    }


def summarize_predictions(
    records: dict[str, list[dict[str, Any]]],
    variants: list[str],
) -> dict[str, Any]:
    if not records or any(not rows for rows in records.values()):
        raise ValueError("every source-dev domain requires records")
    output: dict[str, Any] = {}
    for variant in variants:
        per_domain: dict[str, Any] = {}
        all_base: list[bool] = []
        all_variant: list[bool] = []
        for domain, rows in sorted(records.items()):
            base = np.asarray(
                [row["predictions"]["identity"] == row["ground_truth"] for row in rows],
                dtype=bool,
            )
            changed = np.asarray(
                [row["predictions"][variant] == row["ground_truth"] for row in rows],
                dtype=bool,
            )
            all_base.extend(base.tolist())
            all_variant.extend(changed.tolist())
            per_domain[domain] = {
                "n": len(rows),
                "identity_accuracy": float(base.mean()),
                "accuracy": float(changed.mean()),
                "delta_pp": float(100.0 * (changed.mean() - base.mean())),
                "rescues": int((~base & changed).sum()),
                "harms": int((base & ~changed).sum()),
            }
        base = np.asarray(all_base, dtype=bool)
        changed = np.asarray(all_variant, dtype=bool)
        domain_deltas = np.asarray(
            [item["delta_pp"] for item in per_domain.values()], dtype=np.float64
        )
        output[variant] = {
            "per_domain": per_domain,
            "micro": {
                "n": int(base.size),
                "identity_accuracy": float(base.mean()),
                "accuracy": float(changed.mean()),
                "delta_pp": float(100.0 * (changed.mean() - base.mean())),
                "rescues": int((~base & changed).sum()),
                "harms": int((base & ~changed).sum()),
            },
            "macro_delta_pp": float(domain_deltas.mean()),
            "minimum_domain_delta_pp": float(domain_deltas.min()),
        }
    return output


def _module_from_state(
    state: dict[str, torch.Tensor],
    width: int,
    rank: int,
    maximum: float,
    device: torch.device,
) -> LinearLowRankResidual:
    module = LinearLowRankResidual(width, rank, maximum).to(device)
    module.load_state_dict(state)
    module.eval()
    return module


def _score_candidate(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    answer: str,
    module: torch.nn.Module | None,
) -> float:
    input_ids, labels = build_teacher_forcing(adapter, prompt, answer)
    _, logits = sequence_forward(
        adapter,
        image,
        input_ids,
        labels,
        module,
        adapter_location="post",
    )
    return float(
        sequence_log_probability(logits, target_ids_from_labels(labels)).detach()
    )


def _predict(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    module: torch.nn.Module | None,
) -> tuple[str, dict[str, float]]:
    scores = {
        answer: _score_candidate(adapter, image, prompt, answer, module)
        for answer in ("Yes.", "No.")
    }
    prediction = max(("Yes.", "No."), key=lambda answer: scores[answer])
    return prediction, scores


def main() -> None:
    args = parse_args()
    if args.max_images_per_domain < 0:
        raise ValueError("max-images-per-domain must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    dev_jsons = parse_named_paths(args.dev_json, "--dev-json")
    dev_roots = parse_named_paths(args.dev_image_root, "--dev-image-root")
    if set(dev_jsons) != set(dev_roots) or len(dev_jsons) < 2:
        raise ValueError("dev JSON/root domains must match and contain >=2 domains")
    for path in [args.source_manifest, args.checkpoint, *dev_jsons.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in dev_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != TRAIN_VERSION:
        raise ValueError("unsupported preference checkpoint")
    if checkpoint.get("prompt_protocol") != "rule_mimic":
        raise ValueError("checkpoint did not use the exact RULE MIMIC prompt")
    contract = validate_dev_contract(
        args.source_manifest.resolve(), checkpoint, dev_jsons
    )
    selected = {
        domain: normalize_source_rows(
            domain,
            dev_jsons[domain],
            dev_roots[domain],
            args.max_images_per_domain,
            args.seed,
        )
        for domain in sorted(dev_jsons)
    }
    source_states = checkpoint.get("per_source_state_dict")
    if source_states is not None and set(source_states) != set(selected):
        raise ValueError("checkpoint source modules and source-dev domains differ")
    if source_states is None and checkpoint.get("pooled_state_dict") is None:
        raise ValueError("checkpoint contains no evaluable module")

    fingerprint_payload = {
        "version": VERSION,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_fingerprint": checkpoint.get("fingerprint"),
        "dev_contract": contract,
        "dev_image_roots": {
            name: str(path) for name, path in sorted(dev_roots.items())
        },
        "selected": {
            name: [
                {"id": row["id"], "image": row["image"]}
                for row in rows
            ]
            for name, rows in sorted(selected.items())
        },
        "max_images_per_domain": args.max_images_per_domain,
        "seed": args.seed,
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
    device = adapter.model.device
    width = int(checkpoint["width"])
    rank = int(checkpoint["rank"])
    maximum = float(checkpoint["max_relative_update"])
    modules: dict[str, torch.nn.Module] = {}
    if checkpoint.get("pooled_state_dict") is not None:
        modules["pooled"] = _module_from_state(
            checkpoint["pooled_state_dict"], width, rank, maximum, device
        )
    lodo_modules: dict[str, SourceBarycenterResidual] = {}
    if source_states is not None:
        modules["barycenter"] = SourceBarycenterResidual(
            source_states, maximum
        ).to(device)
        for domain in sorted(selected):
            included = sorted(set(selected) - {domain})
            lodo_modules[domain] = SourceBarycenterResidual(
                source_states, maximum, included_domains=included
            ).to(device)
        modules["barycenter"].eval()
        for module in lodo_modules.values():
            module.eval()

    variants = [*sorted(modules), *(["barycenter_lodo"] if lodo_modules else [])]
    records: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in selected
    }
    try:
        with torch.no_grad():
            for domain in sorted(selected):
                progress = tqdm(selected[domain], desc=f"preference-dev:{domain}")
                for row in progress:
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_mimic_prompt(row["question"])
                    ground_truth = canonical_binary_answer(
                        canonical_answer(row["answer"])
                    )
                    predictions: dict[str, str] = {}
                    scores: dict[str, dict[str, float]] = {}
                    predictions["identity"], scores["identity"] = _predict(
                        adapter, image, prompt, None
                    )
                    for name, module in sorted(modules.items()):
                        predictions[name], scores[name] = _predict(
                            adapter, image, prompt, module
                        )
                    if lodo_modules:
                        (
                            predictions["barycenter_lodo"],
                            scores["barycenter_lodo"],
                        ) = _predict(adapter, image, prompt, lodo_modules[domain])
                    records[domain].append(
                        {
                            "id": row["id"],
                            "image": row["image"],
                            "ground_truth": ground_truth,
                            "prompt": prompt,
                            "predictions": predictions,
                            "sequence_log_probabilities": scores,
                        }
                    )
    finally:
        adapter.close()

    summary = summarize_predictions(records, variants)
    gate = source_dev_gate(summary)
    atomic_json(
        args.output,
        {
            "version": VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "checkpoint": {
                "path": str(args.checkpoint),
                "version": checkpoint["version"],
                "fingerprint": checkpoint.get("fingerprint"),
                "training_mode": checkpoint.get("training_mode"),
                "pooled_aggregation": checkpoint.get("pooled_aggregation"),
            },
            "variants": variants,
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
