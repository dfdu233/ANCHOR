"""Paired multi-checkpoint source-dev NLL evaluation in one model process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.evaluate_rule_source_adapter_nll import (
    atomic_json,
    summarize_domain_records,
)
from corrected_sgta.infer_rule_dg_adapter import checkpoint_adapter_spec
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import (
    BoundedResidualBottleneck,
    build_teacher_forcing,
    file_sha256,
    rule_no_reference_prompt,
    sequence_forward,
)
from corrected_sgta.train_rule_source_group_adapter import (
    normalize_source_rows,
    parse_named_paths,
    stable_json_sha256,
)

VERSION = "rule-source-adapters-paired-dev-nll-v1"


def checkpoint_training_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("fingerprint_payload")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("config"), dict):
        raise ValueError("checkpoint lacks training fingerprint payload")
    config = dict(provenance["config"])
    objective = config.get("objective")
    if objective is None:
        raise ValueError("checkpoint training objective is missing")
    return {
        "checkpoint_version": payload.get("version"),
        "width": payload.get("width"),
        "next_step": payload.get("next_step"),
        "config": config,
        "source_json_sha256": provenance.get("source_json_sha256"),
        "source_names": provenance.get("source_names"),
        "selected": provenance.get("selected"),
        "train_test_image_overlap": provenance.get("train_test_image_overlap"),
        "training_code_sha256": provenance.get("code_sha256"),
    }


def _validate_selected_prefix(
    metadata: dict[str, dict[str, Any]]
) -> None:
    names = sorted(metadata)
    reference_domains = set(metadata[names[0]]["selected"] or {})
    for name in names:
        selected = metadata[name]["selected"]
        if not isinstance(selected, dict) or set(selected) != reference_domains:
            raise ValueError("source-scaling selected domains differ")
    for domain in sorted(reference_domains):
        sequences = sorted(
            ((name, metadata[name]["selected"][domain]) for name in names),
            key=lambda item: len(item[1]),
        )
        for (small_name, small), (large_name, large) in zip(sequences, sequences[1:]):
            if large[: len(small)] != small:
                raise ValueError(
                    f"source-scaling selected schedule is not a deterministic prefix "
                    f"for {domain}: {small_name} -> {large_name}"
                )


def validate_checkpoint_comparability(
    payloads: dict[str, dict[str, Any]],
    allow_config_diff: tuple[str, ...] | list[str] = (),
    selection_mode: str = "objective_only",
) -> dict[str, Any]:
    if len(payloads) < 2:
        raise ValueError("paired evaluation requires at least two checkpoints")
    allowed = set(allow_config_diff)
    supported = {"steps", "max_images_per_source"}
    if not allowed <= supported:
        raise ValueError(f"unsupported allowed config differences: {sorted(allowed - supported)}")
    if selection_mode not in {"objective_only", "source_scaling"}:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    if selection_mode == "objective_only" and allowed:
        raise ValueError("objective_only mode does not permit config differences")
    if selection_mode == "source_scaling" and not allowed:
        raise ValueError("source_scaling requires explicit --allow-config-diff")

    names = sorted(payloads)
    metadata = {name: checkpoint_training_metadata(payloads[name]) for name in names}
    strict_contracts: dict[str, dict[str, Any]] = {}
    objectives: dict[str, str] = {}
    scaling: dict[str, Any] = {}
    for name in names:
        current = metadata[name]
        config = dict(current["config"])
        objectives[name] = str(config.pop("objective"))
        operational_values = {"save_every": config.pop("save_every", None)}
        allowed_values = {key: config.pop(key, None) for key in sorted(allowed)}
        contract = {
            "checkpoint_version": current["checkpoint_version"],
            "width": current["width"],
            "config_except_allowed": config,
            "source_json_sha256": current["source_json_sha256"],
            "source_names": current["source_names"],
            "train_test_image_overlap": current["train_test_image_overlap"],
            "training_code_sha256": current["training_code_sha256"],
        }
        if "steps" not in allowed:
            contract["next_step"] = current["next_step"]
        if selection_mode != "source_scaling":
            contract["selected"] = current["selected"]
        strict_contracts[name] = contract
        selected = current["selected"] or {}
        scaling[name] = {
            "allowed_config_values": allowed_values,
            "operational_config_values": operational_values,
            "next_step": current["next_step"],
            "selected_sizes": {
                domain: len(rows) for domain, rows in sorted(selected.items())
            },
            "selected_sha256": stable_json_sha256(selected),
        }

    reference = strict_contracts[names[0]]
    for name in names[1:]:
        current = strict_contracts[name]
        if current != reference:
            differing = sorted(
                key for key in set(reference) | set(current)
                if reference.get(key) != current.get(key)
            )
            raise ValueError(
                f"checkpoint {name!r} is not comparable under {selection_mode}; "
                f"different fields={differing}"
            )
    if selection_mode == "source_scaling":
        _validate_selected_prefix(metadata)

    differences = ["objective", *sorted(allowed)]
    if selection_mode == "source_scaling":
        differences.append("selected_prefix_length")
        if "steps" in allowed:
            differences.append("next_step")
    return {
        "selection_mode": selection_mode,
        "reference_contract": reference,
        "contract_sha256": stable_json_sha256(reference),
        "objectives": objectives,
        "allowed_training_difference": differences,
        "recorded_nonsemantic_config": ["save_every"],
        "scaling_metadata": scaling,
    }


def summarize_paired_records(
    records: dict[str, list[dict[str, Any]]], adapter_names: list[str]
) -> dict[str, Any]:
    if not records or any(not values for values in records.values()):
        raise ValueError("every source domain requires at least one record")
    identity_per_domain: dict[str, dict[str, float | int]] = {}
    for domain, values in sorted(records.items()):
        identity = np.asarray([row["identity_nll"] for row in values], dtype=np.float64)
        if not np.isfinite(identity).all():
            raise FloatingPointError(f"non-finite identity NLL in {domain}")
        identity_per_domain[domain] = {
            "n": len(values), "identity_nll": float(identity.mean())
        }
    identity_means = np.asarray(
        [value["identity_nll"] for value in identity_per_domain.values()],
        dtype=np.float64,
    )
    adapters: dict[str, Any] = {}
    for adapter_name in adapter_names:
        converted: dict[str, list[dict[str, float]]] = {}
        for domain, values in records.items():
            converted[domain] = [{
                "identity_nll": float(row["identity_nll"]),
                "adapted_nll": float(row["adapters"][adapter_name]["adapted_nll"]),
            } for row in values]
        adapters[adapter_name] = summarize_domain_records(converted)
    return {
        "common_identity": {
            "per_domain": identity_per_domain,
            "macro_mean_nll": float(identity_means.mean()),
            "worst_domain_nll": float(identity_means.max()),
        },
        "adapters": adapters,
    }


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("evaluate_rule_source_adapter_nll.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("train_rule_source_group_adapter.py"),
        Path(__file__).with_name("infer_rule_dg_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dev-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--selection-mode", choices=("objective_only", "source_scaling"),
        default="objective_only",
    )
    parser.add_argument(
        "--allow-config-diff", action="append", default=[],
        choices=("steps", "max_images_per_source"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images-per-domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_images_per_domain < 0:
        raise ValueError("max-images-per-domain must be nonnegative")
    dev_jsons = parse_named_paths(args.dev_json, "--dev-json")
    dev_roots = parse_named_paths(args.dev_image_root, "--dev-image-root")
    checkpoints = parse_named_paths(args.checkpoint, "--checkpoint")
    if set(dev_jsons) != set(dev_roots) or not dev_jsons:
        raise ValueError("dev JSON/root names must match")
    if len(checkpoints) < 2:
        raise ValueError("at least two named checkpoints are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in [*dev_jsons.values(), *checkpoints.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in dev_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in sorted(checkpoints.items())
    }
    comparability = validate_checkpoint_comparability(
        payloads, tuple(args.allow_config_diff), args.selection_mode
    )
    module_specs = {
        name: checkpoint_adapter_spec(payload) for name, payload in payloads.items()
    }
    locations = {spec["location"] for spec in module_specs.values()}
    if len(locations) != 1:
        raise ValueError(f"checkpoint adapter locations differ: {locations}")
    adapter_location = locations.pop()

    selected = {
        name: normalize_source_rows(
            name, dev_jsons[name], dev_roots[name],
            args.max_images_per_domain, args.seed,
        )
        for name in sorted(dev_jsons)
    }
    fingerprint_payload = {
        "version": VERSION,
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in sorted(checkpoints.items())
        },
        "checkpoint_contract_sha256": comparability["contract_sha256"],
        "selection_mode": args.selection_mode,
        "allow_config_diff": sorted(args.allow_config_diff),
        "dev_json_sha256": {
            name: file_sha256(path) for name, path in sorted(dev_jsons.items())
        },
        "dev_image_roots": {
            name: str(path) for name, path in sorted(dev_roots.items())
        },
        "selected": {
            name: [{"id": row["id"], "image": row["image"]} for row in rows]
            for name, rows in sorted(selected.items())
        },
        "max_images_per_domain": args.max_images_per_domain,
        "seed": args.seed,
        "code_sha256": code_hashes(),
        "single_model_instance": True,
        "identity_evaluations_per_sample": 1,
        "target_domain_labels_accessed": False,
    }
    fingerprint = stable_json_sha256(fingerprint_payload)

    adapter_names = sorted(checkpoints)
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    modules: dict[str, BoundedResidualBottleneck] = {}
    for name, payload in payloads.items():
        spec = module_specs[name]
        module = BoundedResidualBottleneck(
            int(payload["width"]), spec["rank"], spec["max_relative_update"]
        ).to(adapter.model.device)
        module.load_state_dict(payload["state_dict"])
        module.eval()
        modules[name] = module

    records: dict[str, list[dict[str, Any]]] = {name: [] for name in selected}
    try:
        with torch.no_grad():
            for domain in sorted(selected):
                for row in tqdm(selected[domain], desc=f"paired-dev-nll-{domain}"):
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_no_reference_prompt(row["question"])
                    input_ids, labels = build_teacher_forcing(
                        adapter, prompt, row["answer"]
                    )
                    identity_nll, _ = sequence_forward(
                        adapter, image, input_ids, labels, None,
                        adapter_location=adapter_location,
                    )
                    identity_value = float(identity_nll)
                    if not math.isfinite(identity_value):
                        raise FloatingPointError(
                            f"non-finite identity NLL for {domain}:{row['id']}"
                        )
                    adapter_values: dict[str, dict[str, float]] = {}
                    for name in sorted(modules):
                        adapted_nll, _ = sequence_forward(
                            adapter, image, input_ids, labels, modules[name],
                            adapter_location=adapter_location,
                        )
                        adapted_value = float(adapted_nll)
                        if not math.isfinite(adapted_value):
                            raise FloatingPointError(
                                f"non-finite adapted NLL for {name}:{domain}:{row['id']}"
                            )
                        adapter_values[name] = {
                            "adapted_nll": adapted_value,
                            "excess_nll": adapted_value - identity_value,
                        }
                    records[domain].append({
                        "id": row["id"], "image": row["image"],
                        "identity_nll": identity_value,
                        "adapters": adapter_values,
                    })
    finally:
        modules.clear()
        adapter.close()
        torch.cuda.empty_cache()

    summary = summarize_paired_records(records, adapter_names)
    output = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "comparability": comparability,
        "checkpoint_metadata": {
            name: {
                "checkpoint_version": payload.get("version"),
                "checkpoint_fingerprint": payload.get("fingerprint"),
                "objective": module_specs[name]["objective"],
                "adapter_location": module_specs[name]["location"],
            }
            for name, payload in payloads.items()
        },
        "target_interface": (
            "complete assistant sequence NLL; reference text removed from question"
        ),
        "records": records,
        **summary,
    }
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output), "fingerprint": fingerprint,
        "common_identity": summary["common_identity"],
        "adapters": {
            name: value["macro_mean"] for name, value in summary["adapters"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
