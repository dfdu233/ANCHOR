"""Read-only source-dev NLL audit for a RULE visual-projector adapter.

This evaluator intentionally accepts source development domains only.  It has no
target-test argument and never selects a method using target-domain labels.
"""

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

VERSION = "rule-source-adapter-dev-nll-eval-v1"


def summarize_domain_records(
    records: dict[str, list[dict[str, float]]]
) -> dict[str, Any]:
    if not records or any(not values for values in records.values()):
        raise ValueError("every source domain requires at least one record")
    per_domain: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(records.items()):
        identity = np.asarray([row["identity_nll"] for row in values], dtype=np.float64)
        adapted = np.asarray([row["adapted_nll"] for row in values], dtype=np.float64)
        excess = adapted - identity
        if not all(np.isfinite(item).all() for item in (identity, adapted, excess)):
            raise FloatingPointError(f"non-finite NLL in source domain {name}")
        per_domain[name] = {
            "n": len(values),
            "identity_nll": float(identity.mean()),
            "adapted_nll": float(adapted.mean()),
            "excess_nll": float(excess.mean()),
        }
    domain_names = list(per_domain)
    identity_means = np.asarray(
        [per_domain[name]["identity_nll"] for name in domain_names], dtype=np.float64
    )
    adapted_means = np.asarray(
        [per_domain[name]["adapted_nll"] for name in domain_names], dtype=np.float64
    )
    excess_means = np.asarray(
        [per_domain[name]["excess_nll"] for name in domain_names], dtype=np.float64
    )
    return {
        "per_domain": per_domain,
        "macro_mean": {
            "identity_nll": float(identity_means.mean()),
            "adapted_nll": float(adapted_means.mean()),
            "excess_nll": float(excess_means.mean()),
        },
        "worst_domain": {
            "identity_nll": float(identity_means.max()),
            "adapted_nll": float(adapted_means.max()),
            "excess_nll": float(excess_means.max()),
            "excess_domain": domain_names[int(excess_means.argmax())],
        },
    }


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("train_rule_source_group_adapter.py"),
        Path(__file__).with_name("infer_rule_dg_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--dev-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    if set(dev_jsons) != set(dev_roots) or not dev_jsons:
        raise ValueError("dev JSON/root names must match")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in dev_jsons.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in dev_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)

    selected = {
        name: normalize_source_rows(
            name, dev_jsons[name], dev_roots[name],
            args.max_images_per_domain, args.seed,
        )
        for name in sorted(dev_jsons)
    }
    checkpoint_sha256 = file_sha256(args.checkpoint)
    fingerprint_payload = {
        "version": VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "dev_json_sha256": {
            name: file_sha256(path) for name, path in sorted(dev_jsons.items())
        },
        "dev_image_roots": {
            name: str(path) for name, path in sorted(dev_roots.items())
        },
        "selected": {
            name: [
                {"id": row["id"], "image": row["image"]} for row in rows
            ]
            for name, rows in sorted(selected.items())
        },
        "max_images_per_domain": args.max_images_per_domain,
        "seed": args.seed,
        "code_sha256": code_hashes(),
        "target_domain_labels_accessed": False,
    }
    fingerprint = stable_json_sha256(fingerprint_payload)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    module_spec = checkpoint_adapter_spec(payload)
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    module = BoundedResidualBottleneck(
        int(payload["width"]), module_spec["rank"],
        module_spec["max_relative_update"],
    ).to(adapter.model.device)
    module.load_state_dict(payload["state_dict"])
    module.eval()

    records: dict[str, list[dict[str, Any]]] = {name: [] for name in selected}
    try:
        with torch.no_grad():
            for name in sorted(selected):
                for row in tqdm(selected[name], desc=f"dev-nll-{name}"):
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_no_reference_prompt(row["question"])
                    input_ids, labels = build_teacher_forcing(
                        adapter, prompt, row["answer"]
                    )
                    identity_nll, _ = sequence_forward(
                        adapter, image, input_ids, labels, None,
                        adapter_location=module_spec["location"],
                    )
                    adapted_nll, _ = sequence_forward(
                        adapter, image, input_ids, labels, module,
                        adapter_location=module_spec["location"],
                    )
                    identity_value = float(identity_nll)
                    adapted_value = float(adapted_nll)
                    if not math.isfinite(identity_value) or not math.isfinite(adapted_value):
                        raise FloatingPointError(f"non-finite NLL for {name}:{row['id']}")
                    records[name].append({
                        "id": row["id"],
                        "image": row["image"],
                        "identity_nll": identity_value,
                        "adapted_nll": adapted_value,
                        "excess_nll": adapted_value - identity_value,
                    })
    finally:
        del module
        adapter.close()
        torch.cuda.empty_cache()

    summary = summarize_domain_records(records)
    output = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "checkpoint_metadata": {
            "checkpoint_version": payload.get("version"),
            "checkpoint_fingerprint": payload.get("fingerprint"),
            "objective": module_spec["objective"],
            "adapter_location": module_spec["location"],
        },
        "target_interface": "complete assistant sequence NLL; reference text removed from question",
        "records": records,
        **summary,
    }
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output), "fingerprint": fingerprint,
        "macro_mean": summary["macro_mean"],
        "worst_domain": summary["worst_domain"],
    }, indent=2))


if __name__ == "__main__":
    main()
