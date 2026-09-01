#!/usr/bin/env python3
"""CPU-only necessary-condition screen for calibration-free claim exchange.

The screen uses already generated MIMIC-CXR reports.  It deliberately gives a
frozen visual specialist an *optimistic* advantage: ground truth is used only
after generation to ask whether a false-positive draft claim can be paired
with an omitted true claim that the specialist ranks higher.  If this
single-specialist oracle headroom is small, requiring a second (VLM) ranking to
agree cannot rescue the method.

This is not a mitigation result and never changes the baseline queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from PIL import Image


VERSION = "pareto-claim-exchange-optimistic-headroom-v1"
LABELS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_patient(patient_id: str) -> str:
    value = int(hashlib.sha256(f"pce-v1:{patient_id}".encode()).hexdigest()[:8], 16)
    return "development" if value % 10 < 3 else "confirmation"


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_biomedclip(root: Path):
    config = json.loads((root / "open_clip_config.json").read_text())
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(root / "text_encoder")
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(root)
    model_name = "biomedclip_local_pareto_headroom_v1"
    _MODEL_CONFIGS[model_name] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(root / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    model.eval().to("cpu")
    return model, preprocess, get_tokenizer(model_name)


def encode_text(model, tokenizer) -> np.ndarray:
    prompts = []
    for label in LABELS:
        name = label.lower()
        prompts.extend(
            [
                f"a frontal chest radiograph showing {name}",
                f"a frontal chest radiograph without {name}",
            ]
        )
    with torch.inference_mode():
        features = model.encode_text(
            tokenizer(prompts, context_length=256), normalize=True
        ).cpu().numpy()
    return features.reshape(len(LABELS), 2, -1).astype(np.float32)


def encode_images(
    paths: list[Path], model, preprocess, batch_size: int
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch in chunks(paths, batch_size):
            tensors = []
            for path in batch:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            features = model.encode_image(torch.stack(tensors), normalize=True)
            for path, feature in zip(batch, features.cpu().numpy()):
                result[str(path)] = feature.astype(np.float32)
    return result


def maximum_strict_matches(fp_scores: list[float], fn_scores: list[float]) -> int:
    """Maximum one-to-one pairs satisfying score(fn) > score(fp)."""
    fps = sorted(fp_scores)
    fns = sorted(fn_scores)
    i = j = matched = 0
    while i < len(fps) and j < len(fns):
        if fns[j] > fps[i]:
            matched += 1
            i += 1
            j += 1
        else:
            j += 1
    return matched


def row_counts(row: dict[str, Any]) -> dict[str, int]:
    predicted = np.asarray(row["predicted"], dtype=np.int64)
    truth = np.asarray(row["truth"], dtype=np.int64)
    score = np.asarray(row["expert_scores"], dtype=np.float64)
    fp = np.flatnonzero((predicted == 1) & (truth == 0))
    fn = np.flatnonzero((predicted == 0) & (truth == 1))
    matched = maximum_strict_matches(score[fp].tolist(), score[fn].tolist())

    k = int(predicted.sum())
    expert_predicted = np.zeros_like(predicted)
    if k:
        order = np.argsort(-score, kind="stable")
        expert_predicted[order[:k]] = 1

    one_swap = predicted.copy()
    swap_made = 0
    swap_correct = 0
    swap_harmful = 0
    if 0 < k < predicted.size:
        selected = np.flatnonzero(predicted == 1)
        omitted = np.flatnonzero(predicted == 0)
        remove = int(selected[np.argmin(score[selected])])
        add = int(omitted[np.argmax(score[omitted])])
        if score[add] > score[remove]:
            one_swap[remove] = 0
            one_swap[add] = 1
            swap_made = 1
            swap_correct = int(truth[remove] == 0 and truth[add] == 1)
            swap_harmful = int(truth[remove] == 1 and truth[add] == 0)
    return {
        "fp": int(fp.size),
        "fn": int(fn.size),
        "optimistic_matches": int(matched),
        "expert_topk_fp": int(((expert_predicted == 1) & (truth == 0)).sum()),
        "expert_topk_fn": int(((expert_predicted == 0) & (truth == 1)).sum()),
        "one_swap_fp": int(((one_swap == 1) & (truth == 0)).sum()),
        "one_swap_fn": int(((one_swap == 0) & (truth == 1)).sum()),
        "swap_made": swap_made,
        "swap_correct": swap_correct,
        "swap_harmful": swap_harmful,
        "k": k,
        "truth_positive": int(truth.sum()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [row_counts(row) for row in rows]
    total = {key: int(sum(item[key] for item in counts)) for key in counts[0]}
    repairable = total["optimistic_matches"] / max(total["fp"], 1)
    return {
        "reports": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        **total,
        "optimistic_repairable_fp_fraction": repairable,
        "fixed_k_baseline_total_errors": total["fp"] + total["fn"],
        "fixed_k_expert_topk_total_errors": total["expert_topk_fp"]
        + total["expert_topk_fn"],
        "fixed_k_one_swap_total_errors": total["one_swap_fp"] + total["one_swap_fn"],
        "one_swap_relative_error_reduction": (
            (total["fp"] + total["fn"] - total["one_swap_fp"] - total["one_swap_fn"])
            / max(total["fp"] + total["fn"], 1)
        ),
    }


def bootstrap(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(row)
    patients = sorted(groups)
    rng = np.random.default_rng(seed)
    fractions = []
    one_swap_relative = []
    for _ in range(draws):
        sampled = rng.choice(patients, len(patients), replace=True)
        counts = [row_counts(row) for patient in sampled for row in groups[patient]]
        fp = sum(item["fp"] for item in counts)
        matches = sum(item["optimistic_matches"] for item in counts)
        if fp:
            fractions.append(matches / fp)
        baseline_errors = sum(item["fp"] + item["fn"] for item in counts)
        one_swap_errors = sum(item["one_swap_fp"] + item["one_swap_fn"] for item in counts)
        if baseline_errors:
            one_swap_relative.append((baseline_errors - one_swap_errors) / baseline_errors)
    values = np.asarray(fractions, dtype=np.float64)
    swap_values = np.asarray(one_swap_relative, dtype=np.float64)
    return {
        "cluster": "patient_id",
        "draws": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "one_swap_relative_error_reduction": {
            "mean": float(swap_values.mean()),
            "ci95": [
                float(np.quantile(swap_values, 0.025)),
                float(np.quantile(swap_values, 0.975)),
            ],
        },
    }


def load_model_rows(
    records_path: Path,
    pairs_path: Path,
    image_root: Path,
) -> list[dict[str, Any]]:
    pairs = {row["item_id"]: row for row in read_jsonl(pairs_path)}
    rows = []
    for record in read_jsonl(records_path):
        item_id = record["item_id"]
        pair = pairs[item_id]
        metrics = record["metrics"]
        predicted = metrics["chexbert_hypothesis_labels_14"][: len(LABELS)]
        truth = metrics["chexbert_reference_labels_14"][: len(LABELS)]
        image_path = (image_root / pair["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(
            {
                "item_id": item_id,
                "patient_id": str(pair["patient_id"]),
                "image_path": str(image_path),
                "predicted": [int(value) for value in predicted],
                "truth": [int(value) for value in truth],
                "split": split_for_patient(str(pair["patient_id"])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-records", type=Path, required=True)
    parser.add_argument("--huatuo-pairs", type=Path, required=True)
    parser.add_argument("--hulu-records", type=Path, required=True)
    parser.add_argument("--hulu-pairs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--biomedclip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' so the baseline GPU remains untouched.")
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 16)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "result.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)

    paths = {
        "huatuo": (args.huatuo_records, args.huatuo_pairs),
        "hulu": (args.hulu_records, args.hulu_pairs),
    }
    model_rows = {
        name: load_model_rows(records, pairs, args.image_root)
        for name, (records, pairs) in paths.items()
    }
    unique_paths = sorted(
        {Path(row["image_path"]) for rows in model_rows.values() for row in rows}
    )
    model, preprocess, tokenizer = load_biomedclip(args.biomedclip_root)
    text_features = encode_text(model, tokenizer)
    image_features = encode_images(unique_paths, model, preprocess, args.batch_size)
    for rows in model_rows.values():
        for row in rows:
            feature = image_features[row["image_path"]]
            row["expert_scores"] = [
                float(feature @ pair[0] - feature @ pair[1]) for pair in text_features
            ]

    analyses = {}
    passes = []
    for model_name, rows in model_rows.items():
        analyses[model_name] = {}
        for split in ("development", "confirmation"):
            current = [row for row in rows if row["split"] == split]
            analyses[model_name][split] = aggregate(current)
            if split == "confirmation":
                boot = bootstrap(current, args.bootstrap_draws, args.seed)
                analyses[model_name][split]["patient_bootstrap"] = boot
                point = analyses[model_name][split]["optimistic_repairable_fp_fraction"]
                one_swap = analyses[model_name][split]["one_swap_relative_error_reduction"]
                one_swap_ci = boot["one_swap_relative_error_reduction"]["ci95"]
                passes.append(
                    point >= 0.20
                    and boot["ci95"][0] > 0.10
                    and one_swap >= 0.05
                    and one_swap_ci[0] > 0.0
                )

    config = {
        "version": VERSION,
        "labels": LABELS,
        "score": "BiomedCLIP cosine(image, positive prompt) - cosine(image, negative prompt)",
        "split": "patient-hash 30/70 development/confirmation",
        "paths": {
            name: {
                "records": str(records),
                "records_sha256": sha256_file(records),
                "pairs": str(pairs),
                "pairs_sha256": sha256_file(pairs),
            }
            for name, (records, pairs) in paths.items()
        },
        "image_root": str(args.image_root.resolve()),
        "biomedclip_root": str(args.biomedclip_root.resolve()),
        "biomedclip_weights_sha256": sha256_file(
            args.biomedclip_root / "open_clip_pytorch_model.bin"
        ),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    result = {
        "status": "complete_cpu_optimistic_necessary_condition",
        "decision": "PASS" if all(passes) else "NO_GO",
        "decision_rule": (
            "For both Huatuo and Hulu confirmation splits, at least 20% of draft false-positive "
            "claims must admit a one-to-one omitted true claim with a strictly larger specialist "
            "score, and patient-bootstrap 95% CI lower bound must exceed 10%. In addition, the "
            "label-free steepest one-swap update must reduce fixed-K total errors by at least 5%, "
            "with patient-bootstrap 95% CI lower bound above zero."
        ),
        "claim_boundary": (
            "Ground truth is used only to measure optimistic headroom after generation. "
            "No correction method is evaluated and no test label is available to a decoder."
        ),
        "config": config,
        "analyses": analyses,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
