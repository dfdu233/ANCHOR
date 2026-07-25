"""Label-free CT/MRI native-support probe for exact LLaVA-Med source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from corrected_sgta.build_visual_centers_v2 import ordered_descriptors
from corrected_sgta.probe_llava_native_support import extract_features, normalize_rows
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.protocol_v2 import file_sha256, resolve_image
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import load_index, normalize_modality, sha256_file
from corrected_sgta.source_bank_v3 import verify_descriptor


VERSION = "llava-exact-modality-support-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("MODALITY", "INDEX"),
        required=True,
    )
    parser.add_argument("--target-dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-source", type=int, default=64)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-train-fraction", type=float, default=0.75)
    return parser.parse_args()


def deterministic_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def source_descriptors(path: Path, maximum: int, seed: int) -> list[dict]:
    rows = ordered_descriptors(load_index(path), seed)[:maximum]
    for row in rows:
        verify_descriptor(row)
    return rows


def unique_target_descriptors(
    path: Path,
    modality: str,
    maximum: int,
    seed: int,
) -> list[dict]:
    by_path = {}
    for row in json.loads(path.read_text()):
        if normalize_modality(row.get("modality")) != modality:
            continue
        image = resolve_image(row.get("img_name", ""))
        if image is None:
            continue
        resolved = str(image.resolve())
        candidate = {
            "qid": str(row["qid"]),
            "path": resolved,
            "img_name": row.get("img_name", ""),
        }
        old = by_path.get(resolved)
        if old is None or deterministic_key(seed, candidate["qid"]) < deterministic_key(
            seed, old["qid"]
        ):
            by_path[resolved] = candidate
    rows = sorted(
        by_path.values(), key=lambda row: deterministic_key(seed, row["path"])
    )
    return rows[:maximum] if maximum else rows


def superiority_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    greater = (positive[:, None] > negative[None, :]).sum()
    equal = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(positive) * len(negative)))


def metric_summary(source: np.ndarray, target: np.ndarray, train_fraction: float) -> dict:
    n_train = max(2, min(len(source) - 2, int(len(source) * train_fraction)))
    train = np.asarray(source[:n_train], dtype=np.float64)
    heldout = np.asarray(source[n_train:], dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    train_unit = normalize_rows(train)
    heldout_unit = normalize_rows(heldout)
    target_unit = normalize_rows(target)
    heldout_1nn = 1.0 - (heldout_unit @ train_unit.T).max(axis=1)
    target_1nn = 1.0 - (target_unit @ train_unit.T).max(axis=1)
    center = normalize_rows(train_unit.mean(axis=0, keepdims=True))[0]
    heldout_center = 1.0 - heldout_unit @ center
    target_center = 1.0 - target_unit @ center

    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    cumulative = np.cumsum(singular**2) / np.maximum((singular**2).sum(), 1e-12)
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
    basis = vt[:rank90].T

    def normal_residual(values: np.ndarray) -> np.ndarray:
        delta = values - mean
        tangent = (delta @ basis) @ basis.T
        return np.linalg.norm(delta - tangent, axis=1)

    heldout_normal = normal_residual(heldout)
    target_normal = normal_residual(target)

    def comparison(target_values: np.ndarray, heldout_values: np.ndarray) -> dict:
        heldout_median = float(np.median(heldout_values))
        target_median = float(np.median(target_values))
        return {
            "source_heldout_median": heldout_median,
            "target_median": target_median,
            "target_to_source_ratio": target_median / max(heldout_median, 1e-12),
            "target_vs_source_auc": superiority_auc(target_values, heldout_values),
        }

    local = comparison(target_1nn, heldout_1nn)
    gate = {
        "target_to_source_ratio_at_least_1_25": (
            local["target_to_source_ratio"] >= 1.25
        ),
        "target_vs_source_auc_at_least_0_75": (
            local["target_vs_source_auc"] >= 0.75
        ),
    }
    return {
        "n_source_train": len(train),
        "n_source_heldout": len(heldout),
        "n_target": len(target),
        "local_1nn_cosine": local,
        "global_center_cosine": comparison(target_center, heldout_center),
        "source_affine_normal": {
            "source_only_rank90": rank90,
            "source_train_explained_variance": float(cumulative[rank90 - 1]),
            **comparison(target_normal, heldout_normal),
        },
        "dg_gate": {"checks": gate, "pass": all(gate.values())},
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.source_train_fraction < 1.0:
        raise ValueError("source-train-fraction must lie in (0, 1)")
    specs = []
    for modality_raw, index_raw in args.source:
        modality = normalize_modality(modality_raw)
        if modality not in {"ct", "mri"}:
            raise ValueError(f"unsupported modality: {modality_raw}")
        specs.append((modality, Path(index_raw)))
    if len({modality for modality, _ in specs}) != len(specs):
        raise ValueError("duplicate source modality")

    config = {
        "version": VERSION,
        "sources": [
            {
                "modality": modality,
                "index": str(index.resolve()),
                "index_sha256": file_sha256(index),
            }
            for modality, index in specs
        ],
        "target_dataset": str(args.target_dataset.resolve()),
        "target_dataset_sha256": file_sha256(args.target_dataset),
        "max_source": args.max_source,
        "max_target": args.max_target,
        "batch_size": args.batch_size,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "source_train_fraction": args.source_train_fraction,
        "target_sampling": "unique resolved image path, sha256(seed:path)",
        "target_labels_used": False,
        "dg_gate": "local 1NN median ratio >=1.25 and target-vs-heldout AUROC >=0.75",
        "model_identity": model_identity("llava"),
    }
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis.json"
    if output.exists():
        old = json.loads(output.read_text())
        if old.get("fingerprint") != fingerprint:
            raise RuntimeError(f"output fingerprint mismatch: {output}")
        print(json.dumps(old, indent=2))
        return

    adapter = LlavaLocalSourceAdapter()
    arrays = {}
    ids = {}
    metrics = {}
    try:
        for modality, index in specs:
            source_rows = source_descriptors(index, args.max_source, args.seed)
            target_rows = unique_target_descriptors(
                args.target_dataset, modality, args.max_target, args.seed
            )
            source_raw, source_projected, source_ids = extract_features(
                adapter, source_rows, args.batch_size, args.max_image_side
            )
            target_raw, target_projected, target_ids = extract_features(
                adapter, target_rows, args.batch_size, args.max_image_side
            )
            arrays[f"{modality}_source_raw"] = source_raw
            arrays[f"{modality}_source_projected"] = source_projected
            arrays[f"{modality}_target_raw"] = target_raw
            arrays[f"{modality}_target_projected"] = target_projected
            ids[modality] = {"source": source_ids, "target": target_ids}
            metrics[modality] = {
                "raw_clip": metric_summary(
                    source_raw, target_raw, args.source_train_fraction
                ),
                "projected": metric_summary(
                    source_projected, target_projected, args.source_train_fraction
                ),
            }
    finally:
        adapter.close()

    arrays_path = args.output_dir / "features.npz"
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "fingerprint": fingerprint,
        "config": config,
        "features": str(arrays_path.resolve()),
        "features_sha256": sha256_file(arrays_path),
        "ids": ids,
        "metrics": metrics,
        "decision": {
            modality: {
                "pass": values["projected"]["dg_gate"]["pass"],
                "next": (
                    "run_single_aligned_view"
                    if values["projected"]["dg_gate"]["pass"]
                    else "stop_alignment_for_this_modality"
                ),
            }
            for modality, values in metrics.items()
        },
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
