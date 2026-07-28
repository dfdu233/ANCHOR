"""Pre-validate source-style nuisance subspaces for modality-aware DG.

This script does not train a model and does not use yes/no logits as results.
It extracts frozen VLM visual features, fits low-rank source-domain mean
subspaces, and asks whether MIMIC/report images have appreciable energy in
those subspaces and whether that energy correlates with existing generated
answer errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_alignment import load_alignment_adapter

ImageFile.LOAD_TRUNCATED_IMAGES = True

VERSION = "anchor-style-nuisance-prevalidation-v1"

DEFAULT_SOURCE_SPECS = (
    "rule_iuxray:xray:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/rule_iuxray.json",
    "slake_xray:xray:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/slake_xray.json",
    "vqa_rad_train:mixed:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/vqa_rad_train.json",
)
DEFAULT_MIMIC_CE = "/root/autodl-tmp/ANCHOR/data/mimic_cxr_rule/questions.target.jsonl"
DEFAULT_MIMIC_RESULT = "/root/autodl-tmp/ANCHOR/results_reference/rule_mimic_source_margin/result.json"
DEFAULT_MIMIC_REPORT = "/root/autodl-tmp/ANCHOR/data/mmedrag/test/report/mimic_test.json"
DEFAULT_IUXRAY_REPORT = "/root/autodl-tmp/ANCHOR/data/mmedrag/test/report/iuxray_test.json"
DEFAULT_MIMIC_IMAGE_ROOT = "/root/autodl-tmp/MedHEval/images"
DEFAULT_IUXRAY_IMAGE_ROOT = "/root/autodl-tmp/MedHEval/images/IU-Xray"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def parse_source_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError("source spec must be domain:modality:path")
    domain, modality, path = parts
    return domain.strip(), modality.strip(), Path(path)


def unique_source_records(specs: list[str], max_per_domain: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        domain, modality, path = parse_source_spec(spec)
        rows = read_json_or_jsonl(path)
        candidates = []
        for row in rows:
            image = row.get("image") or row.get("image_path")
            if isinstance(image, list):
                image = image[0] if image else None
            if not image:
                continue
            image_path = Path(str(image))
            if not image_path.exists():
                continue
            key = str(image_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "id": str(row.get("id") or row.get("source_id") or key),
                    "domain": domain,
                    "modality": modality,
                    "split": "source",
                    "image_path": str(image_path.resolve()),
                }
            )
        candidates.sort(key=lambda item: stable_key(item["id"], seed))
        records.extend(candidates[:max_per_domain] if max_per_domain else candidates)
    return records


def mimic_ce_records(path: Path, image_root: Path, max_images: int, seed: int) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(path)
    by_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        image = str(row.get("image", "")).strip()
        if not image:
            continue
        full = image_root / image
        if not full.exists():
            continue
        key = str(full.resolve())
        by_image.setdefault(
            key,
            {
                "id": key,
                "domain": "mimic_ce",
                "modality": "xray",
                "split": "target_ce",
                "image_path": key,
            },
        )
    values = sorted(by_image.values(), key=lambda item: stable_key(item["id"], seed))
    return values[:max_images] if max_images else values


def report_records(path: Path, image_root: Path, domain: str, max_images: int, seed: int) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(path)
    records = []
    seen = set()
    for row in rows:
        images = row.get("image_path")
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list):
            continue
        for image in images[:1]:
            full = image_root / str(image)
            if not full.exists():
                continue
            key = str(full.resolve())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": str(row.get("id") or key),
                    "domain": domain,
                    "modality": "xray",
                    "split": "target_report",
                    "image_path": key,
                }
            )
    records.sort(key=lambda item: stable_key(item["id"], seed))
    return records[:max_images] if max_images else records


def load_error_by_image(result_path: Path) -> dict[str, dict[str, float]]:
    result = json.loads(result_path.read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result.get("records", []):
        image = str(row.get("image", ""))
        if image:
            grouped[image].append(row)
    out = {}
    for image, rows in grouped.items():
        n = len(rows)
        if not n:
            continue
        greedy_correct = [
            str(row.get("identity_greedy_pope", "")).strip().lower()
            == str(row.get("ground_truth", "")).strip().lower()
            for row in rows
        ]
        identity_correct = [
            str(row.get("identity_constrained", "")).strip().lower()
            == str(row.get("ground_truth", "")).strip().lower()
            for row in rows
        ]
        calibrated_correct = [
            str(row.get("calibrated", "")).strip().lower()
            == str(row.get("ground_truth", "")).strip().lower()
            for row in rows
        ]
        out[image] = {
            "n_questions": float(n),
            "greedy_error_rate": 1.0 - float(np.mean(greedy_correct)),
            "identity_error_rate": 1.0 - float(np.mean(identity_correct)),
            "calibrated_error_rate": 1.0 - float(np.mean(calibrated_correct)),
            "any_greedy_error": float(not all(greedy_correct)),
        }
    return out


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def extract_features(
    records: list[dict[str, Any]],
    model: str,
    output_npz: Path,
    output_meta: Path,
    batch_size: int,
    max_image_side: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if output_npz.exists() and output_meta.exists():
        meta = json.loads(output_meta.read_text())
        features = np.load(output_npz)["features"]
        if len(meta.get("records", [])) == features.shape[0]:
            return features.astype(np.float64), meta["records"]
    adapter = load_alignment_adapter(model)
    features = []
    try:
        for start in tqdm(range(0, len(records), batch_size), desc=f"features:{model}"):
            batch = records[start : start + batch_size]
            images = []
            for item in batch:
                with Image.open(item["image_path"]) as handle:
                    images.append(resize_image(handle.convert("RGB"), max_image_side))
            features.append(adapter.visual_features(images))
    finally:
        adapter.close()
    matrix = normalize_rows(np.concatenate(features, axis=0)).astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_npz.with_suffix(output_npz.suffix + ".tmp.npz")
    np.savez_compressed(tmp, features=matrix)
    tmp.replace(output_npz)
    metadata = {
        "version": VERSION,
        "model": model,
        "n": len(records),
        "dimension": int(matrix.shape[1]),
        "batch_size": batch_size,
        "max_image_side": max_image_side,
        "records": records,
    }
    output_meta.write_text(json.dumps(metadata, indent=2))
    return matrix.astype(np.float64), records


def fit_mean_subspace(features: np.ndarray, domains: list[str], rank: int | None = None) -> dict[str, Any]:
    domains_arr = np.asarray(domains)
    unique = sorted(set(domains))
    means = np.stack([features[domains_arr == domain].mean(axis=0) for domain in unique])
    global_mean = features.mean(axis=0)
    centered_means = means - global_mean
    _, singular, vt = np.linalg.svd(centered_means, full_matrices=False)
    maximum = max(1, min(len(unique) - 1, vt.shape[0]))
    chosen = maximum if rank is None else max(1, min(rank, maximum))
    basis = vt[:chosen]
    return {
        "domains": unique,
        "mean": global_mean,
        "basis": basis,
        "singular_values": singular.tolist(),
        "rank": int(chosen),
    }


def projection_energy(features: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = features - mean[None, :]
    denominator = np.sum(centered * centered, axis=1).clip(min=1e-12)
    projected = centered @ basis.T @ basis
    numerator = np.sum(projected * projected, axis=1)
    return numerator / denominator


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean_diff(x: np.ndarray, mask: np.ndarray, seed: int, n_boot: int = 1000) -> dict[str, float | list[float] | None]:
    true = x[mask]
    false = x[~mask]
    if len(true) < 2 or len(false) < 2:
        return {"mean_true": None, "mean_false": None, "diff": None, "ci95": None}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        diffs.append(float(rng.choice(true, len(true), replace=True).mean() - rng.choice(false, len(false), replace=True).mean()))
    return {
        "mean_true": float(true.mean()),
        "mean_false": float(false.mean()),
        "diff": float(true.mean() - false.mean()),
        "ci95": [float(v) for v in np.quantile(diffs, [0.025, 0.975])],
    }


def summarize_eta(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "fraction_gt_0_05": float(np.mean(values > 0.05)),
        "fraction_gt_0_10": float(np.mean(values > 0.10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava", choices=("llava", "hulu"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", default=list(DEFAULT_SOURCE_SPECS), help="domain:modality:path")
    parser.add_argument("--mimic-ce", type=Path, default=Path(DEFAULT_MIMIC_CE))
    parser.add_argument("--mimic-result", type=Path, default=Path(DEFAULT_MIMIC_RESULT))
    parser.add_argument("--mimic-report", type=Path, default=Path(DEFAULT_MIMIC_REPORT))
    parser.add_argument("--iuxray-report", type=Path, default=Path(DEFAULT_IUXRAY_REPORT))
    parser.add_argument("--mimic-image-root", type=Path, default=Path(DEFAULT_MIMIC_IMAGE_ROOT))
    parser.add_argument("--iuxray-image-root", type=Path, default=Path(DEFAULT_IUXRAY_IMAGE_ROOT))
    parser.add_argument("--max-source-per-domain", type=int, default=96)
    parser.add_argument("--max-mimic-ce-images", type=int, default=256)
    parser.add_argument("--max-report-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    source = unique_source_records(args.source, args.max_source_per_domain, args.seed)
    mimic_ce = mimic_ce_records(args.mimic_ce, args.mimic_image_root, args.max_mimic_ce_images, args.seed)
    mimic_report = report_records(args.mimic_report, args.mimic_image_root, "mimic_report", args.max_report_images, args.seed)
    iuxray_report = report_records(args.iuxray_report, args.iuxray_image_root, "iuxray_report", args.max_report_images, args.seed)
    records = source + mimic_ce + mimic_report + iuxray_report
    if not records:
        raise RuntimeError("no records found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, records = extract_features(
        records,
        args.model,
        args.output_dir / "features.npz",
        args.output_dir / "features.meta.json",
        args.batch_size,
        args.max_image_side,
    )
    source_idx = [i for i, row in enumerate(records) if row["split"] == "source"]
    xray_source_idx = [
        i for i, row in enumerate(records)
        if row["split"] == "source" and row.get("modality") == "xray"
    ]
    if len(set(records[i]["domain"] for i in source_idx)) < 3:
        raise RuntimeError("P_all needs at least three source domains")
    if len(set(records[i]["domain"] for i in xray_source_idx)) < 2:
        raise RuntimeError("P_xray needs at least two xray source domains")

    p_all = fit_mean_subspace(
        features[source_idx],
        [records[i]["domain"] for i in source_idx],
    )
    p_xray = fit_mean_subspace(
        features[xray_source_idx],
        [records[i]["domain"] for i in xray_source_idx],
    )
    eta_all = projection_energy(features, p_all["mean"], p_all["basis"])
    eta_xray = projection_energy(features, p_xray["mean"], p_xray["basis"])

    error_by_image = load_error_by_image(args.mimic_result)
    per_record = []
    for i, row in enumerate(records):
        item = {
            **row,
            "eta_all": float(eta_all[i]),
            "eta_xray": float(eta_xray[i]),
        }
        if row["split"] == "target_ce":
            rel = str(Path(row["image_path"]).relative_to(args.mimic_image_root))
            item.update(error_by_image.get(rel, {}))
        per_record.append(item)

    by_group: dict[str, dict[str, Any]] = {}
    for key_name, selector in {
        "source_all": lambda r: r["split"] == "source",
        "source_xray": lambda r: r["split"] == "source" and r["modality"] == "xray",
        "source_mixed": lambda r: r["split"] == "source" and r["modality"] != "xray",
        "target_mimic_ce": lambda r: r["split"] == "target_ce",
        "target_mimic_report": lambda r: r["domain"] == "mimic_report",
        "target_iuxray_report": lambda r: r["domain"] == "iuxray_report",
    }.items():
        idx = [j for j, row in enumerate(per_record) if selector(row)]
        by_group[key_name] = {
            "count": len(idx),
            "eta_all": summarize_eta(eta_all[idx]),
            "eta_xray": summarize_eta(eta_xray[idx]),
            "domains": dict(Counter(per_record[j]["domain"] for j in idx)),
        }

    ce_idx = [
        j for j, row in enumerate(per_record)
        if row["split"] == "target_ce" and "greedy_error_rate" in row
    ]
    ce_errors = np.asarray([per_record[j]["greedy_error_rate"] for j in ce_idx], dtype=np.float64)
    ce_any_error = np.asarray([per_record[j]["any_greedy_error"] > 0.5 for j in ce_idx], dtype=bool)
    correlation = {
        "n_images_with_error_labels": int(len(ce_idx)),
        "pearson_eta_all_vs_greedy_error_rate": safe_corr(eta_all[ce_idx], ce_errors),
        "pearson_eta_xray_vs_greedy_error_rate": safe_corr(eta_xray[ce_idx], ce_errors),
        "eta_all_any_error_minus_clean": bootstrap_mean_diff(eta_all[ce_idx], ce_any_error, args.seed),
        "eta_xray_any_error_minus_clean": bootstrap_mean_diff(eta_xray[ce_idx], ce_any_error, args.seed),
    }

    gate = {
        "eta_xray_mimic_median_gt_0_05": by_group["target_mimic_ce"]["eta_xray"].get("median", 0.0) > 0.05,
        "eta_xray_error_correlation_positive": (
            correlation["pearson_eta_xray_vs_greedy_error_rate"] is not None
            and correlation["pearson_eta_xray_vs_greedy_error_rate"] > 0
        ),
        "eta_xray_error_diff_ci_lower_positive": (
            correlation["eta_xray_any_error_minus_clean"]["ci95"] is not None
            and correlation["eta_xray_any_error_minus_clean"]["ci95"][0] > 0
        ),
    }
    gate["pass_for_generation_probe"] = bool(
        gate["eta_xray_mimic_median_gt_0_05"]
        and (
            gate["eta_xray_error_correlation_positive"]
            or gate["eta_xray_error_diff_ci_lower_positive"]
        )
    )

    payload = {
        "version": VERSION,
        "model": args.model,
        "elapsed_sec": time.time() - started,
        "config": {
            "sources": args.source,
            "max_source_per_domain": args.max_source_per_domain,
            "max_mimic_ce_images": args.max_mimic_ce_images,
            "max_report_images": args.max_report_images,
            "rank_policy": "domain mean-difference SVD; P_all rank<=D-1, P_xray rank<=D_xray-1",
            "no_yes_no_logits_as_results": True,
        },
        "input_sha256": {
            "mimic_ce": file_sha256(args.mimic_ce),
            "mimic_result": file_sha256(args.mimic_result),
            "mimic_report": file_sha256(args.mimic_report),
            "iuxray_report": file_sha256(args.iuxray_report),
        },
        "feature_shape": list(features.shape),
        "counts": {
            " | ".join(key): value
            for key, value in sorted(
                Counter((row["split"], row["domain"], row["modality"]) for row in records).items()
            )
        },
        "subspaces": {
            "P_all": {
                "domains": p_all["domains"],
                "rank": p_all["rank"],
                "singular_values": p_all["singular_values"],
            },
            "P_xray": {
                "domains": p_xray["domains"],
                "rank": p_xray["rank"],
                "singular_values": p_xray["singular_values"],
            },
        },
        "by_group": by_group,
        "mimic_ce_error_correlation": correlation,
        "gate": gate,
        "interpretation": (
            "If P_xray projection energy on MIMIC is weak or unrelated to generated-answer "
            "errors, source-style nuisance removal should not proceed to generation/training."
        ),
    }
    (args.output_dir / "style_nuisance_prevalidation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    detail_path = args.output_dir / "style_nuisance_records.jsonl"
    with detail_path.open("w") as handle:
        for row in per_record:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output_dir / "style_nuisance_prevalidation.json"), "gate": gate, "by_group": by_group, "correlation": correlation}, indent=2))


if __name__ == "__main__":
    main()
