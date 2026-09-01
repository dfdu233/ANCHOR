#!/usr/bin/env python3
"""CPU-only admission screen for an independent visual evidence source.

This is deliberately *not* a mitigation result.  It asks whether a frozen
BiomedCLIP score adds held-out, image-level label information beyond the final
VLM claim margin.  A failure closes the cheap proxy gate for the proposed
reciprocal-score experiment; a pass only authorizes the much smaller diffusion
gradient experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from anchor.corrected_sgta.sith_vindr_activation_probe_v1 import render_dicom


PROTOCOL_VERSION = "external-visual-increment-biomedclip-v1"
FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
DISPLAY = {
    "aortic_enlargement": "aortic enlargement",
    "cardiomegaly": "cardiomegaly",
    "lung_opacity": "lung opacity",
    "nodule_mass": "a pulmonary nodule or mass",
    "pleural_effusion": "pleural effusion",
    "pleural_thickening": "pleural thickening",
    "pulmonary_fibrosis": "pulmonary fibrosis",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final_margin(row: dict[str, Any]) -> float:
    lens = row["diagnostic_plain_logit_lens"]
    layer = max(lens, key=lambda value: int(value))
    return float(lens[layer]["supported"] - lens[layer]["refuted"])


def load_claims(path: Path, split: str, model: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row["finding"] not in FINDINGS or int(row["positive_votes"]) not in (0, 3):
            continue
        rows.append(
            {
                "image_id": row["image_id"],
                "finding": row["finding"],
                "label": int(row["positive_votes"] == 3),
                "margin": final_margin(row),
                "split": split,
                "model": model,
            }
        )
    return rows


def load_biomedclip(root: Path):
    config = json.loads((root / "open_clip_config.json").read_text())
    config["model_cfg"]["text_cfg"]["hf_model_name"] = str(root / "text_encoder")
    config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(root)
    model_name = "biomedclip_local_external_increment_v1"
    _MODEL_CONFIGS[model_name] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(root / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    model.eval().to("cpu")
    tokenizer = get_tokenizer(model_name)
    return model, preprocess, tokenizer


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def encode_images(
    image_ids: list[str],
    image_root: Path,
    model,
    preprocess,
    batch_size: int,
) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch_ids in chunks(image_ids, batch_size):
            tensors = [
                preprocess(render_dicom(image_root / f"{image_id}.dicom"))
                for image_id in batch_ids
            ]
            features = model.encode_image(torch.stack(tensors), normalize=True)
            for image_id, feature in zip(batch_ids, features.cpu().numpy()):
                encoded[image_id] = feature.astype(np.float32)
    return encoded


def encode_prompts(model, tokenizer) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prompts: list[str] = []
    order: list[tuple[str, str]] = []
    for finding in FINDINGS:
        name = DISPLAY[finding]
        prompts.extend(
            [
                f"a frontal chest radiograph showing {name}",
                f"a frontal chest radiograph without {name}",
            ]
        )
        order.extend([(finding, "positive"), (finding, "negative")])
    tokens = tokenizer(prompts, context_length=256)
    with torch.inference_mode():
        features = model.encode_text(tokens, normalize=True).cpu().numpy()
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, feature in zip(order, features):
        result[key[0]][key[1]] = feature.astype(np.float32)
    return {finding: (values["positive"], values["negative"]) for finding, values in result.items()}


def attach_clip_scores(
    rows: list[dict[str, Any]],
    images: dict[str, np.ndarray],
    texts: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for row in rows:
        positive, negative = texts[row["finding"]]
        image = images[row["image_id"]]
        row["clip_score"] = float(image @ positive - image @ negative)


def design_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[float, float]]]:
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for finding in FINDINGS:
        current = [row for row in rows if row["finding"] == finding]
        stats[finding] = {}
        for key in ("margin", "clip_score"):
            values = np.asarray([row[key] for row in current], dtype=np.float64)
            scale = float(values.std())
            stats[finding][key] = (float(values.mean()), scale if scale > 1e-8 else 1.0)
    return stats


def design(rows: list[dict[str, Any]], stats, enhanced: bool) -> np.ndarray:
    matrix = []
    for row in rows:
        finding = row["finding"]
        one_hot = [float(finding == name) for name in FINDINGS[:-1]]
        margin_mean, margin_scale = stats[finding]["margin"]
        values = one_hot + [(row["margin"] - margin_mean) / margin_scale]
        if enhanced:
            clip_mean, clip_scale = stats[finding]["clip_score"]
            values.append((row["clip_score"] - clip_mean) / clip_scale)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


def macro_auc(rows: list[dict[str, Any]], scores: np.ndarray) -> float:
    aucs = []
    for finding in FINDINGS:
        indices = [index for index, row in enumerate(rows) if row["finding"] == finding]
        labels = np.asarray([rows[index]["label"] for index in indices])
        if len(np.unique(labels)) == 2:
            aucs.append(roc_auc_score(labels, scores[indices]))
    return float(np.mean(aucs))


def metrics(rows: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    by_finding = {}
    for finding in FINDINGS:
        indices = [index for index, row in enumerate(rows) if row["finding"] == finding]
        y = labels[indices]
        p = probabilities[indices]
        by_finding[finding] = {
            "n": len(indices),
            "auroc": float(roc_auc_score(y, p)),
            "nll": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
        }
    return {
        "n": len(rows),
        "unique_images": len({row["image_id"] for row in rows}),
        "macro_auroc": macro_auc(rows, probabilities),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "by_finding": by_finding,
    }


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    enhanced: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["image_id"]].append(index)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    auc_delta, nll_improvement, brier_improvement = [], [], []
    labels_all = np.asarray([row["label"] for row in rows], dtype=np.int64)
    for _ in range(draws):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        indices = np.asarray([idx for image_id in sampled for idx in groups[image_id]])
        sampled_rows = [rows[idx] for idx in indices]
        y = labels_all[indices]
        base = baseline[indices]
        extra = enhanced[indices]
        try:
            auc_delta.append(macro_auc(sampled_rows, extra) - macro_auc(sampled_rows, base))
        except ValueError:
            continue
        nll_improvement.append(log_loss(y, base, labels=[0, 1]) - log_loss(y, extra, labels=[0, 1]))
        brier_improvement.append(brier_score_loss(y, base) - brier_score_loss(y, extra))

    def summary(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }

    return {
        "draws_requested": draws,
        "draws_valid": len(auc_delta),
        "macro_auroc_delta": summary(auc_delta),
        "nll_improvement": summary(nll_improvement),
        "brier_improvement": summary(brier_improvement),
    }


def analyze_model(
    development: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    stats = design_stats(development)
    y_dev = np.asarray([row["label"] for row in development], dtype=np.int64)
    baseline_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    enhanced_model = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    baseline_model.fit(design(development, stats, enhanced=False), y_dev)
    enhanced_model.fit(design(development, stats, enhanced=True), y_dev)
    base_prob = baseline_model.predict_proba(design(confirmation, stats, enhanced=False))[:, 1]
    extra_prob = enhanced_model.predict_proba(design(confirmation, stats, enhanced=True))[:, 1]
    clip_raw = np.asarray([row["clip_score"] for row in confirmation], dtype=np.float64)
    base_metrics = metrics(confirmation, base_prob)
    extra_metrics = metrics(confirmation, extra_prob)
    return {
        "development_n": len(development),
        "confirmation_n": len(confirmation),
        "baseline": base_metrics,
        "enhanced": extra_metrics,
        "clip_standalone_macro_auroc": macro_auc(confirmation, clip_raw),
        "point_deltas": {
            "macro_auroc": extra_metrics["macro_auroc"] - base_metrics["macro_auroc"],
            "nll_improvement": base_metrics["nll"] - extra_metrics["nll"],
            "brier_improvement": base_metrics["brier"] - extra_metrics["brier"],
        },
        "image_cluster_bootstrap": cluster_bootstrap(
            confirmation, base_prob, extra_prob, draws=draws, seed=seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--biomedclip-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise RuntimeError("This admission screen must run CPU-only with CUDA_VISIBLE_DEVICES=''.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "result.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force explicitly.")

    sources = {
        "huatuo": {
            "development": args.huatuo_dev,
            "confirmation": args.huatuo_confirmation,
        },
        "hulu": {
            "development": args.hulu_dev,
            "confirmation": args.hulu_confirmation,
        },
    }
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_rows: list[dict[str, Any]] = []
    for model, paths in sources.items():
        rows[model] = {}
        for split, path in paths.items():
            current = load_claims(path, split, model)
            rows[model][split] = current
            all_rows.extend(current)

    image_ids = sorted({row["image_id"] for row in all_rows})
    missing = [image_id for image_id in image_ids if not (args.image_root / f"{image_id}.dicom").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} DICOM files; first={missing[0]}")

    model, preprocess, tokenizer = load_biomedclip(args.biomedclip_root)
    image_features = encode_images(image_ids, args.image_root, model, preprocess, args.batch_size)
    text_features = encode_prompts(model, tokenizer)
    for model_rows in rows.values():
        for split_rows in model_rows.values():
            attach_clip_scores(split_rows, image_features, text_features)

    analyses = {
        model_name: analyze_model(
            model_rows["development"],
            model_rows["confirmation"],
            draws=args.bootstrap_draws,
            seed=args.seed,
        )
        for model_name, model_rows in rows.items()
    }
    passes = []
    for analysis in analyses.values():
        boot = analysis["image_cluster_bootstrap"]
        passes.append(
            analysis["point_deltas"]["macro_auroc"] >= 0.02
            and boot["macro_auroc_delta"]["ci95"][0] > 0.0
            and boot["nll_improvement"]["ci95"][0] > 0.0
        )

    config = {
        "protocol": PROTOCOL_VERSION,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "bootstrap_draws": args.bootstrap_draws,
        "findings": FINDINGS,
        "prompt_pairs": {
            finding: [
                f"a frontal chest radiograph showing {DISPLAY[finding]}",
                f"a frontal chest radiograph without {DISPLAY[finding]}",
            ]
            for finding in FINDINGS
        },
        "sources": {
            model_name: {
                split: {"path": str(path), "sha256": sha256_file(path)}
                for split, path in paths.items()
            }
            for model_name, paths in sources.items()
        },
        "image_root": str(args.image_root),
        "biomedclip_root": str(args.biomedclip_root),
        "biomedclip_weights_sha256": sha256_file(args.biomedclip_root / "open_clip_pytorch_model.bin"),
        "renderer": "sith_vindr_activation_probe_v1.render_dicom; percentile 1/99; MONOCHROME1 fix",
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "script_sha256": sha256_file(Path(__file__)),
    }
    result = {
        "status": "complete_cpu_proxy_admission",
        "protocol": PROTOCOL_VERSION,
        "decision": "PASS" if all(passes) else "NO_GO",
        "decision_rule": (
            "For both models: confirmation macro AUROC gain >=0.02, image-cluster bootstrap "
            "95% CI lower bound >0, and NLL-improvement CI lower bound >0. PASS only "
            "authorizes a diffusion-gradient pilot; it is not mitigation evidence."
        ),
        "claim_boundary": (
            "BiomedCLIP is an inexpensive independent-discriminator proxy, not a diffusion "
            "posterior score and not reciprocal-gradient evidence."
        ),
        "config": config,
        "config_fingerprint": canonical_hash(config),
        "image_count": len(image_ids),
        "analyses": analyses,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
