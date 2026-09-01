#!/usr/bin/env python3
"""CPU fatal screen for luma-preserving grayscale geometry packets.

The candidate starts from an accidental interface bottleneck: a grayscale
radiograph is commonly replicated into three RGB channels.  We keep the exact
linear luma of the ordinary low-pass render, while using the two luma-null
channel degrees of freedom to carry either

* the horizontal/vertical Haar details discarded by 2x downsampling; or
* the two components of the Riesz transform of the low-pass image.

This script is deliberately an information/accessibility screen, not a VLM
generation experiment.  It runs one local frozen BiomedCLIP tower on CPU,
uses image-disjoint development/confirmation manifests, and compares true
geometry with spatial, cross-image, orientation, and equal-energy placebos.
It never touches the baseline GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from anchor.corrected_sgta.run_huatuo_dicom_render_pilot_v1 import (
    canonical_polarity,
    read_dicom_pixels,
)
from anchor.corrected_sgta.screen_external_visual_increment_v1 import (
    load_claims,
    sha256_file,
)


VERSION = "luma-geometry-packet-biomedclip-v1"
FINDINGS = (
    "cardiomegaly",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)
DISPLAY = {
    "cardiomegaly": "cardiomegaly",
    "pleural_effusion": "pleural effusion",
    "pleural_thickening": "pleural thickening",
    "pulmonary_fibrosis": "pulmonary fibrosis",
}
VIEWS = (
    "base_uint8",
    "base_float",
    "haar_true",
    "haar_shuffle",
    "haar_cross",
    "riesz_true",
    "riesz_shuffle",
    "riesz_rotated",
    "equal_energy_noise",
)
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def balanced_cap(rows: list[dict[str, Any]], per_finding: int, seed: int):
    if per_finding % 2:
        raise ValueError("per-finding cap must be even")
    selected = []
    for finding in FINDINGS:
        for label in (0, 1):
            candidates = [
                row for row in rows
                if row["finding"] == finding and row["label"] == label
            ]
            if len(candidates) < per_finding // 2:
                raise ValueError(
                    f"{finding}/{label} has {len(candidates)} rows, below requested cap"
                )
            rng = np.random.default_rng(stable_seed(f"{finding}:{label}", seed))
            order = rng.permutation(len(candidates))[: per_finding // 2]
            selected.extend(dict(candidates[index]) for index in order)
    return selected


def float_window(path: Path, side: int) -> np.ndarray:
    pixels = read_dicom_pixels(path)
    finite = pixels.modality[pixels.valid]
    lo, hi = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    image = np.clip((pixels.modality - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    image = canonical_polarity(image, pixels.photometric)
    import cv2

    return cv2.resize(
        image.astype(np.float32),
        (2 * side, 2 * side),
        interpolation=cv2.INTER_AREA,
    )


def haar_packet(x: np.ndarray):
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    low = (a + b + c + d) / 4.0
    horizontal = (a - b + c - d) / 4.0
    vertical = (a + b - c - d) / 4.0
    diagonal = (a - b - c + d) / 4.0
    return low, horizontal, vertical, diagonal


def riesz_packet(image: np.ndarray):
    """Periodic discrete first-order Riesz pair.

    This arm is a deterministic accessibility transform of the low-pass image;
    unlike Haar H/V it does not add information absent from ``image``.
    """
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(image.astype(np.float64) - float(image.mean()))
    r1 = np.fft.ifft2((-1j * fx / radius) * spectrum).real
    r2 = np.fft.ifft2((-1j * fy / radius) * spectrum).real
    return r1.astype(np.float32), r2.astype(np.float32)


def luma_null_basis() -> np.ndarray:
    # Columns of V span ker(LUMA^T).  Their signs are fixed for provenance.
    _, _, vh = np.linalg.svd(LUMA[None, :], full_matrices=True)
    basis = vh[1:].T
    for column in range(2):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis.astype(np.float64)


CHROMA_BASIS = luma_null_basis()


def unit_pair(first: np.ndarray, second: np.ndarray):
    energy = float(np.mean(first.astype(np.float64) ** 2 + second.astype(np.float64) ** 2))
    scale = max(np.sqrt(energy), 1e-8)
    return first / scale, second / scale


def safe_luma_encode(
    low: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, dict[str, float]]:
    first, second = unit_pair(first, second)
    payload = amplitude * (
        first[..., None] * CHROMA_BASIS[:, 0]
        + second[..., None] * CHROMA_BASIS[:, 1]
    )
    capacity = np.ones(low.shape, dtype=np.float64)
    for channel in range(3):
        current = payload[..., channel]
        positive = current > 0
        negative = current < 0
        candidate = np.ones(low.shape, dtype=np.float64)
        candidate[positive] = (1.0 - low[positive]) / current[positive]
        candidate[negative] = low[negative] / (-current[negative])
        capacity = np.minimum(capacity, candidate)
    capacity = np.clip(capacity, 0.0, 1.0)
    rgb = low[..., None].astype(np.float64) + capacity[..., None] * payload
    luma = rgb @ LUMA
    diagnostics = {
        "max_abs_luma_error": float(np.max(np.abs(luma - low))),
        "mean_capacity": float(capacity.mean()),
        "fraction_at_full_capacity": float(np.mean(capacity >= 1.0 - 1e-12)),
        "min_rgb": float(rgb.min()),
        "max_rgb": float(rgb.max()),
    }
    return rgb.astype(np.float32), diagnostics


def shuffled_pair(first, second, image_id, seed):
    rng = np.random.default_rng(stable_seed(f"shuffle:{image_id}", seed))
    order = rng.permutation(first.size)
    return first.reshape(-1)[order].reshape(first.shape), second.reshape(-1)[order].reshape(second.shape)


def rotated_pair(first, second, image_id, seed):
    rng = np.random.default_rng(stable_seed(f"rotate:{image_id}", seed))
    angle = float(rng.uniform(0, 2 * np.pi))
    cosine, sine = np.cos(angle), np.sin(angle)
    return cosine * first - sine * second, sine * first + cosine * second


def noise_pair(shape, image_id, seed):
    rng = np.random.default_rng(stable_seed(f"noise:{image_id}", seed))
    return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(shape).astype(np.float32)


class BiomedTower:
    def __init__(self, root: Path, threads: int):
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS, create_model_and_transforms, get_tokenizer

        config = json.loads((root / "open_clip_config.json").read_text())
        config["model_cfg"]["text_cfg"]["hf_model_name"] = str(root / "text_encoder")
        config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = str(root)
        model_name = "biomedclip_luma_geometry_packet_v1"
        _MODEL_CONFIGS[model_name] = config["model_cfg"]
        self.model, _, _ = create_model_and_transforms(
            model_name=model_name,
            pretrained=str(root / "open_clip_pytorch_model.bin"),
            **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
        )
        self.model.eval().to("cpu")
        self.tokenizer = get_tokenizer(model_name)
        self.mean = np.asarray(config["preprocess_cfg"]["mean"], dtype=np.float32)
        self.std = np.asarray(config["preprocess_cfg"]["std"], dtype=np.float32)
        torch.set_num_threads(threads)
        self.provenance = {
            "root": str(root.resolve()),
            "weights_sha256": sha256_file(root / "open_clip_pytorch_model.bin"),
            "image_mean": self.mean.tolist(),
            "image_std": self.std.tolist(),
        }

    def tensor(self, images: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(images).astype(np.float32)
        array = (array - self.mean[None, None, None, :]) / self.std[None, None, None, :]
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def image(self, images: list[np.ndarray]) -> np.ndarray:
        with torch.inference_mode():
            features = self.model.encode_image(self.tensor(images), normalize=True)
        return features.cpu().numpy().astype(np.float32)

    def text_directions(self) -> np.ndarray:
        prompts = []
        for finding in FINDINGS:
            prompts.extend(
                [
                    f"a frontal chest radiograph showing {DISPLAY[finding]}",
                    f"a frontal chest radiograph without {DISPLAY[finding]}",
                ]
            )
        tokens = self.tokenizer(prompts, context_length=256)
        with torch.inference_mode():
            encoded = self.model.encode_text(tokens, normalize=True)
        encoded = encoded.cpu().numpy().reshape(len(FINDINGS), 2, -1)
        return encoded[:, 0] - encoded[:, 1]


def make_components(image_ids, image_root, side):
    result = {}
    for image_id in image_ids:
        x = float_window(image_root / f"{image_id}.dicom", side)
        low, horizontal, vertical, diagonal = haar_packet(x)
        riesz_x, riesz_y = riesz_packet(low)
        result[image_id] = {
            "low": low,
            "haar_h": horizontal,
            "haar_v": vertical,
            "haar_d": diagonal,
            "riesz_x": riesz_x,
            "riesz_y": riesz_y,
        }
    return result


def construct_views(image_id, other_id, components, amplitude, seed):
    current, other = components[image_id], components[other_id]
    low = current["low"]
    hs = shuffled_pair(current["haar_h"], current["haar_v"], image_id, seed)
    rs = shuffled_pair(current["riesz_x"], current["riesz_y"], image_id, seed + 1)
    rr = rotated_pair(current["riesz_x"], current["riesz_y"], image_id, seed)
    noise = noise_pair(low.shape, image_id, seed)
    pairs = {
        "haar_true": (current["haar_h"], current["haar_v"]),
        "haar_shuffle": hs,
        "haar_cross": (other["haar_h"], other["haar_v"]),
        "riesz_true": (current["riesz_x"], current["riesz_y"]),
        "riesz_shuffle": rs,
        "riesz_rotated": rr,
        "equal_energy_noise": noise,
    }
    views = {
        "base_uint8": np.repeat((np.rint(low * 255.0) / 255.0)[..., None], 3, axis=-1).astype(np.float32),
        "base_float": np.repeat(low[..., None], 3, axis=-1).astype(np.float32),
    }
    diagnostics = {}
    for name, pair in pairs.items():
        views[name], diagnostics[name] = safe_luma_encode(low, *pair, amplitude)
    return views, diagnostics


def macro_auc(rows, score):
    values = []
    for finding in FINDINGS:
        indices = [i for i, row in enumerate(rows) if row["finding"] == finding]
        labels = [rows[i]["label"] for i in indices]
        values.append(roc_auc_score(labels, score[indices]))
    return float(np.mean(values))


def choose_c(dev, features, seed):
    grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    values = {regularizer: [] for regularizer in grid}
    for finding_index, finding in enumerate(FINDINGS):
        indices = np.asarray([i for i, row in enumerate(dev) if row["finding"] == finding])
        labels = np.asarray([dev[i]["label"] for i in indices])
        matrix = features[[dev[i]["image_index"] for i in indices]]
        class_count = int(min(np.sum(labels == 0), np.sum(labels == 1)))
        splitter = StratifiedKFold(
            n_splits=min(4, class_count),
            shuffle=True,
            random_state=seed + finding_index,
        )
        for regularizer in grid:
            predictions = np.zeros(len(indices), dtype=np.float64)
            for train, heldout in splitter.split(matrix, labels):
                model = LogisticRegression(
                    C=regularizer,
                    max_iter=3000,
                    solver="liblinear",
                    random_state=seed,
                )
                model.fit(matrix[train], labels[train])
                predictions[heldout] = model.predict_proba(matrix[heldout])[:, 1]
            values[regularizer].append(roc_auc_score(labels, predictions))
    macro = {str(key): float(np.mean(value)) for key, value in values.items()}
    best = max(grid, key=lambda key: (np.mean(values[key]), -key))
    return float(best), macro


def probe(dev, confirmation, features, regularizer, seed):
    output = np.zeros(len(confirmation), dtype=np.float64)
    for finding_index, finding in enumerate(FINDINGS):
        train = [i for i, row in enumerate(dev) if row["finding"] == finding]
        test = [i for i, row in enumerate(confirmation) if row["finding"] == finding]
        x_train = features[[dev[i]["image_index"] for i in train]]
        x_test = features[[confirmation[i]["image_index"] for i in test]]
        labels = [dev[i]["label"] for i in train]
        model = LogisticRegression(
            C=regularizer,
            max_iter=3000,
            solver="liblinear",
            random_state=seed + finding_index,
        )
        model.fit(x_train, labels)
        output[test] = model.predict_proba(x_test)[:, 1]
    return output


def bootstrap(rows, scores, comparisons, draws, seed):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["image_id"]].append(index)
    image_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in comparisons}
    for _ in range(draws):
        selected = rng.choice(image_ids, len(image_ids), replace=True)
        indices = np.asarray([index for image_id in selected for index in groups[image_id]])
        sampled_rows = [rows[index] for index in indices]
        try:
            for name, (first, second) in comparisons.items():
                samples[name].append(
                    macro_auc(sampled_rows, scores[first][indices])
                    - macro_auc(sampled_rows, scores[second][indices])
                )
        except ValueError:
            continue
    result = {}
    for name, values in samples.items():
        array = np.asarray(values)
        result[name] = {
            "mean": float(array.mean()),
            "ci95": [float(value) for value in np.quantile(array, [0.025, 0.975])],
            "draws": len(array),
        }
    return result


def summarize_diagnostics(records):
    output = {}
    for view in records[0]:
        output[view] = {
            key: float(max(record[view][key] for record in records))
            if key == "max_abs_luma_error"
            else float(np.mean([record[view][key] for record in records]))
            for key in records[0][view]
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--biomedclip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--amplitude", type=float, default=0.06)
    parser.add_argument("--per-finding-dev", type=int, default=40)
    parser.add_argument("--per-finding-confirmation", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if (
        os.environ.get("LUMA_GEOMETRY_ALLOW_GPU") != "1"
        and os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1")
    ):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' or '-1'; this screen must not use GPU")
    if args.output.exists() or args.cache.exists():
        raise FileExistsError("Refusing to overwrite output/cache")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = balanced_cap(
        [row for row in load_claims(args.dev, "dev", "label") if row["finding"] in FINDINGS],
        args.per_finding_dev,
        args.seed,
    )
    confirmation = balanced_cap(
        [
            row for row in load_claims(args.confirmation, "confirmation", "label")
            if row["finding"] in FINDINGS
        ],
        args.per_finding_confirmation,
        args.seed + 1,
    )
    dev_images = {row["image_id"] for row in dev}
    confirmation_images = {row["image_id"] for row in confirmation}
    overlap = dev_images & confirmation_images
    if overlap:
        raise ValueError(f"Development/confirmation image leakage: {len(overlap)} images")
    image_ids = sorted(dev_images | confirmation_images)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    for row in dev + confirmation:
        row["image_index"] = image_index[row["image_id"]]
    shifted = image_ids[1:] + image_ids[:1]
    cross = dict(zip(image_ids, shifted))

    components = make_components(image_ids, args.image_root, args.side)
    tower = BiomedTower(args.biomedclip_root, args.threads)
    text = tower.text_directions()
    feature_parts = {view: [] for view in VIEWS}
    diagnostic_records = []
    for start in range(0, len(image_ids), args.batch_size):
        current_ids = image_ids[start : start + args.batch_size]
        images = {view: [] for view in VIEWS}
        for image_id in current_ids:
            views, diagnostics = construct_views(
                image_id, cross[image_id], components, args.amplitude, args.seed
            )
            diagnostic_records.append(diagnostics)
            for view in VIEWS:
                images[view].append(views[view])
        for view in VIEWS:
            feature_parts[view].append(tower.image(images[view]))
    features = {view: np.concatenate(parts) for view, parts in feature_parts.items()}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        image_ids=np.asarray(image_ids),
        views=np.asarray(VIEWS),
        **{f"feature_{view}": matrix for view, matrix in features.items()},
    )

    base_c, cv = choose_c(dev, features["base_float"], args.seed)
    probe_scores = {
        view: probe(dev, confirmation, matrix, base_c, args.seed)
        for view, matrix in features.items()
    }
    # A concatenated probe asks whether the transformed frozen representation
    # adds accessible information beyond the standard frozen representation.
    for view in ("haar_true", "riesz_true"):
        combined = np.concatenate([features["base_float"], features[view]], axis=1)
        probe_scores[f"base_plus_{view}"] = probe(
            dev, confirmation, combined, base_c / 2.0, args.seed
        )

    zero_shot_scores = {}
    finding_index = {finding: i for i, finding in enumerate(FINDINGS)}
    for view, matrix in features.items():
        all_finding = matrix @ text.T
        zero_shot_scores[view] = np.asarray(
            [
                all_finding[row["image_index"], finding_index[row["finding"]]]
                for row in confirmation
            ]
        )

    probe_auc = {view: macro_auc(confirmation, score) for view, score in probe_scores.items()}
    zero_shot_auc = {
        view: macro_auc(confirmation, score) for view, score in zero_shot_scores.items()
    }
    comparisons = {
        "base_float_minus_uint8": ("base_float", "base_uint8"),
        "haar_true_minus_base_float": ("haar_true", "base_float"),
        "haar_true_minus_shuffle": ("haar_true", "haar_shuffle"),
        "haar_true_minus_cross": ("haar_true", "haar_cross"),
        "riesz_true_minus_base_float": ("riesz_true", "base_float"),
        "riesz_true_minus_shuffle": ("riesz_true", "riesz_shuffle"),
        "riesz_true_minus_rotated": ("riesz_true", "riesz_rotated"),
        "haar_true_minus_noise": ("haar_true", "equal_energy_noise"),
        "riesz_true_minus_noise": ("riesz_true", "equal_energy_noise"),
        "base_plus_haar_minus_base": ("base_plus_haar_true", "base_float"),
        "base_plus_riesz_minus_base": ("base_plus_riesz_true", "base_float"),
    }
    boot = bootstrap(
        confirmation,
        probe_scores,
        comparisons,
        args.bootstrap_draws,
        args.seed,
    )

    def gate(family):
        required = [
            f"{family}_true_minus_base_float",
            f"{family}_true_minus_shuffle",
            f"{family}_true_minus_noise",
            f"base_plus_{family}_minus_base",
        ]
        if family == "haar":
            required.append("haar_true_minus_cross")
        else:
            required.append("riesz_true_minus_rotated")
        return (
            probe_auc[f"{family}_true"] - probe_auc["base_float"] >= 0.02
            and probe_auc[f"base_plus_{family}_true"] - probe_auc["base_float"] >= 0.02
            and all(boot[name]["ci95"][0] > 0 for name in required)
        )

    family_gate = {family: gate(family) for family in ("haar", "riesz")}
    result = {
        "version": VERSION,
        "status": "complete_cpu_fatal_screen",
        "decision": "PASS_L0" if any(family_gate.values()) else "NO_GO_L0",
        "family_gate": family_gate,
        "scope": (
            "Frozen BiomedCLIP accessibility/information screen only. A pass does not "
            "establish VLM generation mitigation, non-target preservation, or novelty."
        ),
        "command": shlex.join(sys.argv),
        "n_dev_claims": len(dev),
        "n_confirmation_claims": len(confirmation),
        "n_unique_images": len(image_ids),
        "n_dev_images": len(dev_images),
        "n_confirmation_images": len(confirmation_images),
        "findings": list(FINDINGS),
        "views": list(VIEWS),
        "biomedclip": tower.provenance,
        "input_sha256": {
            "dev": sha256_file(args.dev),
            "confirmation": sha256_file(args.confirmation),
        },
        "encoding": {
            "source_side": 2 * args.side,
            "target_side": args.side,
            "amplitude": args.amplitude,
            "luma": LUMA.tolist(),
            "chroma_basis": CHROMA_BASIS.tolist(),
            "basis_orthogonality_error": float(
                np.max(np.abs(CHROMA_BASIS.T @ CHROMA_BASIS - np.eye(2)))
            ),
            "basis_luma_null_error": float(np.max(np.abs(LUMA @ CHROMA_BASIS))),
            "haar_identity": (
                "For each 2x2 block, sum pixel squared error after LL-only reconstruction "
                "is 4(H^2+V^2+D^2); keeping LL,H,V leaves exactly 4D^2."
            ),
            "riesz_identity": (
                "Before finite-grid/roundoff effects, ||R1 f||^2+||R2 f||^2="
                "||f-mean(f)||^2 and the pair is rotation-covariant."
            ),
            "important_boundary": (
                "Riesz(lowpass) is an invertible/deterministic accessibility transform and "
                "contains no source information absent from the low-pass image. Only Haar "
                "H/V carries pre-downsample spatial degrees discarded by base_float."
            ),
        },
        "encoding_diagnostics": summarize_diagnostics(diagnostic_records),
        "probe": {
            "regularization_c_frozen_from_base_dev": base_c,
            "base_dev_cv_macro_auroc_by_c": cv,
            "confirmation_macro_auroc": probe_auc,
            "paired_image_bootstrap": boot,
        },
        "zero_shot_confirmation_macro_auroc": zero_shot_auc,
        "preregistered_gate": {
            "candidate_over_float_base": ">=0.02 macro AUROC and CI lower >0",
            "base_plus_candidate_over_base": ">=0.02 macro AUROC and CI lower >0",
            "specificity": (
                "true geometry must beat spatial shuffle, equal-energy noise, and its "
                "family-specific cross-image/orientation placebo with CI lower >0"
            ),
            "pass_rule": "at least one of Haar or Riesz passes every condition",
            "failure_action": "do not run a VLM generation canary for this candidate",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
