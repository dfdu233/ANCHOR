#!/usr/bin/env python3
"""CPU upper-bound screen for a frozen chest-radiograph specialist.

Direct expert/VLM fusion is established prior art (for example CCD).  This
screen therefore tests only whether an independent, disease-trained visual
model contributes held-out case information beyond a medical VLM's final
claim margin.  It is not presented as a mitigation method.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from anchor.corrected_sgta.screen_external_visual_increment_v1 import (
    FINDINGS,
    analyze_model,
    canonical_hash,
    load_claims,
    sha256_file,
)


PROTOCOL_VERSION = "external-visual-increment-xrv-densenet-v1"
XRV_LABELS = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)
FINDING_TARGETS = {
    "aortic_enlargement": ("Enlarged Cardiomediastinum",),
    "cardiomegaly": ("Cardiomegaly",),
    "lung_opacity": ("Lung Opacity",),
    "nodule_mass": ("Nodule", "Mass", "Lung Lesion"),
    "pleural_effusion": ("Effusion",),
    "pleural_thickening": ("Pleural_Thickening",),
    "pulmonary_fibrosis": ("Fibrosis",),
}


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def install_xrv_import_contract(models_source: Path):
    """Load the official models.py without installing its optional dataset stack."""

    package = types.ModuleType("torchxrayvision")
    package.__path__ = []
    datasets = types.ModuleType("torchxrayvision.datasets")
    datasets.default_pathologies = list(XRV_LABELS)
    utils = types.ModuleType("torchxrayvision.utils")
    warning_log: dict[str, bool] = {}

    def fix_resolution(x: torch.Tensor, resolution: int, model: torch.nn.Module):
        if len(x.shape) == 3:
            x = x[None, ...]
        if x.shape[2] != x.shape[3]:
            raise ValueError(f"XRV requires square input; got {tuple(x.shape)}")
        if x.shape[2:] != (resolution, resolution):
            return F.interpolate(x, size=(resolution, resolution), mode="bilinear", antialias=True)
        return x

    def warn_normalization(x: torch.Tensor) -> None:
        if "checked" not in warning_log:
            low, high = float(x.min()), float(x.max())
            if low < -1025 or high > 1025:
                raise ValueError(f"XRV normalization outside [-1024,1024]: [{low},{high}]")
            warning_log["checked"] = True

    utils.fix_resolution = fix_resolution
    utils.warn_normalization = warn_normalization
    utils.get_cache_dir = lambda: str(models_source.parent)
    utils.download = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("Downloads are disabled in the isolated XRV loader")
    )
    package.datasets = datasets
    package.utils = utils
    sys.modules["torchxrayvision"] = package
    sys.modules["torchxrayvision.datasets"] = datasets
    sys.modules["torchxrayvision.utils"] = utils

    spec = importlib.util.spec_from_file_location("torchxrayvision.models", models_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official XRV source: {models_source}")
    module = importlib.util.module_from_spec(spec)
    package.models = module
    sys.modules["torchxrayvision.models"] = module
    spec.loader.exec_module(module)
    return module


def load_xrv(models_source: Path, checkpoint: Path) -> torch.nn.Module:
    module = install_xrv_import_contract(models_source)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(saved, torch.nn.Module):
        model = saved
    else:
        state = saved.get("state_dict", saved) if isinstance(saved, dict) else None
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported XRV checkpoint object: {type(saved)}")
        model = module.DenseNet(weights=None, num_classes=len(XRV_LABELS), op_threshs=None)
        model.load_state_dict(state)
    model.eval().to("cpu")
    if len(getattr(model, "pathologies", getattr(model, "targets", ()))) != len(XRV_LABELS):
        raise ValueError("XRV target count drift")
    return model


def dicom_tensor(path: Path) -> torch.Tensor:
    ds = pydicom.dcmread(path, force=True)
    if ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"Unsupported photometric interpretation: {ds.PhotometricInterpretation}")
    array = ds.pixel_array.astype(np.float32)
    max_value = float(2 ** int(ds.BitsStored) - 1)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        array = max_value - array
    array = (2.0 * array / max_value - 1.0) * 1024.0
    height, width = array.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    array = array[top : top + side, left : left + side]
    tensor = torch.from_numpy(array)[None, None]
    return F.interpolate(tensor, size=(224, 224), mode="bilinear", antialias=True)[0]


def encode_images(
    image_ids: list[str],
    image_root: Path,
    model: torch.nn.Module,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch_ids in chunks(image_ids, batch_size):
            images = torch.stack([dicom_tensor(image_root / f"{image_id}.dicom") for image_id in batch_ids])
            if hasattr(model, "features2") and hasattr(model, "classifier"):
                logits = model.classifier(model.features2(images))
            else:
                probabilities = torch.clamp(model(images), 1e-6, 1 - 1e-6)
                logits = torch.logit(probabilities)
            for image_id, row in zip(batch_ids, logits.cpu().numpy()):
                output[image_id] = np.asarray(row, dtype=np.float32)
    return output


def attach_xrv_scores(rows: list[dict[str, Any]], logits: dict[str, np.ndarray]) -> None:
    label_index = {name: index for index, name in enumerate(XRV_LABELS)}
    for row in rows:
        values = [logits[row["image_id"]][label_index[name]] for name in FINDING_TARGETS[row["finding"]]]
        row["clip_score"] = float(max(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--huatuo-dev", type=Path, required=True)
    parser.add_argument("--huatuo-confirmation", type=Path, required=True)
    parser.add_argument("--hulu-dev", type=Path, required=True)
    parser.add_argument("--hulu-confirmation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--models-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' so the baseline GPU remains untouched")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "result.json"
    if output_path.exists():
        raise FileExistsError(output_path)

    sources = {
        "huatuo": {"development": args.huatuo_dev, "confirmation": args.huatuo_confirmation},
        "hulu": {"development": args.hulu_dev, "confirmation": args.hulu_confirmation},
    }
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_rows: list[dict[str, Any]] = []
    for model_name, paths in sources.items():
        rows[model_name] = {}
        for split, path in paths.items():
            current = load_claims(path, split, model_name)
            rows[model_name][split] = current
            all_rows.extend(current)
    image_ids = sorted({row["image_id"] for row in all_rows})
    missing = [image_id for image_id in image_ids if not (args.image_root / f"{image_id}.dicom").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} DICOM files; first={missing[0]}")

    model = load_xrv(args.models_source, args.checkpoint)
    logits = encode_images(image_ids, args.image_root, model, args.batch_size)
    for model_rows in rows.values():
        for split_rows in model_rows.values():
            attach_xrv_scores(split_rows, logits)

    score_matrix = np.stack([logits[image_id] for image_id in image_ids])
    np.savez_compressed(
        args.output_dir / "xrv_logits.npz",
        image_ids=np.asarray(image_ids),
        logits=score_matrix,
        labels=np.asarray(XRV_LABELS),
    )
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
            and boot["macro_auroc_delta"]["ci95"][0] > 0
            and boot["nll_improvement"]["ci95"][0] > 0
        )

    config = {
        "protocol": PROTOCOL_VERSION,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "bootstrap_draws": args.bootstrap_draws,
        "findings": list(FINDINGS),
        "finding_targets": FINDING_TARGETS,
        "sources": {
            model_name: {
                split: {"path": str(path), "sha256": sha256_file(path)}
                for split, path in paths.items()
            }
            for model_name, paths in sources.items()
        },
        "image_root": str(args.image_root),
        "models_source": str(args.models_source),
        "models_source_sha256": sha256_file(args.models_source),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "renderer": "official XRV DICOM normalization [-1024,1024], MONOCHROME1 fix, center crop, 224 bilinear",
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "script_sha256": sha256_file(Path(__file__)),
    }
    result = {
        "status": "complete_cpu_specialist_upper_bound",
        "protocol": PROTOCOL_VERSION,
        "decision": "PASS" if all(passes) else "NO_GO",
        "decision_rule": (
            "For both VLMs: confirmation macro AUROC gain >=0.02, image-cluster bootstrap "
            "95% CI lower bound >0, and NLL-improvement CI lower bound >0. A pass is only "
            "an external-specialist upper bound because direct expert fusion collides with CCD."
        ),
        "claim_boundary": (
            "This tests incremental case information from a frozen CXR specialist. It is not a "
            "new hallucination mitigation method, a general medical-modality result, or a VLM-internal mechanism."
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
