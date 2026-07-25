"""Mean-preserving, same-modality-control Wave-A inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from corrected_sgta import infer_alignment_v2 as implementation
from corrected_sgta.frequency_alignment_release2 import feddg_frequency_interpolation_release2
from corrected_sgta.infer_ce import _structure_metrics as base_structure_metrics
from corrected_sgta.provenance_release2 import (
    center_code_identity,
    inference_code_identity,
    model_identity,
)
from corrected_sgta.source_bank_v2 import load_feature_centers as base_load_feature_centers
from corrected_sgta.source_bank_v2 import normalize_modality
from corrected_sgta.source_bank_v3 import verify_source_artifacts
from corrected_sgta.structure_audit_v2 import gray, ssim
from corrected_sgta.structure_audit_wave_a import structure_proxy


ALIGNMENT_CACHE_VERSION = "sgta-alignment-release2-v1"
_base_select_alignment_views = implementation.select_alignment_views


def choose_same_modality_control(target: dict, controls: list[dict], key: str):
    candidates = [
        item
        for item in controls
        if item["source_id"] != target["source_id"]
        and normalize_modality(item.get("modality"))
        == normalize_modality(target.get("modality"))
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{key}:{target['source_id']}:{item['source_id']}".encode()
        ).hexdigest(),
    )


def clinical_structure_metrics(left, right) -> dict:
    metrics = base_structure_metrics(left, right)
    left_gray, right_gray = gray(left), gray(right)
    metrics["ssim"] = ssim(left_gray, right_gray)
    metrics.update(structure_proxy(left_gray, right_gray))
    return metrics


def safe_structure(metrics: dict, _args) -> bool:
    ratio = metrics.get("central_gradient_magnitude_ratio")
    return (
        metrics.get("edge_correlation") is not None
        and float(metrics["edge_correlation"]) >= 0.90
        and metrics.get("ssim") is not None
        and float(metrics["ssim"]) >= 0.90
        and float(metrics.get("central_local_contrast_correlation", 0.0)) >= 0.85
        and ratio is not None
        and 0.75 <= float(ratio) <= 1.25
    )


def strict_load_feature_centers(path: Path, expected_model=None, expected_source_bank_sha256=None):
    metadata, centers = base_load_feature_centers(
        path,
        expected_model=expected_model,
        expected_source_bank_sha256=expected_source_bank_sha256,
    )
    if expected_model is not None and metadata.get("model_identity") != model_identity(expected_model):
        raise RuntimeError("visual center/current checkpoint or model-code identity mismatch")
    if metadata.get("code_identity") != center_code_identity(Path(__file__).resolve().parents[1]):
        raise RuntimeError("visual center/current encoder code identity mismatch; rebuild centers")
    if metadata.get("batch_size") is None:
        raise RuntimeError("visual center metadata lacks batch-size provenance")
    return metadata, centers


def paired_select_alignment_views(*args, **kwargs):
    selected, original, candidates = _base_select_alignment_views(*args, **kwargs)
    return [
        item
        for item in selected
        if item.get("wrong_image") is not None and item.get("wrong_safe")
    ], original, candidates


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = ALIGNMENT_CACHE_VERSION
    implementation.feddg_frequency_interpolation = feddg_frequency_interpolation_release2
    implementation.choose_wrong_entry = choose_same_modality_control
    implementation._structure_metrics = clinical_structure_metrics
    implementation.safe_structure = safe_structure
    implementation.model_identity = model_identity
    implementation.code_identity = inference_code_identity
    implementation.verify_source_artifacts = verify_source_artifacts
    implementation.load_feature_centers = strict_load_feature_centers
    implementation.select_alignment_views = paired_select_alignment_views
    implementation.main()


if __name__ == "__main__":
    main()
