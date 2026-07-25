"""Final reviewed Wave-A inference entry point."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta import infer_alignment_v2 as implementation
from corrected_sgta.frequency_alignment_v2 import feddg_frequency_interpolation_v2
from corrected_sgta.provenance_wave_a import code_identity, model_identity
from corrected_sgta.source_bank_v2 import load_feature_centers as base_load_feature_centers
from corrected_sgta.source_bank_v3 import verify_source_artifacts


ALIGNMENT_CACHE_VERSION = "sgta-alignment-wave-a-v1"
_base_select_alignment_views = implementation.select_alignment_views


def strict_load_feature_centers(
    path: Path,
    expected_model: str | None = None,
    expected_source_bank_sha256: str | None = None,
):
    metadata, centers = base_load_feature_centers(
        path,
        expected_model=expected_model,
        expected_source_bank_sha256=expected_source_bank_sha256,
    )
    if expected_model is not None and metadata.get("model_identity") != model_identity(expected_model):
        raise RuntimeError("visual center/current checkpoint or model-code identity mismatch")
    current_code = code_identity(Path(__file__).resolve().parents[1])
    if metadata.get("code_identity") != current_code:
        raise RuntimeError("visual center/current SGTA code identity mismatch; rebuild centers")
    if metadata.get("batch_size") is None:
        raise RuntimeError("visual center metadata lacks batch-size provenance")
    return metadata, centers


def paired_select_alignment_views(*args, **kwargs):
    selected, original, candidates = _base_select_alignment_views(*args, **kwargs)
    paired = [
        item
        for item in selected
        if item.get("wrong_image") is not None and item.get("wrong_safe")
    ]
    return paired, original, candidates


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = ALIGNMENT_CACHE_VERSION
    implementation.feddg_frequency_interpolation = feddg_frequency_interpolation_v2
    implementation.model_identity = model_identity
    implementation.code_identity = code_identity
    implementation.verify_source_artifacts = verify_source_artifacts
    implementation.load_feature_centers = strict_load_feature_centers
    implementation.select_alignment_views = paired_select_alignment_views
    implementation.main()


if __name__ == "__main__":
    main()
