"""One-view matched-versus-shuffled Wave-A mechanism test."""

from __future__ import annotations

from corrected_sgta import infer_alignment_release2 as release2
from corrected_sgta import infer_alignment_v2 as implementation
from corrected_sgta.frequency_alignment_release3 import feddg_frequency_interpolation_release3
from corrected_sgta.provenance_release4 import inference_code_identity, model_identity


_base_protocol_fingerprint = implementation.protocol_fingerprint


def corrected_protocol_fingerprint(config: dict) -> str:
    config["wrong_control"] = (
        "deterministic same-modality source-amplitude shuffle; target feature center unchanged"
    )
    config["wave_a_view_budget"] = "one matched view and one paired shuffled view"
    return _base_protocol_fingerprint(config)


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = "sgta-alignment-release4-v1"
    implementation.feddg_frequency_interpolation = feddg_frequency_interpolation_release3
    implementation.choose_wrong_entry = release2.choose_same_modality_control
    implementation._structure_metrics = release2.clinical_structure_metrics
    implementation.safe_structure = release2.safe_structure
    implementation.model_identity = model_identity
    implementation.code_identity = inference_code_identity
    implementation.protocol_fingerprint = corrected_protocol_fingerprint
    implementation.verify_source_artifacts = release2.verify_source_artifacts
    implementation.load_feature_centers = release2.strict_load_feature_centers
    implementation.select_alignment_views = release2.paired_select_alignment_views
    implementation.main()


if __name__ == "__main__":
    main()
