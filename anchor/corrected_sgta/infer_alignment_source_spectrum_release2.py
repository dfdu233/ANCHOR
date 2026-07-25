"""Reviewed preregistered model-source full-spectrum alignment inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from corrected_sgta import infer_alignment_v2 as implementation
from corrected_sgta.frequency_alignment_source_spectrum_release2 import (
    source_spectrum_alignment_release2,
)
from corrected_sgta.infer_alignment_release2 import (
    clinical_structure_metrics,
    safe_structure,
    strict_load_feature_centers,
)
from corrected_sgta.provenance_source_spectrum_release2 import inference_code_identity
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import normalize_modality
from corrected_sgta.source_bank_v3 import verify_source_artifacts


ALIGNMENT_CACHE_VERSION = "sgta-source-spectrum-preregistered-release2-v1"
SOURCE_ID = "pubmedvision_xray_formal"
_base_candidate_metadata = implementation.candidate_metadata
_base_protocol_fingerprint = implementation.protocol_fingerprint
_base_select_alignment_views = implementation.select_alignment_views


def source_sample_modality(sample: dict, dataset: Path) -> str | None:
    explicit = normalize_modality(sample.get("modality"))
    if explicit is not None:
        return explicit
    name = dataset.name.lower()
    return "xray" if "cxr" in name or "xray" in name else None


def model_source_entries(manifest: dict, modality: str | None, formal_only: bool = True) -> list[dict]:
    if normalize_modality(modality) != "xray":
        return []
    return [
        item for item in manifest.get("entries", [])
        if item.get("formal") and item.get("source_id") == SOURCE_ID
    ]


def choose_target_domain_control(target: dict, controls: list[dict], key: str):
    candidates = [
        item for item in controls
        if item.get("formal")
        and item.get("source_id") in {"iuxray_xray_leaksafe", "mimic_cxr_leaksafe"}
        and normalize_modality(item.get("modality")) == "xray"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: hashlib.sha256(
        f"{key}:{item['source_id']}".encode()
    ).hexdigest())


def paired_safe_select_alignment_views(*args, **kwargs):
    _ignored, original, candidates = _base_select_alignment_views(*args, **kwargs)
    parsed_args = args[5] if len(args) > 5 else kwargs["args"]
    selected = []
    for source_id in sorted({item["source_id"] for item in candidates}):
        safe = [
            item for item in candidates
            if item["source_id"] == source_id and item["safe"]
            and item.get("wrong_image") is not None and item.get("wrong_safe")
        ]
        if not safe:
            continue
        best = min(safe, key=lambda item: (item["distance_after"], item["low_frequency_ratio"]))
        if best["relative_closure"] >= parsed_args.min_relative_closure:
            selected.append(best)
    selected.sort(key=lambda item: (-item["relative_closure"], item["source_id"]))
    return selected[: parsed_args.max_views], original, candidates


def candidate_metadata(item: dict, selected: bool) -> dict:
    payload = _base_candidate_metadata(item, selected)
    payload.update({
        "spectral_alpha": payload["low_frequency_ratio"],
        "transform": "hermitian_phase_preserving_full_spectrum_amplitude_residual",
        "dc_policy": "preserve_target_pre_quantization",
        "legacy_parameter_alias": "low_frequency_ratio stores spectral_alpha",
    })
    return payload


def protocol_fingerprint(config: dict) -> str:
    config.update({
        "spectral_alpha_grid": list(config["l_grid"]),
        "transform": "hermitian_phase_preserving_full_spectrum_amplitude_residual",
        "frequency_support": "all_non_dc_bins",
        "dc_policy": "preserve_target_pre_quantization",
        "matched_source_policy": "forced_pubmedvision_xray_formal_model_source_proxy",
        "parameter_alias": "l_grid is spectral_alpha_grid for harness compatibility",
        "wrong_control": "deterministic IU/MIMIC amplitude at identical alpha; PubMed center unchanged",
        "model_source_claim_scope": {
            "llava": "training-adjacent PMC publication-image domain; exact PMC-15M membership not claimed",
            "hulu": "public medical multimodal proxy; sample-level training manifest unavailable",
        },
    })
    return _base_protocol_fingerprint(config)


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = ALIGNMENT_CACHE_VERSION
    implementation.feddg_frequency_interpolation = source_spectrum_alignment_release2
    implementation.entries_for_modality = model_source_entries
    implementation.sample_modality = source_sample_modality
    implementation.choose_wrong_entry = choose_target_domain_control
    implementation.candidate_metadata = candidate_metadata
    implementation._structure_metrics = clinical_structure_metrics
    implementation.safe_structure = safe_structure
    implementation.model_identity = model_identity
    implementation.code_identity = inference_code_identity
    implementation.protocol_fingerprint = protocol_fingerprint
    implementation.verify_source_artifacts = verify_source_artifacts
    implementation.load_feature_centers = strict_load_feature_centers
    implementation.select_alignment_views = paired_safe_select_alignment_views
    implementation.main()


if __name__ == "__main__":
    main()

