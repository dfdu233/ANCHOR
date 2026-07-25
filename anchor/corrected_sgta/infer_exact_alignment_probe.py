"""Exact-source Fourier alignment probe without a proxy-source control."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta import infer_alignment_release2 as release2
from corrected_sgta import infer_alignment_v2 as implementation
from corrected_sgta.frequency_alignment_release3 import (
    feddg_frequency_interpolation_release3,
)
from corrected_sgta.provenance_release3 import (
    inference_code_identity,
    model_identity,
)
from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts


ALIGNMENT_CACHE_VERSION = "sgta-exact-source-fourier-probe-v1"
_base_protocol_fingerprint = implementation.protocol_fingerprint


def exact_probe_code_identity(project_root: Path) -> dict:
    identity = inference_code_identity(project_root)
    relative = "corrected_sgta/infer_exact_alignment_probe.py"
    identity[relative] = sha256_file(project_root / relative)
    return identity


def protocol_fingerprint(config: dict) -> str:
    config.update(
        {
            "probe_scope": (
                "fixed-dose causal probe; no outcome-based view or parameter selection"
            ),
            "matched_source_policy": (
                "only formal same-modality entries in the exact-source manifest"
            ),
            "wrong_control": "none in exact-source falsification probe",
            "model_source_claim_scope": {
                "llava": (
                    "exact released LLaVA-Med alignment-stage source membership; "
                    "caption-filtered CXR candidates"
                ),
                "hulu": "unsupported; this probe is LLaVA-only",
            },
        }
    )
    return _base_protocol_fingerprint(config)


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = ALIGNMENT_CACHE_VERSION
    implementation.feddg_frequency_interpolation = (
        feddg_frequency_interpolation_release3
    )
    implementation._structure_metrics = release2.clinical_structure_metrics
    implementation.safe_structure = release2.safe_structure
    implementation.model_identity = model_identity
    implementation.code_identity = exact_probe_code_identity
    implementation.protocol_fingerprint = protocol_fingerprint
    implementation.verify_source_artifacts = verify_source_artifacts
    implementation.load_feature_centers = release2.strict_load_feature_centers
    implementation.main()


if __name__ == "__main__":
    main()
