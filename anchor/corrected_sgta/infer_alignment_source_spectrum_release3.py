"""Final reviewed inference entry with release-3 transitive identity."""

from __future__ import annotations

from corrected_sgta import infer_alignment_source_spectrum_release2 as implementation
from corrected_sgta.provenance_source_spectrum_release3 import inference_code_identity


def main() -> None:
    implementation.ALIGNMENT_CACHE_VERSION = "sgta-source-spectrum-preregistered-release3-v1"
    implementation.inference_code_identity = inference_code_identity
    implementation.main()


if __name__ == "__main__":
    main()

