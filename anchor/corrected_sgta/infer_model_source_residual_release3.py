"""Final V5 inference entry with strict-audit transitive identity."""

from __future__ import annotations

from corrected_sgta import infer_model_source_residual as implementation
from corrected_sgta.infer_model_source_residual_release2 import protocol_fingerprint
from corrected_sgta.model_source_residual_provenance_release3 import model_source_residual_identity_release3


def main() -> None:
    implementation.CACHE_VERSION = "sgta-model-source-visual-residual-release3-v1"
    implementation.model_source_residual_identity = model_source_residual_identity_release3
    implementation.protocol_fingerprint = protocol_fingerprint
    implementation.main()


if __name__ == "__main__": main()

