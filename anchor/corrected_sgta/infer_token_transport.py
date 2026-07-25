"""Run the frozen feature-transport protocol with token-aware source weighting."""

from corrected_sgta import infer_feature_transport as implementation
from corrected_sgta.models_token_transport import load_token_transport_adapter
from corrected_sgta.token_transport_provenance import token_transport_identity


def main() -> None:
    implementation.TRANSPORT_CACHE_VERSION = "sgta-source-guided-token-alignment-v1"
    implementation.load_transport_adapter = load_token_transport_adapter
    implementation.transport_code_identity = token_transport_identity
    implementation.main()


if __name__ == "__main__":
    main()
