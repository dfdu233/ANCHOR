"""Token-alignment inference with explicit operator parameters in metadata."""

from corrected_sgta import infer_feature_transport as implementation
from corrected_sgta.models_token_transport import (
    TokenTransportMixin,
    load_token_transport_adapter,
)
from corrected_sgta.token_transport_provenance_release2 import (
    token_transport_identity_release2,
)


def main() -> None:
    original_fingerprint = implementation.protocol_fingerprint

    def token_fingerprint(config: dict) -> str:
        config["transport_operator"] = "source-guided token residual"
        config["token_attention_temperature"] = float(
            TokenTransportMixin.transport_temperature
        )
        config["token_attention_weight_cap"] = float(
            TokenTransportMixin.transport_weight_cap
        )
        config["token_attention_normalization"] = (
            "cap then renormalize to unit mean over visual tokens"
        )
        return original_fingerprint(config)

    implementation.TRANSPORT_CACHE_VERSION = "sgta-source-guided-token-alignment-r2"
    implementation.load_transport_adapter = load_token_transport_adapter
    implementation.transport_code_identity = token_transport_identity_release2
    implementation.protocol_fingerprint = token_fingerprint
    implementation.main()


if __name__ == "__main__":
    main()
