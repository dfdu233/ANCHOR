"""Inference entry for exactly capped source-guided token alignment."""

from corrected_sgta import infer_feature_transport as implementation
from corrected_sgta.models_token_transport_release2 import (
    TokenTransportMixinRelease2,
    load_token_transport_adapter_release2,
)
from corrected_sgta.token_transport_provenance_release4 import (
    token_transport_identity_release4,
)


def main() -> None:
    original_fingerprint = implementation.protocol_fingerprint

    def token_fingerprint(config: dict) -> str:
        config["transport_operator"] = "source-guided token residual"
        config["token_attention_temperature"] = float(
            TokenTransportMixinRelease2.transport_temperature
        )
        config["token_attention_weight_cap"] = float(
            TokenTransportMixinRelease2.transport_weight_cap
        )
        config["token_attention_normalization"] = "exact cap with unit token mean"
        return original_fingerprint(config)

    implementation.TRANSPORT_CACHE_VERSION = "sgta-source-guided-token-alignment-r3"
    implementation.load_transport_adapter = load_token_transport_adapter_release2
    implementation.transport_code_identity = token_transport_identity_release4
    implementation.protocol_fingerprint = token_fingerprint
    implementation.main()


if __name__ == "__main__":
    main()
