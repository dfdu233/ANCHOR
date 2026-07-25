"""Inference entry for robust capped-simplex source-guided token alignment."""

from corrected_sgta import infer_feature_transport as implementation
from corrected_sgta.models_token_transport_release3 import (
    TokenTransportMixinRelease3,
    load_token_transport_adapter_release3,
)
from corrected_sgta.token_transport_provenance_release5 import (
    token_transport_identity_release5,
)


def main() -> None:
    original_fingerprint = implementation.protocol_fingerprint

    def token_fingerprint(config: dict) -> str:
        config["transport_operator"] = "source-guided token residual"
        config["token_attention_temperature"] = float(
            TokenTransportMixinRelease3.transport_temperature
        )
        config["token_attention_weight_cap"] = float(
            TokenTransportMixinRelease3.transport_weight_cap
        )
        config["token_attention_normalization"] = (
            "float64 capped-simplex allocation with exact unit token mean"
        )
        return original_fingerprint(config)

    implementation.TRANSPORT_CACHE_VERSION = "sgta-source-guided-token-alignment-r4"
    implementation.load_transport_adapter = load_token_transport_adapter_release3
    implementation.transport_code_identity = token_transport_identity_release5
    implementation.protocol_fingerprint = token_fingerprint
    implementation.main()


if __name__ == "__main__":
    main()
