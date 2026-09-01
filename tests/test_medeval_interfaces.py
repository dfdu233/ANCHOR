from anchor.medeval.interfaces import validate_claim_evidence
from anchor.medeval.schema import ClaimEvidence


def test_claim_evidence_layer_contract_is_ordered_and_complete():
    rows = [
        ClaimEvidence(
            sample_id="x",
            finding="effusion",
            layer=layer,
            logits={"supported": 1.0, "refuted": 0.0, "undetermined": -1.0},
            polarity=0.5,
            clarity=1.0,
        )
        for layer in (7, 14, "final")
    ]
    validate_claim_evidence((7, 14, "final"), rows)
