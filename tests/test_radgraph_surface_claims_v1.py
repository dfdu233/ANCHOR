from anchor.medeval.radgraph_surface_claims_v1 import claims_from_surface_graph


def test_surface_claims_preserve_state_anatomy_and_attributes_without_ontology() -> None:
    annotation = {
        "text": "Possible small right pleural effusion.",
        "entities": {
            "1": {"tokens": "possible", "label": "Observation::uncertain", "start_ix": 0, "relations": [["modify", "3"]]},
            "2": {"tokens": "small", "label": "Observation::definitely present", "start_ix": 1, "relations": [["modify", "3"]]},
            "3": {"tokens": "effusion", "label": "Observation::definitely present", "start_ix": 4, "relations": [["located_at", "4"]]},
            "4": {"tokens": "pleura", "label": "Anatomy::definitely present", "start_ix": 3, "relations": []},
            "5": {"tokens": "right", "label": "Anatomy::definitely present", "start_ix": 2, "relations": [["modify", "4"]]},
        },
    }
    claims, audit = claims_from_surface_graph(annotation)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.finding == "effusion"
    assert claim.anatomy == "right_pleura"
    assert claim.attributes == ("possible", "small")
    assert claim.uncertainty == "uncertain"
    assert audit["unparsed_as_no_structured_claim"] is False


def test_surface_claims_make_no_claim_from_anatomy_only_answer() -> None:
    claims, audit = claims_from_surface_graph(
        {
            "text": "right",
            "entities": {
                "1": {"tokens": "right", "label": "Anatomy::definitely present", "start_ix": 0, "relations": []}
            },
        }
    )
    assert claims == []
    assert audit["unparsed_as_no_structured_claim"] is True
