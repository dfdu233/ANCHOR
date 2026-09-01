from anchor.corrected_sgta.analyze_cached_natural_laterality_v1 import (
    parse_lateralized_positive_findings,
)


def test_parses_unambiguous_positive_side() -> None:
    parsed = parse_lateralized_positive_findings(
        "There is a small left pleural effusion. A right apical pneumothorax is present."
    )
    assert parsed == {"pleural_effusion": "left", "pneumothorax": "right"}


def test_excludes_negated_and_bilateral_claims() -> None:
    parsed = parse_lateralized_positive_findings(
        "No left pleural effusion. There are right and left lung opacities."
    )
    assert parsed == {}


def test_nearest_side_binds_separate_findings() -> None:
    parsed = parse_lateralized_positive_findings(
        "A left basilar consolidation and a small right pleural effusion are seen."
    )
    assert parsed == {"consolidation": "left", "pleural_effusion": "right"}
