from anchor.corrected_sgta.analyze_natural_oe_diagnostic_completion_pilot_v1 import (
    concentration,
    normalize_text,
    prefix,
    sentence_count,
)


def test_response_geometry_normalization_is_deterministic():
    assert normalize_text("Opacity, LEFT! 2.0 cm") == "opacity left 2 0 cm"
    assert prefix("One two three four", 3) == "one two three"
    assert sentence_count("One. Two? Three!") == 3


def test_concentration_exposes_template_top_share():
    result = concentration(["a", "a", "b", "c"])
    assert result["total"] == 4
    assert result["unique"] == 3
    assert result["top1_value"] == "a"
    assert result["top1_share"] == 0.5
