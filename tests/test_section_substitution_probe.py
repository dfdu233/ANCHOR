from corrected_sgta.run_huatuo_section_substitution_probe import split_sections


def test_split_sections_impression_first() -> None:
    report = "impression: Mild edema. Findings: Bilateral interstitial opacity."
    assert split_sections(report) == {
        "impression": "Mild edema",
        "findings": "Bilateral interstitial opacity",
    }


def test_split_sections_findings_first_and_whitespace() -> None:
    report = "FINDINGS:\n Lungs are clear.\nIMPRESSION: No acute disease."
    assert split_sections(report) == {
        "findings": "Lungs are clear",
        "impression": "No acute disease",
    }
