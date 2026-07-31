from anchor.corrected_sgta.run_style_prior_template_probe import (
    TEMPLATES,
    render_template,
)


def test_template_families_are_complete_and_semantically_opposed() -> None:
    for template_id in TEMPLATES:
        rendered = render_template(template_id, "pleural effusion")
        assert rendered["question"].endswith(".")
        assert rendered["positive"].endswith(".")
        assert rendered["negative"].endswith(".")
        assert rendered["positive"] != rendered["negative"]
        assert "pleural effusion" in rendered["positive"]
        assert "pleural effusion" in rendered["negative"]


def test_negative_templates_encode_absence() -> None:
    assert "absent" in render_template("evidence", "edema")["negative"]
    assert " no " in render_template("demonstrates", "edema")["negative"]
