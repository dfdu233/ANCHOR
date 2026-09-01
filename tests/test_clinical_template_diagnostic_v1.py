from anchor.corrected_sgta.analyze_clinical_template_diagnostic_v1 import (
    analyze,
    benjamini_hochberg,
    concentration,
    extreme_collision_rate,
    mutual_information,
)


FINDINGS = (
    "aortic_enlargement",
    "cardiomegaly",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "pleural_effusion",
    "pleural_thickening",
    "pulmonary_fibrosis",
)


def _fixture():
    rows = []
    references = []
    for index in range(8):
        image = f"image-{index}"
        vote = index % 4
        for finding in FINDINGS:
            references.append({"image_id": image, "finding": finding, "positive_votes": vote})
        for condition in ("neutral", "existential", "negative_obligation"):
            text = "template low" if vote < 2 else "template high"
            rows.append(
                {
                    "image_id": image,
                    "prompt_condition": condition,
                    "text": text,
                    "prompt": condition,
                    "generated_token_ids": [vote, 2, 3],
                    "fingerprint": "frozen",
                    "hit_max_new_tokens": False,
                    "surface_refusal_match": False,
                    "clinical_claim_evaluation_status": "pending_shared_audit",
                }
            )
    integrity = {
        "images": 8,
        "dicom_file_identity_passed": True,
        "render_identity_passed": True,
        "cross_patient_study_collision_exclusion_passed": False,
    }
    return rows, references, integrity


def test_template_statistics_and_reader_association_are_exact():
    assert concentration(["a", "a", "b", "c"])["top1_share"] == 0.5
    assert mutual_information(["a", "a", "b", "b"], [0, 0, 3, 3]) > 0.6
    assert extreme_collision_rate(["a", "a", "a", "b"], [0, 0, 3, 3]) == 0.5
    assert benjamini_hochberg([0.001, 0.02, 0.8]) == [0.003, 0.03, 0.8000000000000002]


def test_analysis_is_descriptive_and_fails_closed_without_patient_linkage():
    rows, references, integrity = _fixture()
    result = analyze(rows, references, integrity, draws=20, seed=42)
    assert result["items"] == 8
    assert result["conditions"]["neutral"]["exact_report"]["unique"] == 2
    assert result["gates"]["cross_patient_study_collision_exclusion"] is False
    assert result["reader_association_multiple_testing"]["tests"] == 24
    assert result["paper_mechanism_authorized"] is False


def test_analysis_rejects_incomplete_triplets():
    rows, references, integrity = _fixture()
    try:
        analyze(rows[:-1], references, integrity, draws=10, seed=42)
    except ValueError as error:
        assert "incomplete prompt triplet" in str(error)
    else:
        raise AssertionError("incomplete generation was accepted")
