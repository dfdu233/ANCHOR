from anchor.corrected_sgta.build_specificity_ratchet_physician_pack_v2 import (
    extract_edges,
    split_sentences,
)


def test_exact_sentence_offsets_roundtrip():
    text = "A first finding.  A second finding due to infection."
    for sentence in split_sentences(text):
        assert text[sentence.start : sentence.end] == sentence.text


def test_extracts_modifier_and_etiology_without_assigning_truth():
    text = (
        "There is a large left pleural effusion. "
        "This may be caused by pneumonia or malignancy."
    )
    edges = extract_edges("What pathology is present?", text)
    assert {"laterality", "size_morph", "etiology"}.issubset(edges)
    assert edges["laterality"].answer_span in text
    assert edges["size_morph"].answer_span in text
    assert edges["etiology"].observability_screen == "explicit_or_likely_nonvisual_context"


def test_multiple_slices_is_not_a_size_morph_edge():
    text = "Diagnosing bowel obstruction requires viewing multiple slices through the abdomen."
    assert "size_morph" not in extract_edges("Where is obstruction present?", text)


def test_generic_due_to_orientation_is_not_etiology_edge():
    text = "The kidney appears on the left due to the orientation of medical images."
    assert "etiology" not in extract_edges("Where is the kidney?", text)
