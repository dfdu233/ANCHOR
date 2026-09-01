from PIL import Image

from anchor.medeval.prepare_common_rag_controls import (
    context_text,
    image_identity_group,
    shuffled_context,
    swapped_images,
)


def row(qid, image, doc, context):
    return {
        "qid": qid,
        "img_name": image,
        "question": (
            "Use image.\nRetrieved reports:\n" + context +
            "\nQuestion: Is edema present?\nBegin the answer with Yes or No."
        ),
        "answer": "No.",
        "retrieved_doc_ids": [doc],
    }


def test_controls_derange_context_and_image_without_changing_target(tmp_path):
    rows = [
        row("a", "a.png", "d1", "[1] No edema."),
        row("b", "b.png", "d2", "[1] Cardiomegaly."),
        row("c", "c.png", "d3", "[1] Pleural effusion."),
    ]
    for index, name in enumerate(("a.png", "b.png", "c.png")):
        Image.new("RGB", (10 + index, 10), "gray").save(tmp_path / name)

    shuffled, context_assignment = shuffled_context(rows)
    swapped, image_assignment = swapped_images(rows, tmp_path)

    assert [value["qid"] for value in shuffled] == ["a", "b", "c"]
    assert [value["answer"] for value in shuffled] == ["No."] * 3
    assert all(not value["document_overlap"] for value in context_assignment)
    assert all(value["context_donor_qid"] != value["qid"] for value in shuffled)
    assert all(value["image_donor_qid"] != value["qid"] for value in swapped)
    assert all(value["target_image"] != value["donor_image"] for value in image_assignment)
    assert sorted(context_text(value["question"]) for value in shuffled) == sorted(
        context_text(value["question"]) for value in rows
    )
    assert all(
        value["target_identity_group"] != value["donor_identity_group"]
        for value in image_assignment
    )


def test_image_identity_group_uses_mimic_patient_and_iuxray_study():
    assert image_identity_group("p15/p15518538/s53078789/image.jpg") == (
        "patient",
        "p15518538",
    )
    assert image_identity_group("p15/p15518538/s99999999/other.jpg") == (
        "patient",
        "p15518538",
    )
    assert image_identity_group("CXR3030_IM-1405/0.png") == (
        "study",
        "CXR3030_IM-1405",
    )


def test_current_shared_rag_newline_question_delimiter_is_supported():
    prompt = (
        "Use image.\nRetrieved reports:\n[1] No edema."
        "\nQuestion:\nWhat abnormalities are present?"
    )
    assert context_text(prompt) == "[1] No edema."
