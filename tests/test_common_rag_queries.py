from anchor.medeval.prepare_common_rag_queries import prepare, route


def row(question, answer="Yes.", qid="1"):
    return {"question_id": qid, "question": question, "answer": answer, "image": "x.jpg"}


def test_query_contract_separates_information_requirements():
    assert route(row("Is edema visible on the image?"))[0] == "image_grounded"
    assert route(row("Should the patient be treated for edema?"))[0] == "knowledge_claim"
    assert route(row("Has edema changed since the previous image?"))[0] == "unobservable"
    assert route(row("Has the patient undergone surgery?"))[0] == "knowledge_claim"
    assert route(row("There is no edema?", answer="The report says no."))[0] == "invalid_reference"


def test_prepare_retains_excluded_rows_for_audit():
    admitted, excluded = prepare([
        row("Is edema present?", qid="a"),
        row("Should edema be treated?", qid="b"),
    ])
    assert [item["qid"] for item in admitted] == ["a"]
    assert [item["qid"] for item in excluded] == ["b"]
