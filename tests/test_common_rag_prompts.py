from anchor.medeval.prepare_common_rag_prompts import prepare


def test_common_rag_prompts_differ_only_by_context_content():
    queries = [{"question_id": "q", "image": "x.jpg", "question": "Is edema present? <image>", "answer": "No.", "observability": "image_grounded"}]
    retrieval = [{"sample_id":"q", "index_version":"v", "documents":[{"doc_id":"d", "rank":1, "report":"No edema."}]}]
    no_context = prepare(queries, retrieval, False)[0]
    rag = prepare(queries, retrieval, True)[0]
    assert no_context["retrieved_doc_ids"] == []
    assert rag["retrieved_doc_ids"] == ["d"]
    assert "No edema." not in no_context["question"]
    assert "No edema." in rag["question"]
    assert no_context["answer"] == rag["answer"] == "No."
