from anchor.medeval.retrieve_common_rag import retrieve


def test_common_retriever_is_deterministic_and_prefers_matching_report():
    rows = [
        {"doc_id":"b", "report":"No pleural effusion.", "report_sha256":"b"*64, "dataset":"x"},
        {"doc_id":"a", "report":"Large pleural effusion.", "report_sha256":"a"*64, "dataset":"x"},
        {"doc_id":"c", "report":"Normal heart size.", "report_sha256":"c"*64, "dataset":"x"},
    ]
    first = retrieve(rows, "Is there pleural effusion?", 2)
    second = retrieve(rows, "Is there pleural effusion?", 2)
    assert first == second
    assert {row["doc_id"] for row in first} == {"a", "b"}
    assert [row["rank"] for row in first] == [1, 2]
