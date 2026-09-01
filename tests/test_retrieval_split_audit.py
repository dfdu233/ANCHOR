import json

from anchor.medeval.audit_retrieval_split import audit
from anchor.medeval.build_common_rag_corpus import build
from anchor.medeval.decontaminate_rag_corpus import decontaminate


def test_common_corpus_and_leakage_audit(tmp_path):
    source = tmp_path / "train.json"
    source.write_text(json.dumps([{
        "id": "a", "subject_id": 1, "study_id": 2,
        "image_path": ["p10/p1/s2/a.jpg"], "report": "No edema."
    }]))
    rows, manifest = build(source, "mimic")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"image": "p10/p3/s4/b.jpg", "report": "Normal heart."}) + "\n")
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text(json.dumps({"image": "p10/p1/s9/b.jpg", "report": "No edema."}) + "\n")
    assert manifest["documents"] == 1
    assert audit(corpus, clean)["passed"] is True
    result = audit(corpus, leaked)
    assert result["passed"] is False
    assert result["overlap_counts"]["patient_id"] == 1
    assert result["overlap_counts"]["report_sha256"] == 1
    kept, removed = decontaminate(rows, [json.loads(leaked.read_text())])
    assert kept == []
    assert removed == [{"doc_id": "mimic:a", "reasons": ["patient_id", "report_sha256"]}]
