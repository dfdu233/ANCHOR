import json

from anchor.medeval.summarize_rag_controls import summarize


def test_rag_claim_requires_both_relevance_and_image_controls(tmp_path):
    directory = tmp_path / "iuxray" / "visual_ce_v2" / "ladder_v3" / "causal_controls_v1"
    directory.mkdir(parents=True)
    (directory / "rag_vs_shuffled_context.json").write_text(
        json.dumps({"full_run_authorized": True})
    )
    result = summarize(tmp_path, ["iuxray"])
    assert result["supported"] == []
    (directory / "rag_vs_image_swap.json").write_text(
        json.dumps({"full_run_authorized": True})
    )
    result = summarize(tmp_path, ["iuxray"])
    assert result["supported"] == [{"dataset": "iuxray", "model": "llava"}]
