import json

from anchor.medeval.summarize_common_rag_ladder import summarize


def test_rag_summary_never_pools_datasets(tmp_path):
    root = tmp_path / "root"
    for arm, accuracy in (("no_context", .8), ("rag", .9)):
        directory = root / "iuxray" / "visual_ce_v2" / "ladder_v3" / "T3_n200" / "huatuo" / arm
        directory.mkdir(parents=True)
        (directory / "qualification.json").write_text(json.dumps({"passed": True}))
        (directory / "evaluation.json").write_text(json.dumps({"invalid_ground_truth": 0, "accuracy": accuracy}))
    comparison = root / "iuxray" / "visual_ce_v2" / "ladder_v3" / "T3_n200" / "huatuo" / "comparison.json"
    comparison.write_text(json.dumps({"full_run_authorized": True}))
    result = summarize(root, ["iuxray"], ["huatuo"])
    assert result["dataset_pooling_forbidden"] is True
    assert result["records"][0]["t3_passed"] is True
    assert result["full_authorized"] == [{"dataset": "iuxray", "model": "huatuo"}]
