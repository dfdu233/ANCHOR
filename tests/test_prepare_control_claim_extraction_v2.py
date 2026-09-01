import json

from anchor.medeval.prepare_control_claim_extraction_v1 import prepare


def _answer(stream_seed=42):
    return {
        "question_id": "q1",
        "text": "A complete answer.",
        "metadata": {
            "mean_token_nll": 0.2,
            "generated_token_ids": [1, 2],
            "base_seed": stream_seed,
            "stop_reason": "eos_or_template",
            "hit_max_new_tokens": False,
        },
    }


def test_claim_extraction_uses_generation_audit_baseline_arm(tmp_path) -> None:
    root = tmp_path / "run"
    for arm in ("sample_t07_p09_seed42", "greedy512"):
        path = root / "m" / arm / "answers.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_answer()) + "\n")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "passed": True,
                "expected_qids": ["q1"],
                "baseline_arm": "greedy512",
                "models": {"m": {"passed": True}},
            }
        )
    )
    contract = tmp_path / "aggregation.json"
    contract.write_text(json.dumps({"models": ["m"], "aggregation": {"seeds": [42]}}))
    output = tmp_path / "input.jsonl"
    manifest = tmp_path / "manifest.json"
    result = prepare(
        run_root=root,
        generation_audit=audit,
        aggregation_contract=contract,
        output=output,
        manifest_path=manifest,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["baseline_arm"] == "greedy512"
    assert rows[-1]["source"]["stream"] == "greedy512"
    assert rows[-1]["id"].endswith(":greedy512")
