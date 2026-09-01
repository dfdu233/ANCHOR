import json
from pathlib import Path

from anchor.medeval.audit_internal_control_generation_t2_v1 import audit
from anchor.medeval.hashing import sha256_file
from anchor.medeval.run_native_oe_vqa import stable_seed


def _write_arm(root: Path, model: str, name: str, qids: list[str], seed: int, variant: int = 0) -> None:
    path = root / model / name
    path.mkdir(parents=True)
    config = {
        "model": model,
        "manifest_sha256": "frozen",
        "seed": seed,
        "generation": {"do_sample": "sample" in name or "replay" in name},
    }
    (path / "generation_config.json").write_text(json.dumps(config))
    rows = []
    for index, qid in enumerate(qids):
        token = index + 1 + variant
        rows.append(
            {
                "question_id": qid,
                "text": f"answer {token}",
                "metadata": {
                    "generated_token_ids": [token],
                    "generated_token_count": 1,
                    "mean_token_nll": 0.5,
                    "base_seed": seed,
                    "sample_seed": stable_seed(seed, qid),
                    "hit_max_new_tokens": False,
                    "stop_reason": "eos_or_template",
                },
            }
        )
    (path / "answers.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_generation_audit_accepts_complete_non_degenerate_replay(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.json"
    freeze = tmp_path / "freeze.json"
    contract = tmp_path / "contract.json"
    root = tmp_path / "runs"
    rows = [{"qid": "q1", "image_sha256": "a" * 64}]
    pilot.write_text(json.dumps(rows))
    contract.write_text(json.dumps({"models": ["huatuo", "hulu"]}))
    freeze.write_text(
        json.dumps(
            {
                "pilot_manifest_sha256": sha256_file(pilot),
                "execution_contract_sha256": sha256_file(contract),
                "source_test_image_overlap": 0,
                "held_out_manifest_sha256": "test",
                "development_manifest_sha256": "dev",
            }
        )
    )
    for model in ("huatuo", "hulu"):
        for arm in ("greedy128", "greedy256", "sample_t02_p09_seed42", "sample_t10_p09_seed42"):
            _write_arm(root, model, arm, ["q1"], 42)
        for offset, seed in enumerate((42, 1042, 2042, 3042, 4042)):
            _write_arm(root, model, f"sample_t07_p09_seed{seed}", ["q1"], seed, offset)
        _write_arm(root, model, "replay_t07_p09_seed42", ["q1"], 42)
    result = audit(
        run_root=root,
        pilot_manifest=pilot,
        freeze_provenance=freeze,
        execution_contract=contract,
        limit=1,
    )
    assert result["passed"] is True
    assert all(value["deterministic_replay_exact"] for value in result["models"].values())
    assert all(value["sampling_non_degenerate"] for value in result["models"].values())
