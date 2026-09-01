import json

import numpy as np
import pytest

from anchor.corrected_sgta.collect_vindr_hidden_states_v2 import (
    atomic_jsonl,
    atomic_npz,
    checkpoint_arrays,
    load_checkpoint,
)
from anchor.corrected_sgta.run_huatuo_vindr_commitment_probe import sha256_file


def test_checkpoint_is_exact_ordered_prefix_and_hash_bound(tmp_path):
    rows = [
        {"finding": "effusion", "image_id": "a"},
        {"finding": "effusion", "image_id": "b"},
    ]
    metadata = [{"record_key": "effusion:a"}]
    arrays = checkpoint_arrays(
        [np.ones((2, 3), dtype=np.float32)],
        [np.ones((2, 3), dtype=np.float32) * 2],
        [np.ones((2, 3), dtype=np.float32) * 3],
        [np.ones((2, 7), dtype=np.float32) * 4],
        [1, 2],
    )
    atomic_npz(tmp_path / "checkpoint.npz", **arrays)
    atomic_jsonl(tmp_path / "checkpoint_metadata.jsonl", metadata)
    contract = "frozen-contract"
    (tmp_path / "config.json").write_text(
        json.dumps({"resume_contract_sha256": contract})
    )
    (tmp_path / "checkpoint_state.json").write_text(
        json.dumps(
            {
                "completed": 1,
                "metadata_sha256": sha256_file(tmp_path / "checkpoint_metadata.jsonl"),
                "arrays_sha256": sha256_file(tmp_path / "checkpoint.npz"),
            }
        )
    )

    restored_metadata, restored = load_checkpoint(tmp_path, rows, contract)
    assert restored_metadata == metadata
    assert len(restored["claim"]) == 1
    assert restored["claim"][0].dtype == np.float16

    atomic_jsonl(tmp_path / "checkpoint_metadata.jsonl", [{"record_key": "effusion:b"}])
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        load_checkpoint(tmp_path, rows, contract)


def test_checkpoint_arrays_use_frozen_storage_dtypes():
    arrays = checkpoint_arrays(
        [np.zeros((1, 2), dtype=np.float32)],
        [np.zeros((1, 2), dtype=np.float32)],
        [np.zeros((1, 2), dtype=np.float32)],
        [np.zeros((1, 7), dtype=np.float64)],
        [4],
    )
    assert arrays["claim"].dtype == np.float16
    assert arrays["visual_mean"].dtype == np.float16
    assert arrays["visual_std"].dtype == np.float16
    assert arrays["routing_statistics"].dtype == np.float32
    assert arrays["layers"].tolist() == [4]


def test_resume_before_first_checkpoint_is_safe_when_contract_matches(tmp_path):
    contract = "frozen-contract"
    (tmp_path / "config.json").write_text(
        json.dumps({"resume_contract_sha256": contract})
    )
    metadata, restored = load_checkpoint(
        tmp_path,
        [{"finding": "effusion", "image_id": "a"}],
        contract,
    )
    assert metadata == []
    assert all(values == [] for values in restored.values())
