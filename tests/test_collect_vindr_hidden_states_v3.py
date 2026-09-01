import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from corrected_sgta.collect_vindr_hidden_states_v3 import (
    aggregate_shards,
    freeze_or_validate_run,
    shard_path,
    validate_ordered_keys,
    validate_shard_set,
    write_case_shard,
)


LAYERS = [1, 2]


def fake_features(offset: float):
    return {
        layer: {
            "claim": np.asarray([offset + layer, offset + layer + 0.5], dtype=np.float32),
            "visual_mean": np.asarray(
                [offset + layer + 1.0, offset + layer + 1.5], dtype=np.float32
            ),
            "visual_std": np.asarray(
                [offset + layer + 2.0, offset + layer + 2.5], dtype=np.float32
            ),
            "routing_statistics": np.arange(7, dtype=np.float32) + offset + layer,
        }
        for layer in LAYERS
    }


def frozen_run(tmp_path: Path, keys=("effusion:image-a", "nodule:image-b")):
    output = tmp_path / "run"
    static = {
        "version": "test-v3",
        "manifest_sha256": "manifest-hash",
        "model_inventory": [{"name": "weights", "sha256": "model-hash"}],
        "layers": LAYERS,
    }
    config = freeze_or_validate_run(
        output, static, keys, resume=False, command="initial command"
    )
    return output, static, config


def write_shard(output: Path, config: dict, keys, index: int):
    key = keys[index]
    write_case_shard(
        shard_path(output / "shards", index, key),
        index=index,
        record_key=key,
        config_fingerprint=config["fingerprint"],
        layers=LAYERS,
        features=fake_features(float(index * 10)),
        metadata={"record_key": key, "elapsed_seconds": index + 0.25},
    )


def test_resume_and_aggregation_are_strict_and_deterministic(tmp_path: Path):
    keys = ["effusion:image-a", "nodule:image-b"]
    output, static, config = frozen_run(tmp_path, keys)
    # Physical completion order is intentionally the reverse of manifest order.
    write_shard(output, config, keys, 1)
    write_shard(output, config, keys, 0)

    resumed = freeze_or_validate_run(
        output, static, keys, resume=True, command="same run --resume"
    )
    assert resumed["fingerprint"] == config["fingerprint"]
    assert validate_shard_set(output, keys, config["fingerprint"], LAYERS) == []

    metadata = aggregate_shards(output, keys, config["fingerprint"], LAYERS)
    first_hash = hashlib.sha256((output / "hidden_states.npz").read_bytes()).hexdigest()
    aggregate_shards(output, keys, config["fingerprint"], LAYERS)
    second_hash = hashlib.sha256((output / "hidden_states.npz").read_bytes()).hexdigest()
    assert second_hash == first_hash
    assert [row["record_key"] for row in metadata] == keys
    assert [json.loads(line)["record_key"] for line in (output / "metadata.jsonl").read_text().splitlines()] == keys
    with np.load(output / "hidden_states.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "claim",
            "visual_mean",
            "visual_std",
            "routing_statistics",
            "routing_statistic_names",
            "layers",
        }
        assert archive["claim"].shape == (2, 2, 2)
        assert archive["claim"].dtype == np.float16
        assert archive["routing_statistics"].dtype == np.float32
        assert archive["claim"][0, 0, 0] == np.float16(1.0)
        assert archive["claim"][1, 0, 0] == np.float16(11.0)


def test_duplicate_keys_are_rejected_before_artifact_creation(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate record keys"):
        validate_ordered_keys(["same", "same"])
    with pytest.raises(ValueError, match="duplicate record keys"):
        freeze_or_validate_run(
            tmp_path / "run",
            {"version": "test"},
            ["same", "same"],
            resume=False,
            command="test",
        )
    assert not (tmp_path / "run").exists()


def test_resume_rejects_config_and_order_drift(tmp_path: Path):
    keys = ["effusion:image-a", "nodule:image-b"]
    output, static, _config = frozen_run(tmp_path, keys)
    with pytest.raises(ValueError, match="config drift"):
        freeze_or_validate_run(
            output,
            {**static, "layers": [1, 3]},
            keys,
            resume=True,
            command="resume",
        )
    with pytest.raises(ValueError, match="ordered record-key drift"):
        freeze_or_validate_run(
            output,
            static,
            list(reversed(keys)),
            resume=True,
            command="resume",
        )


def test_missing_and_corrupt_shards_fail_closed(tmp_path: Path):
    keys = ["effusion:image-a", "nodule:image-b"]
    output, _static, config = frozen_run(tmp_path, keys)
    write_shard(output, config, keys, 0)
    assert validate_shard_set(output, keys, config["fingerprint"], LAYERS) == [1]
    with pytest.raises(FileNotFoundError, match="partial aggregation"):
        aggregate_shards(output, keys, config["fingerprint"], LAYERS)

    bad_path = shard_path(output / "shards", 0, keys[0])
    bad_path.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="invalid shard"):
        validate_shard_set(output, keys, config["fingerprint"], LAYERS)
