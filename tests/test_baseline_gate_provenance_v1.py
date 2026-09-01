import json
from pathlib import Path

import pytest

from anchor.medeval.build_baseline_gate_provenance_v1 import checkpoint_identity


def _checkpoint(tmp_path: Path, present: tuple[str, ...]) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    }))
    for name in present:
        (root / name).write_bytes(b"weights")
    return root


def test_checkpoint_identity_requires_every_indexed_shard(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path, ("model-00001-of-00002.safetensors",))
    with pytest.raises(ValueError, match="model-00002-of-00002"):
        checkpoint_identity(root)


def test_checkpoint_identity_records_complete_inventory(tmp_path: Path) -> None:
    root = _checkpoint(tmp_path, (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ))
    identity = checkpoint_identity(root)
    assert identity["index_complete"] is True
    assert identity["index_referenced_weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
