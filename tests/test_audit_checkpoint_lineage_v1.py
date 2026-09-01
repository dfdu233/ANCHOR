from pathlib import Path

import torch
from safetensors.torch import save_file

from anchor.corrected_sgta.audit_checkpoint_lineage_v1 import (
    component,
    find_lfs_pointers,
    read_schema,
    summarize_equality,
)


def test_component_partition_is_disjoint_and_specific():
    assert component("visual.blocks.0.attn.qkv.weight") == "vision_encoder"
    assert component("visual.merger.mlp.0.weight") == "projector_merger"
    assert component("model.layers.0.self_attn.q_proj.weight") == "language_model"
    assert component("lm_head.weight") == "language_model"


def test_pointer_scan_and_strict_index_header_validation(tmp_path: Path):
    save_file({"model.x": torch.ones(2, dtype=torch.bfloat16)}, tmp_path / "model-00001-of-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":4},"weight_map":{"model.x":"model-00001-of-00001.safetensors"}}'
    )
    (tmp_path / "pointer.bin").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    )
    schema, validation = read_schema(tmp_path)
    assert schema["model.x"]["nbytes"] == 4
    assert validation["pass"] is True
    assert find_lfs_pointers(tmp_path) == ["pointer.bin"]


def test_equality_summary_separates_components():
    schema = {
        "visual.blocks.0.x": {"shape": [2], "dtype": "BF16", "nbytes": 4},
        "visual.merger.x": {"shape": [2], "dtype": "BF16", "nbytes": 4},
        "model.layers.0.x": {"shape": [2], "dtype": "BF16", "nbytes": 4},
    }
    summary = summarize_equality(
        "a",
        "b",
        schema,
        schema,
        {"visual.blocks.0.x": "same", "visual.merger.x": "a", "model.layers.0.x": "a"},
        {"visual.blocks.0.x": "same", "visual.merger.x": "b", "model.layers.0.x": "a"},
    )
    rows = {row["component"]: row for row in summary["component_equality"]}
    assert rows["vision_encoder"]["exact_tensor_rate"] == 1.0
    assert rows["projector_merger"]["exact_tensor_rate"] == 0.0
    assert rows["language_model"]["exact_tensor_rate"] == 1.0
