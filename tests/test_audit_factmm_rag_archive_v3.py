from __future__ import annotations

import json
import zipfile

import torch

from anchor.medeval.audit_factmm_rag_archive_semantics_v2 import audit_semantics
from anchor.medeval.audit_factmm_rag_archive_v3 import audit_archive
from anchor.medeval.hashing import sha256_file


def _make_sharded_archive(tmp_path):
    first = tmp_path / "pytorch_model-00001-of-00002.bin"
    second = tmp_path / "pytorch_model-00002-of-00002.bin"
    torch.save({"model.layers.0.weight": torch.ones(300_000)}, first)
    torch.save(
        {
            "lm_head.weight": torch.ones(300_000),
            "model.mm_projector.0.weight": torch.ones(1),
            "model.vision_tower.block.weight": torch.ones(1),
        },
        second,
    )
    index = {
        "metadata": {"total_size": first.stat().st_size + second.stat().st_size},
        "weight_map": {
            "model.layers.0.weight": first.name,
            "lm_head.weight": second.name,
            "model.mm_projector.0.weight": second.name,
            "model.vision_tower.block.weight": second.name,
        },
    }
    config = {
        "architectures": ["LlavaLlamaForCausalLM"],
        "model_type": "llava",
        "_name_or_path": "fixture",
        "mm_vision_tower": "fixture-vision",
    }
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(first, f"release/{first.name}")
        handle.write(second, f"release/{second.name}")
        handle.writestr("release/pytorch_model.bin.index.json", json.dumps(index))
        handle.writestr("release/config.json", json.dumps(config))
    return archive


def test_sharded_checkpoint_is_one_complete_group(tmp_path) -> None:
    archive = _make_sharded_archive(tmp_path)
    result = audit_archive(archive)
    assert result["complete_checkpoint_group_count"] == 1
    assert len(result["candidate_checkpoint_members"]) == 2
    group = result["sharded_checkpoint_groups"][0]
    assert group["complete"] is True
    assert group["tensor_entries_in_index"] == 4


def test_semantic_audit_classifies_generator_not_retriever(tmp_path) -> None:
    archive = _make_sharded_archive(tmp_path)
    inventory_payload = audit_archive(archive)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(inventory_payload))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"sha256": sha256_file(archive)}))
    result = audit_semantics(archive, inventory, provenance, tmp_path / "materialized")
    assert result["safe_weights_only_load"] is True
    assert result["tensor_entries_loaded"] == 4
    assert result["paper_native_generator_identity_verified"] is True
    assert result["paper_native_projector_present"] is True
    assert result["paper_native_vision_tower_present"] is True
    assert result["paper_native_retriever_identity_verified"] is False
    assert result["decision"] == "generator_tensor_asset_only"
