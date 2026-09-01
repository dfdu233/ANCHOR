from __future__ import annotations

import json

from anchor.medeval.qualify_factmm_rag_t0_v2 import qualify


def test_generator_release_does_not_substitute_for_retriever(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "name": "FactMM-RAG",
                        "source_fingerprint": {"sha256": "source"},
                        "license_sha256": "license",
                    }
                ]
            }
        )
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"safe_to_inventory": True}))
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "official_archive_tensor_asset_admissible": True,
                "paper_native_generator_identity_verified": True,
                "paper_native_projector_present": True,
                "paper_native_vision_tower_present": True,
                "paper_native_retriever_identity_verified": False,
            }
        )
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    result = qualify(
        source,
        inventory,
        semantic,
        repository,
        tmp_path / "missing-mimic",
        tmp_path / "missing-chexpert",
    )
    assert result["released_asset_role"] == "generator"
    assert result["requirements"]["generator_tensor_asset"] is True
    assert result["requirements"]["retriever_role_identity_verified"] is False
    assert result["common_protocol_retriever_asset_candidate"] is False
    assert result["paper_native_t0_status"] == "not_admissible"
