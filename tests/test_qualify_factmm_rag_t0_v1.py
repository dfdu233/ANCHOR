import json

from anchor.medeval.qualify_factmm_rag_t0_v1 import qualify


def test_valid_tensor_does_not_bypass_missing_native_assets(tmp_path) -> None:
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
    assert result["common_protocol_retriever_asset_candidate"] is True
    assert result["paper_native_t0_status"] == "not_admissible"
    assert result["paper_native_t1_authorized"] is False
    assert "paper_native_generator_checkpoint_present" in result["missing_requirements"]
