from __future__ import annotations

import json
import zipfile

import torch

from anchor.medeval.audit_factmm_rag_archive_semantics_v1 import (
    _select_candidate,
    audit_semantics,
)
from anchor.medeval.hashing import sha256_file


def test_semantic_audit_materializes_only_unique_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "dpr.best.pt"
    torch.save({"model": {"query.weight": torch.ones(300_000)}}, checkpoint)
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(checkpoint, "retriever/output/dpr.best.pt")
    archive_hash = sha256_file(archive)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "archive_sha256": archive_hash,
                "safe_to_inventory": True,
                "candidate_checkpoint_members": ["retriever/output/dpr.best.pt"],
            }
        )
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"sha256": archive_hash}))
    result = audit_semantics(archive, inventory, provenance, tmp_path / "materialized")
    assert result["official_archive_tensor_asset_admissible"] is True
    assert result["tensor_entries"] == 1
    assert result["paper_native_retriever_identity_verified"] is False
    assert result["paper_native_end_to_end_efficacy_authorized"] is False


def test_ambiguous_equal_rank_candidates_fail_closed() -> None:
    selected, reason = _select_candidate(["a/model.pt", "b/model.pt"])
    assert selected is None
    assert "equally ranked" in reason
