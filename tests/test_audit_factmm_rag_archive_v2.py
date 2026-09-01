from __future__ import annotations

import zipfile

from anchor.medeval.audit_factmm_rag_archive_v2 import _unsafe_member, audit_archive


def test_archive_audit_is_inventory_only(tmp_path):
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("retriever/dpr.best.pt", b"not executed")
        archive.writestr("embeddings/train.pkl", b"not unpickled")
    result = audit_archive(path)
    assert result["safe_to_inventory"] is True
    assert result["decision"] == "inventory_only"
    assert result["candidate_checkpoint_members"] == ["retriever/dpr.best.pt"]
    assert result["candidate_embedding_members"] == ["embeddings/train.pkl"]
    assert result["retriever_checkpoint_identity_verified"] is False
    assert result["paper_native_efficacy_authorized"] is False


def test_path_traversal_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.pt", b"x")
    result = audit_archive(path)
    assert result["safe_to_inventory"] is False
    assert result["unsafe_members"] == ["../escape.pt"]
    assert _unsafe_member("/absolute.pt") is True
