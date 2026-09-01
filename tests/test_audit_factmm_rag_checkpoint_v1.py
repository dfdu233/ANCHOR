import json

import torch

from anchor.medeval.audit_factmm_rag_checkpoint_v1 import audit
from anchor.medeval.hashing import sha256_file


def test_checkpoint_audit_requires_hash_bound_model_tensors(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {"query.weight": torch.ones(300_000)}}, checkpoint)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"sha256": sha256_file(checkpoint)}))
    result = audit(checkpoint, provenance)
    assert result["paper_native_retriever_asset_admissible"] is True
    assert result["tensor_entries"] == 1
    assert result["paper_native_end_to_end_efficacy_authorized"] is False
