from __future__ import annotations

import json

import pytest

from anchor.medeval.audit_rag_dual_track_v2 import audit_dual_track


def test_v2_requires_role_correct_factmm_qualification(tmp_path) -> None:
    factmm = tmp_path / "factmm.json"
    factmm.write_text(json.dumps({"protocol_version": "factmm-rag-t0-qualification-v1"}))
    with pytest.raises(ValueError, match="role-correct"):
        audit_dual_track(tmp_path / "unused.json", factmm)
