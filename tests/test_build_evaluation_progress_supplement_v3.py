from __future__ import annotations

import json

import pytest

from anchor.medeval.build_evaluation_progress_supplement_v3 import build


def test_v3_rejects_old_rag_audit_before_loading_other_inputs(tmp_path) -> None:
    rag = tmp_path / "rag.json"
    rag.write_text(json.dumps({"protocol_version": "rag-dual-track-qualification-v1"}))
    with pytest.raises(ValueError, match="role-correct"):
        build(
            base_audit=tmp_path / "unused",
            rag_audit=rag,
            v1_failure=tmp_path / "unused",
            v2_execution=tmp_path / "unused",
            v2_provenance=tmp_path / "unused",
            v1_job_state=tmp_path / "unused",
            v2_job_state=tmp_path / "unused",
        )
