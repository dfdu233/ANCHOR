from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor.corrected_sgta import compile_specificity_full_replay_manifest_v1 as compiler
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import (
    AdjudicationValidationError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(index: int) -> dict:
    target = f"A left lesion number {index} is present."
    return {
        "added_constraint_proposal": "left",
        "anatomy_stratum": "thorax",
        "answer_length_stratum": "short_le_50",
        "answer_span": target,
        "case_id": f"CASE-{index}",
        "child_proposal": target,
        "edge_id": f"EDGE-{index}",
        "edge_type": "laterality",
        "image_relpath": f"test_images/{index}.jpg",
        "modality_stratum": "XR",
        "observability_screen": "potentially_single_image_decidable",
        "parent_proposal": f"A lesion number {index} is present.",
        "prompt_requested_increment": False,
        "proposal_only": True,
        "question": "What is present?",
    }


def _final(child: str) -> dict[str, str]:
    return {
        "final_edge_entailment_admitted": "yes",
        "final_parent_visual_support": "supported",
        "final_child_visual_support": child,
        "final_increment_observability": "observable_on_supplied_image",
        "final_logical_scope_preserved": "yes",
        "final_clinical_usefulness_if_backed_off": "minor_loss",
        "final_clinically_harmful_if_wrong": "minor",
    }


def _substrate(tmp_path: Path, count: int = 8):
    repo = tmp_path / "repo"
    pack = repo / "pack"
    source_dir = repo / "source"
    pack.mkdir(parents=True)
    source_dir.mkdir()
    candidates = [_candidate(index) for index in range(count)]
    candidates_path = pack / "candidates.blinded.jsonl"
    candidates_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    private = []
    source = []
    generation = {"model": "huatuo", "seed": 42}
    generation_fingerprint = _fingerprint(generation)
    for index, candidate in enumerate(candidates, start=1):
        qid = f"Q-{index}"
        source.append(
            {
                "question_id": qid,
                "text": "Natural prefix. " + candidate["child_proposal"] + " Natural suffix.",
                "model_id": "huatuo",
                "metadata": {"fingerprint": generation_fingerprint},
            }
        )
        private.append(
            {
                "case_id": candidate["case_id"],
                "edge_id": candidate["edge_id"],
                "question_id": qid,
                "source_answer_line": index,
                "source_answer_path": "source/answers.jsonl",
                "source_model": "huatuo",
            }
        )
    private_path = pack / "provenance.private.jsonl"
    private_path.write_text("".join(json.dumps(row) + "\n" for row in private))
    source_path = source_dir / "answers.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source))
    (source_dir / "generation_config.json").write_text(
        json.dumps({**generation, "fingerprint": generation_fingerprint})
    )
    audit_source = Path(compiler.__file__).with_name(
        "audit_specificity_native_replay_substrate_v1.py"
    )
    audit = {
        "protocol": compiler.SUBSTRATE_PROTOCOL,
        "status": "passed_with_declared_edge_exclusions",
        "native_generation_sequence_certified": False,
        "gpu_identity_canary_required_after_physician_admission": True,
        "exclusions": [{"edge_id": "EDGE-0", "reason": "frozen boundary spill"}],
        "input_sha256": {
            "candidates": _sha(candidates_path),
            "private_provenance": _sha(private_path),
            "source_answers": _sha(source_path),
            "audit_source": _sha(audit_source),
        },
    }
    audit_path = repo / "audit.json"
    audit_path.write_text(json.dumps(audit))
    final_rows = {
        candidate["edge_id"]: _final("supported" if index % 2 == 0 else "refuted")
        for index, candidate in enumerate(candidates)
    }
    validated = SimpleNamespace(
        candidates=tuple(candidates),
        final_rows=final_rows,
        input_sha256={"synthetic_physicians": "frozen"},
    )
    return repo, pack, audit_path, validated


def test_full_replay_compiler_keeps_natural_answer_and_freezes_swaps(tmp_path, monkeypatch):
    repo, pack, audit, validated = _substrate(tmp_path)
    monkeypatch.setattr(compiler, "validate_adjudication", lambda *_: validated)
    output = repo / "out" / "samples.jsonl"
    metadata = repo / "out" / "metadata.json"
    result = compiler.compile_full_replay_manifest(
        pack=pack,
        repo=repo,
        substrate_audit=audit,
        output=output,
        metadata_output=metadata,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows
    assert all("parent_target" not in row and "child_target" not in row for row in rows)
    assert all(row["full_visible_answer"].startswith("Natural prefix.") for row in rows)
    assert all(len(row["swap_candidates"]) >= 2 for row in rows)
    assert all(candidate["split"] == row["split"] for row in rows for candidate in row["swap_candidates"])
    assert all(row["native_generation_sequence_certified"] is False for row in rows)
    assert "EDGE-0" not in {row["edge_id"] for row in rows}
    assert result["gpu_scoring_authorized"] is False
    assert result["isolated_parent_child_runtime_prohibited"] is True


def test_full_replay_compiler_rejects_substrate_hash_drift(tmp_path, monkeypatch):
    repo, pack, audit, validated = _substrate(tmp_path)
    monkeypatch.setattr(compiler, "validate_adjudication", lambda *_: validated)
    payload = json.loads(audit.read_text())
    payload["input_sha256"]["source_answers"] = "0" * 64
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source-answer hash drift"):
        compiler.compile_replay_rows(
            validated=validated, pack=pack, repo=repo, substrate_audit=audit
        )


def test_real_blank_pack_refuses_without_writing_full_replay(tmp_path):
    with pytest.raises(AdjudicationValidationError):
        compiler.compile_full_replay_manifest(
            pack=Path("corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2"),
            repo=Path.cwd(),
            substrate_audit=Path(
                "corrected_runs/specificity_ratchet/native_replay_substrate_audit_v1.json"
            ),
            output=tmp_path / "samples.jsonl",
            metadata_output=tmp_path / "metadata.json",
        )
    assert not list(tmp_path.iterdir())
