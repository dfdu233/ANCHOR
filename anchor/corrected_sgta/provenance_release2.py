"""Separated center and inference identities for SGTA Wave-A release 2."""

from __future__ import annotations

from pathlib import Path

from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


def hash_named(project_root: Path, names: tuple[str, ...]) -> dict:
    output = {}
    for name in names:
        path = project_root / name
        if not path.is_file():
            raise RuntimeError(f"missing behavior-identity artifact: {path}")
        output[name] = sha256_file(path)
    return output


def center_code_identity(project_root: Path) -> dict:
    return hash_named(
        project_root,
        (
            "corrected_sgta/source_bank_v2.py",
            "corrected_sgta/source_bank_v3.py",
            "corrected_sgta/models.py",
            "corrected_sgta/models_surface.py",
            "corrected_sgta/models_alignment.py",
            "corrected_sgta/infer_ce.py",
            "corrected_sgta/build_visual_centers_v2.py",
            "corrected_sgta/build_visual_centers_release2.py",
            "corrected_sgta/provenance_v3.py",
            "corrected_sgta/provenance_release2.py",
        ),
    )


def inference_code_identity(project_root: Path) -> dict:
    identity = center_code_identity(project_root)
    identity.update(
        hash_named(
            project_root,
            (
                "corrected_sgta/cache.py",
                "corrected_sgta/protocol.py",
                "corrected_sgta/protocol_v2.py",
                "corrected_sgta/provenance_v2.py",
                "corrected_sgta/frequency_alignment_v2.py",
                "corrected_sgta/frequency_alignment_release2.py",
                "corrected_sgta/methods.py",
                "corrected_sgta/infer_alignment_v2.py",
                "corrected_sgta/infer_alignment_release2.py",
                "corrected_sgta/analyze_alignment_v2.py",
                "corrected_sgta/freeze_alignment_report_release2.py",
                "corrected_sgta/structure_audit_v2.py",
                "corrected_sgta/structure_audit_wave_a.py",
                "corrected_sgta/merge_alignment_gate_wave_a.py",
                "corrected_sgta/run_alignment_cxr_release2.sh",
                "refine-logs/EXPERIMENT_PLAN_AMENDMENT_V2.md",
            ),
        )
    )
    return identity
