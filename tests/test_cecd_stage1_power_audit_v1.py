import json
from pathlib import Path

from anchor.corrected_sgta.audit_cecd_stage1_power_v1 import (
    DEFAULT_ANALYZER,
    DEFAULT_MANIFEST,
    DEFAULT_RUNNER,
    DEFAULT_SUMMARY,
    at_least_three_of_four,
    canonical_sha256,
    deterministic_selection,
    gate_row,
    load_jsonl,
    paired_auc_variance_constants,
    sha256_file,
    source_split_mismatch,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "corrected_runs/vindr_v2/cecd_stage1_power_audit_v1/power_audit.json"


def test_current_gate_cannot_be_powered_at_its_point_threshold():
    assert at_least_three_of_four(0.5) == 0.3125
    assert at_least_three_of_four(0.5) ** 2 == 0.09765625


def test_paired_auc_variance_uses_pairing_information():
    loose = paired_auc_variance_constants(0.70, 0.03, 0.90)
    tight = paired_auc_variance_constants(0.70, 0.03, 0.98)
    assert all(value > 0 for value in loose + tight)
    assert sum(tight) < sum(loose)


def test_three_stage_selection_is_exact_and_whole_image_disjoint():
    rows = load_jsonl(DEFAULT_MANIFEST)
    stages = {
        "pilot": deterministic_selection(rows, "pilot", 10),
        "dev": deterministic_selection(rows, "dev", 20),
        "confirmation": deterministic_selection(rows, "confirmation", 60),
    }
    assert {name: len(value) for name, value in stages.items()} == {
        "pilot": 160,
        "dev": 320,
        "confirmation": 960,
    }
    image_sets = {
        name: {str(row["image_id"]) for row in value}
        for name, value in stages.items()
    }
    assert image_sets["pilot"].isdisjoint(image_sets["dev"])
    assert image_sets["pilot"].isdisjoint(image_sets["confirmation"])
    assert image_sets["dev"].isdisjoint(image_sets["confirmation"])


def test_confirmation_is_only_powered_under_planning_alternative_not_mcid_boundary():
    mcid = gate_row(60, true_delta=0.03)
    planning = gate_row(60, true_delta=0.05)
    assert (
        mcid["recommended_hierarchical_components"]
        ["pooled_full_mcid_gate_two_models_independent"]
        == 0.25
    )
    assert (
        planning["recommended_hierarchical_components"]
        ["pooled_full_mcid_gate_two_models_independent"]
        > 0.90
    )


def test_source_provenance_alias_is_detected_fail_closed():
    mismatch = source_split_mismatch(
        (ROOT / DEFAULT_RUNNER).read_text(),
        (ROOT / DEFAULT_ANALYZER).read_text(),
    )
    assert mismatch["runner_packs_pilot_as_dev"] is False
    assert mismatch["analyzer_requires_dev_alias"] is False
    assert mismatch["mismatch_present"] is False


def test_committed_audit_binds_inputs_and_never_authorizes_gpu():
    artifact = json.loads(ARTIFACT.read_text())
    fingerprint = artifact.pop("fingerprint")
    assert fingerprint == canonical_sha256(artifact)
    assert artifact["gpu_authorized"] is False
    assert artifact["provenance_mismatch"]["mismatch_present"] is False
    assert artifact["manifest_contract"]["whole_image_split_verified"] is True
    assert artifact["current_gate_asymptotic_power_ceiling_at_mcid"][
        "two_models_independent"
    ] == 0.09765625
    assert artifact["inputs"]["manifest_sha256"] == sha256_file(DEFAULT_MANIFEST)
    assert artifact["inputs"]["manifest_summary_sha256"] == sha256_file(DEFAULT_SUMMARY)
    assert artifact["inputs"]["runner_source_sha256"] == sha256_file(
        ROOT / DEFAULT_RUNNER
    )
    assert artifact["inputs"]["analyzer_source_sha256"] == sha256_file(
        ROOT / DEFAULT_ANALYZER
    )
    stages = artifact["exact_stage_selections"]
    assert [row["selection_keys_sha256"] for row in stages] == [
        "276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a",
        "2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420",
        "39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9",
    ]
