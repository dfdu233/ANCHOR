import json
from pathlib import Path

import pytest

from anchor.corrected_sgta.analyze_dicom_render_pilot_v1 import (
    AnalysisInputError,
    analyze,
    deterministic_dev_half,
    select_and_confirm_transform,
)


def _audit(name: str, passed: bool) -> dict:
    return {
        "pixel_sha256": (name.encode().hex() + "0" * 64)[:64],
        "finite_fraction": 1.0,
        "bbox_retention": True,
        "roi_saturation_fraction": 0.01,
        "display_edge_correlation_with_baseline": 1.0,
        "clinical_guard_pass": passed,
    }


def _write_synthetic_run(
    root: Path, native_guard: bool = True, native_effect: float = 1.2
) -> None:
    root.mkdir(parents=True)
    (root / "shards").mkdir()
    selection_keys = []
    fingerprint = "synthetic-fingerprint"
    for finding_index in range(4):
        finding = f"finding_{finding_index}"
        for votes in range(4):
            for replicate in range(5):
                image_id = f"{finding}-{votes}-{replicate}"
                selection_keys.append(image_id)
                nuisance = (replicate - 2) * 0.01
                baseline = votes * 2.0 + nuisance
                shard = {
                    "status": "ok",
                    "record_key": image_id,
                    "config_fingerprint": fingerprint,
                    "image_id": image_id,
                    "finding": finding,
                    "reader_votes": votes,
                    "views": {
                        "baseline_percentile": {
                            "scores": {"polarity": baseline, "commitment": 1.0},
                            "prediction": "yes" if baseline > 0 else "no",
                            "audit": _audit("baseline", True),
                        },
                        "native_linear": {
                            "scores": {
                                "polarity": baseline + native_effect,
                                "commitment": 1.1,
                            },
                            "prediction": "yes",
                            "audit": _audit("native", native_guard),
                            "is_primary": True,
                        },
                        "identity_lossless_duplicate": {
                            "scores": {"polarity": baseline, "commitment": 1.0},
                            "prediction": "yes" if baseline > 0 else "no",
                            "audit": _audit("duplicate", True),
                            "is_primary": False,
                            "track": "identity_control",
                        },
                        # Deliberately huge and incorrectly declared primary: its
                        # name must hard-exclude it from every formal gate.
                        "polarity_toggle": {
                            "scores": {"polarity": baseline + 100.0, "commitment": 9.0},
                            "prediction": "yes",
                            "audit": _audit("toggle", True),
                            "is_primary": True,
                        },
                    },
                }
                (root / "shards" / f"{image_id}.json").write_text(json.dumps(shard))
    (root / "config.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic-vindr",
                "model": "synthetic-huatuo",
                "baseline_view": "baseline_percentile",
                "seed": 7,
                "split": "pilot",
                "selected_claims": len(selection_keys),
                "selection_keys": selection_keys,
                "fingerprint": fingerprint,
            }
        )
    )
    (root / "run_state.json").write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "selected_claims": len(selection_keys),
                "complete_shards": len(selection_keys),
                "error_shards_this_invocation": 0,
            }
        )
    )


def test_reader_equivalent_three_of_four_gate_and_secondary_exclusion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    report = analyze(
        run_dir=run_dir,
        repetitions=100,
        seed=11,
        minimum_audit_rate=0.95,
        min_per_bin=5,
        median_re_threshold=0.5,
        median_re_ci_threshold=0.25,
        one_step_fraction_threshold=0.2,
        high_margin_min_re=0.25,
        high_margin_ci_min_re=0.1,
    )

    assert report["formal_overall_gate"]["passed"]
    assert report["evidence_tier"] == "exploratory_pilot_only"
    assert report["paper_claim_authorized"] is False
    assert report["clinical_transform_audit"]["eligible_primary_transforms"] == [
        "native_linear"
    ]
    toggle = report["clinical_transform_audit"]["transforms"]["polarity_toggle"]
    assert toggle["hard_secondary_exclusion"]
    assert not toggle["gate_eligible"]
    assert not report["formal_overall_gate"]["flip_rate_used"]
    for finding in report["findings"].values():
        assert finding["finding_gate"]["passed"]
        beta = finding["reader_step_beta"]
        assert 1.9 < beta["estimate"] < 2.1
        assert finding["primary_continuous_orbit"]["median_reader_equivalent"][
            "estimate"
        ] > 0.5


def test_failed_clinical_audit_cannot_be_rescued_by_inversion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, native_guard=False)
    report = analyze(
        run_dir=run_dir,
        repetitions=20,
        seed=13,
        minimum_audit_rate=0.95,
        min_per_bin=5,
        median_re_threshold=0.5,
        median_re_ci_threshold=0.25,
        one_step_fraction_threshold=0.2,
        high_margin_min_re=0.25,
        high_margin_ci_min_re=0.1,
    )

    assert report["clinical_transform_audit"]["eligible_primary_transforms"] == []
    assert not report["formal_overall_gate"]["passed"]
    assert all(
        not item["finding_gate"]["passed"] for item in report["findings"].values()
    )


def test_tiny_but_significant_heldout_effect_fails_reader_equivalent_magnitude(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, native_effect=0.2)  # beta=2, so heldout RE=0.1
    report = analyze(
        run_dir=run_dir,
        repetitions=100,
        seed=11,
        minimum_audit_rate=0.95,
        min_per_bin=5,
        median_re_threshold=0.5,
        median_re_ci_threshold=0.25,
        one_step_fraction_threshold=0.2,
        high_margin_min_re=0.25,
        high_margin_ci_min_re=0.1,
    )
    for finding in report["findings"].values():
        confirmation = finding["deterministic_transform_selection_and_confirmation"]
        assert not confirmation["reader_equivalent_magnitude_pass"]
        assert not finding["finding_gate"]["passed"]


def test_inconsistent_heldout_signs_fail_agreement_gate() -> None:
    finding = "finding"
    seed = 123
    rows = []
    for votes in range(4):
        for replicate in range(20):
            baseline = 2.0 * votes + 0.001 * replicate
            rows.append(
                {
                    "image_id": f"{votes}-{replicate}",
                    "finding": finding,
                    "reader_votes": votes,
                    "baseline_polarity": baseline,
                    "baseline_commitment": 1.0,
                    "baseline_abs_polarity_margin": abs(baseline),
                    "views": {
                        "native_linear": {
                            "polarity": baseline + 1.2,
                            "commitment": 1.0,
                            "clinical_guard_pass": True,
                        },
                        "identity_lossless_duplicate": {
                            "polarity": baseline,
                            "commitment": 1.0,
                            "clinical_guard_pass": True,
                        },
                    },
                }
            )
    heldout = [row for row in rows if deterministic_dev_half(row, seed, finding) == "B"]
    positive_count = int(0.60 * len(heldout))
    for index, row in enumerate(heldout):
        effect = 1.2 if index < positive_count else -1.2
        row["views"]["native_linear"]["polarity"] = row["baseline_polarity"] + effect

    result = select_and_confirm_transform(
        rows=rows,
        finding=finding,
        primary_names=["native_linear"],
        identity_name="identity_lossless_duplicate",
        beta=2.0,
        repetitions=100,
        seed=seed,
        minimum_audit_rate=0.95,
        heldout_min_abs_re=0.5,
        heldout_re_ci_min_magnitude=0.25,
        heldout_sign_agreement_threshold=0.65,
    )
    assert result["heldout_half_b_sign_agreement"] < 0.65
    assert not result["heldout_sign_agreement_pass"]
    assert not result["pass"]


def test_formal_analysis_refuses_partial_or_foreign_shard_sets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    shard = next((run_dir / "shards").glob("*.json"))
    shard.unlink()
    with pytest.raises(AnalysisInputError, match="exactly"):
        analyze(
            run_dir=run_dir,
            repetitions=20,
            seed=11,
            minimum_audit_rate=0.95,
            min_per_bin=5,
            median_re_threshold=0.5,
            median_re_ci_threshold=0.25,
            one_step_fraction_threshold=0.2,
            high_margin_min_re=0.25,
            high_margin_ci_min_re=0.1,
        )
