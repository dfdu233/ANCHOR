from __future__ import annotations

from collections import Counter

from corrected_sgta.analyze_commitment_tetrads import analyze_commitment_tetrads
from corrected_sgta.prepare_vindr_commitment_tetrads import (
    ROLE_ORDER,
    build_commitment_tetrads,
)


def _manifest_row(image_id: str, votes: int, split: str = "dev") -> dict[str, object]:
    return {
        "dataset": "vindr-cxr-1.0.0",
        "reference_source": "vindr_reader_votes",
        "evidence_grade": "A",
        "formal_reference": True,
        "image_id": image_id,
        "finding": "pleural_effusion",
        "positive_votes": votes,
        "reader_count": 3,
        "reader_support": votes / 3,
        "reader_state": (
            "refuted" if votes == 0 else "supported" if votes == 3 else "undetermined"
        ),
        "experiment_split": split,
        "dicom_relpath": f"train/{image_id}.dicom",
        "dicom_metadata": {
            "view_position": "pa",
            "manufacturer": "vendor",
            "manufacturer_model": "model",
            "rows": 2048,
            "columns": 2048,
            "aspect_ratio": 1.0,
        },
    }


def test_commitment_tetrads_are_balanced_and_globally_image_disjoint():
    rows = [
        _manifest_row(f"vote-{votes}-image-{index}", votes)
        for votes in range(4)
        for index in range(4)
    ]
    tetrads, summary = build_commitment_tetrads(
        rows, seed=42, match_manufacturer=True, max_tetrads_per_branch=None
    )
    assert summary["tetrads"] == 4
    assert summary["records"] == 16
    assert summary["role_counts"] == {role: 4 for role in ROLE_ORDER}
    assert summary["branch_counts"] == {"negative": 2, "positive": 2}
    assert len({row["image_id"] for row in tetrads}) == len(tetrads)
    for tetrad_id in {row["tetrad_id"] for row in tetrads}:
        members = [row for row in tetrads if row["tetrad_id"] == tetrad_id]
        assert Counter(row["tetrad_role"] for row in members) == Counter(ROLE_ORDER)
        branch = members[0]["majority_polarity"]
        votes = {int(row["positive_votes"]) for row in members}
        assert votes == ({0, 1} if branch == "negative" else {2, 3})


def _logits(polarity: float, commitment: float = 1.0) -> dict[str, float]:
    # (Yes-No)/2 = polarity and (Yes+No)/2-Maybe = commitment.
    return {
        "supported": polarity,
        "refuted": -polarity,
        "undetermined": -commitment,
    }


def _synthetic_tetrad(
    tetrad_id: str, branch: str, split: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sign = 1.0 if branch == "positive" else -1.0
    votes = {
        "clear_a": 3 if branch == "positive" else 0,
        "clear_b": 3 if branch == "positive" else 0,
        "ambiguous_a": 2 if branch == "positive" else 1,
        "ambiguous_b": 2 if branch == "positive" else 1,
    }
    early_magnitudes = {
        "clear_a": 3.0,
        "clear_b": 2.0,
        "ambiguous_a": 0.5,
        "ambiguous_b": 0.4,
    }
    manifest = []
    raw = []
    for role in ROLE_ORDER:
        image_id = f"{tetrad_id}-{role}"
        manifest.append(
            {
                **_manifest_row(image_id, votes[role], split),
                "tetrad_id": tetrad_id,
                "tetrad_role": role,
                "majority_polarity": branch,
            }
        )
        raw.append(
            {
                "image_id": image_id,
                "finding": "pleural_effusion",
                "status": "ok",
                "measurement": {
                    "trajectory": {
                        "1": {
                            "real_logits": _logits(sign * early_magnitudes[role]),
                            "baseline_state": "supported" if sign > 0 else "refuted",
                        },
                        "2": {
                            "real_logits": _logits(sign * 1.0),
                            "baseline_state": "supported" if sign > 0 else "refuted",
                        },
                    }
                },
            }
        )
    return manifest, raw


def test_tetrad_analyzer_detects_layerwise_reader_support_erasure():
    manifest = []
    raw = []
    for branch in ("negative", "positive"):
        rows, outputs = _synthetic_tetrad(f"dev-{branch}", branch, "dev")
        manifest.extend(rows)
        raw.extend(outputs)
        for index in range(12):
            rows, outputs = _synthetic_tetrad(
                f"test-{branch}-{index}", branch, "test"
            )
            manifest.extend(rows)
            raw.extend(outputs)
    result = analyze_commitment_tetrads(
        manifest, raw, bootstrap_draws=200, seed=42
    )
    assert result["selected_early_layer"] == 1
    assert result["test_layer_metrics"]["1"]["support_macro_auroc"] == 1.0
    assert result["test_layer_metrics"]["2"]["support_macro_auroc"] == 0.5
    assert result["heldout_tests"]["selected_early_minus_final_support_auroc"][
        "estimate"
    ] == 0.5
    assert result["mechanism_gates"]["observational_erasure_authorized"] is True


def test_tetrad_analyzer_rejects_when_final_layer_retains_reader_support():
    manifest = []
    raw = []
    for branch in ("negative", "positive"):
        rows, outputs = _synthetic_tetrad(f"dev-{branch}", branch, "dev")
        manifest.extend(rows)
        raw.extend(outputs)
        for index in range(12):
            rows, outputs = _synthetic_tetrad(
                f"test-{branch}-{index}", branch, "test"
            )
            for output in outputs:
                output["measurement"]["trajectory"]["2"]["real_logits"] = dict(
                    output["measurement"]["trajectory"]["1"]["real_logits"]
                )
            manifest.extend(rows)
            raw.extend(outputs)
    result = analyze_commitment_tetrads(
        manifest, raw, bootstrap_draws=100, seed=7
    )
    assert result["heldout_tests"]["selected_early_minus_final_support_auroc"][
        "estimate"
    ] == 0.0
    assert result["mechanism_gates"]["observational_erasure_authorized"] is False
