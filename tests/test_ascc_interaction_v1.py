from __future__ import annotations

from collections import Counter

import pytest

from anchor.corrected_sgta.prepare_ascc_interaction_v1 import (
    EDGES,
    PROMPTS,
    exact_local_matched_sample,
)
from anchor.corrected_sgta.run_huatuo_ascc_interaction_v1 import marker_coordinates
from anchor.corrected_sgta.analyze_ascc_interaction_v1 import analyze_edge


def _row(image_id: str, edge_id: str, parent: int, child: int, aspect: str = "wide"):
    edge = next(edge for edge in EDGES if edge.edge_id == edge_id)
    return {
        "image_id": image_id,
        "item_id": f"{image_id}:{edge_id}",
        "edge_id": edge_id,
        "fixed_prefix": edge.prefix,
        "parent_votes": parent,
        "child_votes": child,
        "child_support_stratum": f"reader_{child}of3",
        "aspect_bucket": aspect,
    }


def test_commitment_and_polarity_coordinates_are_independent_linear_planes() -> None:
    coordinates = marker_coordinates(
        {" unlikely": 1.0, " possible": 2.0, " present": 5.0}
    )
    assert coordinates["commitment"] == pytest.approx(1.0)
    assert coordinates["polarity"] == pytest.approx(4.0)
    assert coordinates["positive_overcommitment"] == pytest.approx(3.0)
    assert coordinates["negative_overcommitment"] == pytest.approx(-1.0)


def test_every_prompt_pair_has_both_framings_and_shared_marker_instruction() -> None:
    groups: dict[str, set[str]] = {}
    for prompt in PROMPTS:
        groups.setdefault(prompt["prompt_pair_id"], set()).add(prompt["framing"])
        assert "unlikely, possible, or present" in prompt["prompt"]
        assert "Present, absent, and uncertain states are all valid" in prompt["prompt"]
    assert groups == {
        "a": {"neutral", "existential"},
        "b": {"neutral", "existential"},
    }


def test_exact_local_matching_preserves_parent_and_acquisition_nuisances() -> None:
    edge_id = EDGES[0].edge_id
    rows = []
    for child in range(4):
        for index in range(4):
            rows.append(_row(f"{child}-{index}", edge_id, 2, child))
    selected, diagnostics = exact_local_matched_sample(rows, seed=7, maximum_pairs=3)
    assert len(selected) == 12
    counts = Counter(
        (row["comparison_family"], row["child_votes"]) for row in selected
    )
    assert set(counts.values()) == {3}
    pairs = {}
    for row in selected:
        pairs.setdefault(row["matched_pair_id"], []).append(row)
    assert len(pairs) == 6
    for pair in pairs.values():
        assert len(pair) == 2
        assert {row["parent_votes"] for row in pair} == {2}
        assert {row["aspect_bucket"] for row in pair} == {"wide"}
        votes = {row["child_votes"] for row in pair}
        assert votes in ({0, 1}, {2, 3})
    assert (
        diagnostics[edge_id]["families"]["negative_boundary"][
            "selected_exact_matched_pairs"
        ]
        == 3
    )
    assert (
        diagnostics[edge_id]["families"]["positive_boundary"][
            "selected_exact_matched_pairs"
        ]
        == 3
    )


def test_analysis_accepts_selective_ambiguity_shift_with_stable_polarity() -> None:
    rows = []
    shards = {}
    for family, votes in (("negative_boundary", (0, 1)), ("positive_boundary", (2, 3))):
        for pair_index in range(4):
            pair_id = f"{family}-{pair_index}"
            for vote in votes:
                item_id = f"{pair_id}-{vote}"
                rows.append(
                    {
                        "item_id": item_id,
                        "matched_pair_id": pair_id,
                        "comparison_family": family,
                        "child_votes": vote,
                    }
                )
                for prompt in PROMPTS:
                    commitment = (
                        1.0
                        if prompt["framing"] == "existential" and vote in {1, 2}
                        else 0.0
                    )
                    shards[(item_id, prompt["name"])] = {
                        "layer_scores": {
                            "decoder_final": {
                                "coordinates": {
                                    "commitment": commitment,
                                    "polarity": float(vote),
                                }
                            }
                        }
                    }
    result = analyze_edge(rows, shards, seed=9, iterations=1000)
    assert result["layers"]["decoder_final"]["commitment_did"] == pytest.approx(1.0)
    assert result["gate"]["ascc_behavioral_gate_passed"] is True
