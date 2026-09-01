from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from PIL import Image

from anchor.corrected_sgta.observation_policy_probe_v1 import (
    PROMPTS,
    RENDERS,
    analyze,
    centered_roi_box,
    question,
    render_variants,
)


def test_centered_roi_contains_bbox_center_and_keeps_fixed_fov() -> None:
    box = centered_roi_box((1000, 800), (20.0, 780.0), 0.5)
    assert box == (0, 400, 500, 800)
    assert box[0] <= 20 <= box[2]
    assert box[1] <= 780 <= box[3]


def test_render_factorial_reuses_identical_pixels_across_prompts() -> None:
    full = Image.new("RGB", (200, 160), (80, 90, 100))
    renders = render_variants(
        full,
        (50, 40, 150, 120),
        seed=7,
        blur_radius_fraction=0.025,
    )
    assert tuple(renders) == RENDERS
    assert all(image.size == full.size for image in renders.values())
    digest = hashlib.sha256(renders["zoom_sham_panel"].tobytes()).hexdigest()
    # Prompt framing never calls the renderer and therefore cannot alter it.
    assert len({digest for prompt in PROMPTS if question("nodule_mass", prompt)}) == 1
    assert len({question("nodule_mass", prompt) for prompt in PROMPTS}) == 3


def test_analyzer_detects_crop_specific_provenance_interaction(tmp_path) -> None:
    selections = []
    raw = []
    qid = 1
    for label in (0, 1):
        for index in range(40):
            sample = f"{'positive' if label else 'negative'}-{index}"
            for render in RENDERS:
                render_hash = hashlib.sha256(f"{sample}:{render}".encode()).hexdigest()
                for prompt in PROMPTS:
                    if label == 0:
                        margin = -1.0
                        if render == "zoom_sham_panel":
                            margin = 1.0 if prompt != "random_provenance" else -1.0
                        elif prompt == "suspicious_provenance":
                            margin = -0.5
                    else:
                        margin = 1.0
                    selections.append({
                        "qid": qid,
                        "sample_id": sample,
                        "image_id": sample,
                        "finding": "nodule_mass",
                        "label": label,
                        "render": render,
                        "prompt": prompt,
                        "render_sha256": render_hash,
                    })
                    raw.append({
                        "question_id": qid,
                        "status": "ok",
                        "image_sha256": render_hash,
                        "scores": {"original_margin": margin},
                    })
                    qid += 1
    selections_path = tmp_path / "selections.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "analysis.json"
    selections_path.write_text("".join(json.dumps(row) + "\n" for row in selections))
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw))
    analyze(SimpleNamespace(
        selections=selections_path,
        raw=raw_path,
        output=output_path,
        model="synthetic",
        bootstrap_draws=200,
        seed=42,
    ))
    result = json.loads(output_path.read_text())
    assert result["pixel_identity_across_prompt_counterfactuals"] is True
    assert result["primary"]["single_model_gate"] is True
    assert result["primary"]["gamma_random_margin"]["mean"] == 2.0
    assert result["primary"]["negative_crop_fp_drop_neutral_to_random"]["mean"] == 1.0
    assert result["primary"]["positive_crop_recall_loss_neutral_to_random"]["mean"] == 0.0
