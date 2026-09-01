#!/usr/bin/env python3
"""Test whether a negative CXR claim depends on anatomical coverage.

For pleural effusion, lower-lateral lung bases are the prespecified target
region and equally sized upper-lateral rectangles are the artifact control.
The probe compares target versus control attenuation of the Yes-vs-Maybe
margin on positive images and the No-vs-Maybe margin on negative images.

The positive contrast is the manipulation check.  Positive passing while
negative fails is the distinctive ``witness without coverage`` prediction:
the model uses lesion-bearing anatomy for presence but does not withdraw an
absence claim when the anatomy needed to justify it is hidden.

Labels in the bundled screening manifest are report-derived evidence grade C;
therefore this runner can prune or prioritize a hypothesis but cannot establish
a clinical result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from corrected_sgta.clinical_claims import epistemic_coordinates, softmax_states
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    sha256_file,
)


VERSION = "huatuo-negation-coverage-v1"
DEFAULT_REPO = Path("/home/dbw/ANCHOR")
DEFAULT_MANIFEST = DEFAULT_REPO / "corrected_runs/clinical_selectivity/manifest_v3.jsonl"
DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_OUTPUT = (
    DEFAULT_REPO / "corrected_runs/negation_coverage/huatuo_effusion_n32_v1"
)

# Two equal-shape masks.  Their areas are exactly equal before rasterization;
# the test suite checks the rasterized difference is at most two pixels.
TARGET_BOXES = ((0.05, 0.65, 0.40, 0.93), (0.60, 0.65, 0.95, 0.93))
CONTROL_BOXES = ((0.05, 0.12, 0.40, 0.40), (0.60, 0.12, 0.95, 0.40))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(":".join((str(seed), *values)).encode()).hexdigest()


def raster_mask(width: int, height: int, boxes=TARGET_BOXES) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        left, top = round(x0 * width), round(y0 * height)
        # Rasterize the requested span rather than rounding both endpoints;
        # translated boxes with equal normalized shape then remain exactly
        # equal-area at every resolution.
        right = min(width, left + round((x1 - x0) * width))
        bottom = min(height, top + round((y1 - y0) * height))
        mask[top:bottom, left:right] = True
    return mask


def mean_fill(image: Image.Image, boxes) -> Image.Image:
    """Replace the prespecified region by the per-channel image mean."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = raster_mask(array.shape[1], array.shape[0], boxes)
    visible = array[~mask]
    if visible.size == 0:
        raise ValueError("mask covers the full image")
    fill = np.round(visible.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    array[mask] = fill
    return Image.fromarray(array, mode="RGB")


def select_rows(manifest: Path, finding: str, seed: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in load_jsonl(manifest)
        if str(row.get("finding")) == finding
        and int(row.get("positive_votes", -1)) in {0, 3}
        and Path(str(row.get("image_path", ""))).is_file()
    ]
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(
        candidates,
        key=lambda value: stable_key(seed, str(value["image_path"]), str(value["image_id"])),
    ):
        unique.setdefault(str(Path(row["image_path"]).resolve()), row)
    rows = sorted(
        unique.values(),
        key=lambda value: (
            int(value["positive_votes"]),
            stable_key(seed, str(value["image_path"])),
        ),
    )
    counts = {state: sum(int(row["positive_votes"]) == state for row in rows) for state in (0, 3)}
    if min(counts.values()) < 8:
        raise RuntimeError(f"need at least eight unique images per class, found {counts}")
    return rows


@torch.inference_mode()
def score_image(bot: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt, tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    logits = layer_logits(bot, hidden, [len(hidden) - 1], label_ids(bot))[len(hidden) - 1]
    return {
        "logits": logits,
        "probabilities": softmax_states(logits),
        "coordinates": epistemic_coordinates(logits),
    }


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    )
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def state_margin(score: Mapping[str, Any], state: str) -> float:
    return float(score["logits"][state]) - float(score["logits"]["undetermined"])


def analyze(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [row for row in records if row.get("status") == "ok"]
    output: dict[str, Any] = {"n": len(ok), "by_reference": {}}
    effects: dict[int, np.ndarray] = {}
    for votes, state, name in ((3, "supported", "positive"), (0, "refuted", "negative")):
        rows = [row for row in ok if int(row["positive_votes"]) == votes]
        values = []
        for row in rows:
            original = state_margin(row["scores"]["original"], state)
            target = state_margin(row["scores"]["target_mask"], state)
            control = state_margin(row["scores"]["control_mask"], state)
            # Positive means the clinically relevant mask attenuates the
            # definite state more than the equal-area artifact control.
            values.append((original - target) - (original - control))
        effects[votes] = np.asarray(values, dtype=float)
        output["by_reference"][name] = {
            "state_margin": f"{state}-undetermined",
            "target_minus_control_attenuation": bootstrap(
                effects[votes], seed + votes + 1
            ),
        }
    pos = output["by_reference"]["positive"]["target_minus_control_attenuation"]
    neg = output["by_reference"]["negative"]["target_minus_control_attenuation"]
    original_correct = {
        name: float(
            np.mean(
                [
                    max(row["scores"]["original"]["probabilities"], key=row["scores"]["original"]["probabilities"].get)
                    == state
                    for row in ok
                    if int(row["positive_votes"]) == votes
                ]
            )
        )
        for votes, state, name in ((3, "supported", "positive"), (0, "refuted", "negative"))
    }
    output["original_argmax_accuracy"] = original_correct
    output["frozen_gate"] = {
        "positive_manipulation_ci_above_zero": float(pos["ci_low"]) > 0,
        "negative_coverage_ci_above_zero": float(neg["ci_low"]) > 0,
    }
    output["frozen_gate"]["witness_without_coverage_pattern"] = bool(
        output["frozen_gate"]["positive_manipulation_ci_above_zero"]
        and float(neg["ci_high"]) <= 0
    )
    output["frozen_gate"]["coverage_encoded_pattern"] = bool(
        output["frozen_gate"]["positive_manipulation_ci_above_zero"]
        and output["frozen_gate"]["negative_coverage_ci_above_zero"]
    )
    output["interpretation_guard"] = (
        "Grade-C report-derived labels and atlas rectangles only screen the mechanism. "
        "A positive manipulation check is required before interpreting the negative arm."
    )
    return output


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finding", default="pleural_effusion")
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    rows = select_rows(args.manifest, args.finding, args.seed)
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "model": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "finding": args.finding,
        "evidence_grade": "C",
        "target_boxes": TARGET_BOXES,
        "control_boxes": CONTROL_BOXES,
        "fill": "per-channel mean of unmasked pixels",
        "primary_estimand": (
            "(original-target state-vs-Maybe margin attenuation) minus "
            "(original-control attenuation), separately for positive and negative images"
        ),
        "frozen_gate": (
            "positive manipulation CI_low>0; negative CI diagnoses coverage encoding. "
            "Positive pass plus negative CI_high<=0 is witness-without-coverage."
        ),
        "selected": len(rows),
        "seed": args.seed,
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "version": VERSION,
            "image_id": row["image_id"],
            "image_path": str(Path(row["image_path"]).resolve()),
            "finding": row["finding"],
            "positive_votes": row["positive_votes"],
            "reference_source": row["reference_source"],
            "evidence_grade": row["evidence_grade"],
            "status": "error",
        }
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            variants = {
                "original": image,
                "target_mask": mean_fill(image, TARGET_BOXES),
                "control_mask": mean_fill(image, CONTROL_BOXES),
            }
            record["scores"] = {
                name: score_image(bot, variant, str(row["question"]))
                for name, variant in variants.items()
            }
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(f"[{index + 1}/{len(rows)}] {row['image_id']} {record['status']}", flush=True)

    records = load_jsonl(raw_path)
    summary = {"version": VERSION, **analyze(records, args.seed), "config": config}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
