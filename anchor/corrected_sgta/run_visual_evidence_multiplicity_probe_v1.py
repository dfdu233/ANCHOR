#!/usr/bin/env python3
"""Fatal probe for visual-evidence multiplicity in HuatuoGPT-Vision.

The probe repeats the *same projected visual-token block* 1/2/4/8 times.
It therefore changes neither pixels nor clinical content.  Two position modes
separate pure multiplicity (the copies reuse the original position ids) from
the ordinary multi-view implementation effect (positions are sequential).

This is a mechanism gate, not a mitigation result.  If repeated evidence does
not systematically increase positive commitment on reader-unanimous negative
VinDr claims, crop-induced false positives cannot be attributed to evidence
multiplicity and the Visual Delta Rule branch must close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
)


VERSION = "visual-evidence-multiplicity-probe-v1"
FACTORS = (1, 2, 4, 8)
POSITION_MODES = ("tied", "sequential")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def stable_key(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}:{row['sample_id']}:{row['finding']}"
    return hashlib.sha256(value.encode()).hexdigest()


def load_cases(
    selections_path: Path,
    image_root: Path,
    label: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in selections_path.read_text().splitlines()
        if line.strip()
    ]
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if (
            int(row["label"]) == label
            and row["render"] == "full"
            and row["prompt"] == "neutral"
        ):
            key = (str(row["sample_id"]), str(row["finding"]))
            selected.setdefault(key, row)
    cases = sorted(selected.values(), key=lambda row: stable_key(seed, row))[:limit]
    if len(cases) != limit:
        raise ValueError(f"requested {limit} label={label} cases, found {len(cases)}")
    output = []
    for row in cases:
        path = image_root / str(row["img_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(
            {
                "sample_id": str(row["sample_id"]),
                "image_id": str(row["image_id"]),
                "finding": str(row["finding"]),
                "label": int(row["label"]),
                "image": str(path.resolve()),
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_qid": int(row["qid"]),
            }
        )
    return output


def prompt_for(finding: str) -> str:
    return (
        f"Does this chest X-ray show {finding.replace('_', ' ')}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    )


def repeat_visual_block(
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    position_ids: torch.Tensor | None,
    visual_span: tuple[int, int],
    factor: int,
    position_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if factor not in FACTORS:
        raise ValueError(f"unsupported repetition factor: {factor}")
    if position_mode not in POSITION_MODES:
        raise ValueError(f"unsupported position mode: {position_mode}")
    start, end = visual_span
    prefix, visual, suffix = (
        embeddings[:, :start],
        embeddings[:, start:end],
        embeddings[:, end:],
    )
    repeated = torch.cat([prefix, visual.repeat(1, factor, 1), suffix], dim=1)
    repeated_attention = torch.ones(
        repeated.shape[:2], device=attention.device, dtype=attention.dtype
    )
    if position_ids is None:
        original_positions = torch.arange(
            embeddings.shape[1], device=embeddings.device, dtype=torch.long
        ).unsqueeze(0)
    else:
        original_positions = position_ids
    if factor == 1:
        repeated_positions = original_positions
    elif position_mode == "tied":
        repeated_positions = torch.cat(
            [
                original_positions[:, :start],
                original_positions[:, start:end].repeat(1, factor),
                original_positions[:, end:],
            ],
            dim=1,
        )
    else:
        first = int(original_positions[0, 0])
        repeated_positions = torch.arange(
            first,
            first + repeated.shape[1],
            device=original_positions.device,
            dtype=original_positions.dtype,
        ).unsqueeze(0)
    if repeated_positions.shape != repeated_attention.shape:
        raise AssertionError("position/attention shape drift")
    return repeated, repeated_attention, repeated_positions


@torch.inference_mode()
def score_condition(
    bot: Any,
    ids: dict[str, int],
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, Any]:
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    final_layer = len(hidden) - 1
    logits = layer_logits(bot, hidden, (), ids)[final_layer]
    values = np.asarray(
        [logits["supported"], logits["refuted"], logits["undetermined"]],
        dtype=float,
    )
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    margin = float(logits["supported"] - logits["refuted"])
    return {
        "logits": logits,
        "yes_minus_no": margin,
        "commitment": float(1.0 - probabilities[2]),
        "entropy": float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))),
        "prediction": "yes" if margin > 0 else "no",
        "probabilities": {
            "yes": float(probabilities[0]),
            "no": float(probabilities[1]),
            "maybe": float(probabilities[2]),
        },
    }


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap needs a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for index in range(draws):
        means[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def monotonic_fraction(rows: Iterable[dict[str, Any]], mode: str) -> float:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["position_mode"] == mode and row["status"] == "ok":
            grouped[str(row["case_key"])][int(row["factor"])] = float(
                row["score"]["yes_minus_no"]
            )
    valid = [values for values in grouped.values() if set(values) == set(FACTORS)]
    return float(
        np.mean(
            [all(values[a] <= values[b] for a, b in zip(FACTORS, FACTORS[1:])) for values in valid]
        )
    )


def analyze(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    by_mode: dict[str, Any] = {}
    for mode in POSITION_MODES:
        selected = [row for row in ok if row["position_mode"] == mode]
        by_factor = {}
        for factor in FACTORS:
            cell = [row for row in selected if int(row["factor"]) == factor]
            margins = np.asarray([row["score"]["yes_minus_no"] for row in cell])
            by_factor[str(factor)] = {
                "n": len(cell),
                "mean_margin": float(margins.mean()),
                "fp_rate": float(np.mean(margins > 0)),
                "mean_commitment": float(np.mean([row["score"]["commitment"] for row in cell])),
            }
        per_case: dict[str, dict[int, float]] = defaultdict(dict)
        for row in selected:
            per_case[str(row["case_key"])][int(row["factor"])] = float(
                row["score"]["yes_minus_no"]
            )
        complete = [values for values in per_case.values() if set(values) == set(FACTORS)]
        delta_margin = np.asarray([values[8] - values[1] for values in complete])
        delta_fp = np.asarray(
            [float(values[8] > 0) - float(values[1] > 0) for values in complete]
        )
        margin_ci = bootstrap(delta_margin, draws, seed + (0 if mode == "tied" else 1))
        fp_ci = bootstrap(delta_fp, draws, seed + (10 if mode == "tied" else 11))
        monotonic = monotonic_fraction(selected, mode)
        gate = bool(
            margin_ci["estimate"] >= 0.10
            and margin_ci["ci_low"] > 0
            and fp_ci["estimate"] >= 0.10
            and fp_ci["ci_low"] > 0
            and monotonic >= 0.60
        )
        by_mode[mode] = {
            "by_factor": by_factor,
            "factor8_minus_factor1_margin": margin_ci,
            "factor8_minus_factor1_fp_rate": fp_ci,
            "monotonic_positive_margin_fraction": monotonic,
            "fatal_gate_passed": gate,
        }
    passed = any(value["fatal_gate_passed"] for value in by_mode.values())
    return {
        "version": VERSION,
        "status": "GO_MULTIPLICITY_MECHANISM" if passed else "NO_GO_MULTIPLICITY_MECHANISM",
        "n_cases": len({row["case_key"] for row in ok}),
        "failed_rows": len(rows) - len(ok),
        "gate": {
            "population": "reader-unanimous 0/3 VinDr claims",
            "requirements": {
                "factor8_minus_factor1_mean_margin": ">=0.10 and image-bootstrap CI low >0",
                "factor8_minus_factor1_fp_rate": ">=0.10 and image-bootstrap CI low >0",
                "monotonic_case_fraction": ">=0.60",
                "position_modes": "at least one of tied or sequential must pass all requirements",
            },
            "scientific_boundary": (
                "failure closes repeated-evidence counting as the explanation of the observed crop FP inflation; "
                "it does not claim that all local-global refinement methods are impossible"
            ),
        },
        "position_modes": by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path(
            "corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1/selections.jsonl"
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(
            "corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1/images"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.limit < 2:
        raise ValueError("limit must be at least two")
    cases = load_cases(args.selections, args.image_root, 0, args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    raw_path = args.output_dir / "raw.jsonl"
    config = {
        "version": VERSION,
        "created_at": now(),
        "model": "HuatuoGPT-Vision-7B",
        "model_dir": str(args.model_dir.resolve()),
        "factors": list(FACTORS),
        "position_modes": list(POSITION_MODES),
        "selection": "stable hash from frozen observation-policy panel; label=0; full render; neutral prompt",
        "limit": args.limit,
        "seed": args.seed,
        "cases": cases,
        "intervention": "repeat the exact projected visual-token block; no pixel or token-content change",
        "claim_boundary": "mechanism fatal probe only; not a mitigation result",
    }
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        existing = json.loads(config_path.read_text())
        for key in ("version", "model", "factors", "position_modes", "selection", "limit", "seed", "cases"):
            if existing[key] != config[key]:
                raise RuntimeError(f"resume config drift: {key}")
    else:
        atomic_json(config_path, config)

    completed = set()
    if raw_path.exists() and args.resume:
        for line in raw_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["case_key"], row["position_mode"], int(row["factor"])))

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    ids = label_ids(bot)
    total = len(cases) * len(POSITION_MODES) * len(FACTORS)
    for case_index, case in enumerate(cases):
        case_key = f"{case['sample_id']}|{case['finding']}"
        image = Image.open(case["image"]).convert("RGB")
        tensor = torch.stack(bot.get_image_tensors([image])).to(
            device=bot.model.device, dtype=torch.bfloat16
        )
        embeddings, attention, positions, span = prepared_embeddings(
            bot, prompt_for(case["finding"]), tensor
        )
        for mode in POSITION_MODES:
            for factor in FACTORS:
                key = (case_key, mode, factor)
                if key in completed:
                    continue
                row: dict[str, Any] = {
                    "version": VERSION,
                    "case_key": case_key,
                    "sample_id": case["sample_id"],
                    "image_id": case["image_id"],
                    "finding": case["finding"],
                    "label": case["label"],
                    "position_mode": mode,
                    "factor": factor,
                    "visual_tokens": span[1] - span[0],
                    "status": "error",
                }
                try:
                    repeated, repeated_attention, repeated_positions = repeat_visual_block(
                        embeddings, attention, positions, span, factor, mode
                    )
                    row.update(
                        {
                            "status": "ok",
                            "sequence_tokens": int(repeated.shape[1]),
                            "score": score_condition(
                                bot, ids, repeated, repeated_attention, repeated_positions
                            ),
                            "completed_at": now(),
                        }
                    )
                except Exception as error:
                    row.update(
                        {
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                            "completed_at": now(),
                        }
                    )
                append_jsonl(raw_path, row)
                completed.add(key)
                print(
                    f"[{len(completed)}/{total}] case={case_index + 1}/{len(cases)} "
                    f"mode={mode} factor={factor} status={row['status']}",
                    flush=True,
                )
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    atomic_json(
        args.output_dir / "analysis.json",
        analyze(rows, args.bootstrap_draws, args.seed),
    )


if __name__ == "__main__":
    main()
