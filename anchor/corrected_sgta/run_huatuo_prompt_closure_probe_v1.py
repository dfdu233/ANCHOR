#!/usr/bin/env python3
"""Fatal canary for closing the observed multimodal prompt before decoding.

Three conditions use bit-identical pixels, tokens, positions and verbalizers:

* causal: the checkpoint's native triangular mask;
* visual_read: only visual-token rows may read the later question;
* full_prefix: all already-observed prompt tokens interact bidirectionally.

The first is the baseline, the second is the minimal candidate primitive, and
the third is a PrefixLM-style upper control.  This is a mechanism canary, not a
method result.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.run_huatuo_query_first_topology_probe_v1 import (
    append_jsonl,
    atomic_json,
    load_cases,
    score,
    score_topology,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    dicom_to_pil,
    import_huatuo,
    label_ids,
    prepared_embeddings,
    prompt_for,
)


VERSION = "huatuo-prompt-closure-probe-v1"
MODES = ("causal", "visual_read", "full_prefix")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell_metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    cell = [row for row in rows if row["status"] == "ok" and row["mode"] == mode]
    if not cell:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    truth = np.asarray([int(row["label"]) for row in cell])
    pred = np.asarray([int(row["score"]["prediction"]) for row in cell])
    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": len(cell),
        "accuracy": (tp + tn) / len(cell),
        "balanced_accuracy": (tpr + tnr) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_margin": float(np.mean([row["score"]["yes_minus_no"] for row in cell])),
    }


def bootstrap(rows: list[dict[str, Any]], mode: str, draws: int, seed: int) -> dict[str, float]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            grouped.setdefault(row["record_key"], {})[row["mode"]] = row
    pairs = [cell for cell in grouped.values() if "causal" in cell and mode in cell]
    y = np.asarray([int(cell["causal"]["label"]) for cell in pairs])
    native = np.asarray([int(cell["causal"]["score"]["prediction"]) for cell in pairs])
    changed = np.asarray([int(cell[mode]["score"]["prediction"]) for cell in pairs])
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]

    def bacc(index: np.ndarray, pred: np.ndarray) -> float:
        yy = y[index]; pp = pred[index]
        return float((np.mean(pp[yy == 1] == 1) + np.mean(pp[yy == 0] == 0)) / 2)

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        index = np.concatenate((rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)))
        samples.append(bacc(index, changed) - bacc(index, native))
    index = np.arange(len(y))
    return {
        "estimate": bacc(index, changed) - bacc(index, native),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def analyze(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    cells = {mode: cell_metrics(rows, mode) for mode in MODES}
    deltas = {
        mode: bootstrap(rows, mode, draws, seed + index)
        for index, mode in enumerate(MODES[1:])
    }
    candidate = cells["visual_read"]
    native = cells["causal"]
    gate = bool(
        deltas["visual_read"]["estimate"] >= 0.02
        and deltas["visual_read"]["ci_low"] > 0
        and candidate["fp"] <= native["fp"]
        and candidate["fn"] <= native["fn"]
    )
    return {
        "version": VERSION,
        "status": "GO_VISUAL_READ_TOPOLOGY" if gate else "NO_GO_VISUAL_READ_TOPOLOGY",
        "modes": cells,
        "bacc_delta_vs_causal": deltas,
        "gate_passed": gate,
        "gate": "visual-read BAcc +2pp with CI low>0; neither FP nor FN may increase",
        "interpretation": (
            "full-prefix is an upper control only; it cannot rescue a failed visual-read candidate"
        ),
        "boundary": (
            "a pass authorizes a fresh larger panel and Hulu replication, not an ICLR claim; "
            "a fail closes prompt-closure mask editing without interpolation or threshold tuning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("corrected_runs/c3_guard/vindr_claim_common_mode_canary_v1/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-bin", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.manifest, args.per_bin, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "created_at": now(),
        "model": "HuatuoGPT-Vision-7B",
        "seed": args.seed,
        "per_bin": args.per_bin,
        "modes": list(MODES),
        "cases": cases,
        "invariants": "pixels, input embeddings, token order, position IDs, sequence length and verbalizers are identical",
        "minimal_edit": "visual-token attention rows can read all already-observed prompt tokens",
        "research_role": "fatal topology canary",
    }
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        previous = json.loads(config_path.read_text())
        for key in ("version", "model", "seed", "per_bin", "modes", "cases", "invariants", "minimal_edit"):
            if previous[key] != config[key]:
                raise RuntimeError(f"resume config drift: {key}")
    else:
        atomic_json(config_path, config)

    raw_path = args.output_dir / "raw.jsonl"
    completed: set[tuple[str, str]] = set()
    if raw_path.exists() and args.resume:
        completed = {
            (row["record_key"], row["mode"])
            for row in (json.loads(line) for line in raw_path.read_text().splitlines() if line.strip())
        }

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    bot.debug = False
    ids = label_ids(bot)
    total = len(cases) * len(MODES)
    for case in cases:
        image = dicom_to_pil(Path(case["image_path"]))
        tensor = torch.stack(bot.get_image_tensors([image])).to(device=bot.model.device, dtype=torch.bfloat16)
        embeddings, attention, positions, span = prepared_embeddings(bot, prompt_for(case["finding"]), tensor)
        for mode in MODES:
            key = (case["record_key"], mode)
            if key in completed:
                continue
            row: dict[str, Any] = {
                "version": VERSION,
                "record_key": case["record_key"],
                "image_id": case["image_id"],
                "image_path": case["image_path"],
                "finding": case["finding"],
                "positive_votes": case["positive_votes"],
                "label": int(case["positive_votes"] == 3),
                "mode": mode,
                "visual_span": list(span),
                "sequence_tokens": int(embeddings.shape[1]),
                "status": "error",
            }
            try:
                value = (
                    score(bot, ids, embeddings, attention, positions)
                    if mode == "causal"
                    else score_topology(bot, ids, embeddings, positions, span, mode)
                )
                row.update({"status": "ok", "score": value, "completed_at": now()})
            except Exception as error:
                row.update({"error": repr(error), "traceback": traceback.format_exc(), "completed_at": now()})
            append_jsonl(raw_path, row)
            completed.add(key)
            print(f"[{len(completed)}/{total}] {case['record_key']} {mode} {row['status']} pred={row.get('score', {}).get('prediction')}", flush=True)

    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    atomic_json(args.output_dir / "analysis.json", analyze(rows, args.bootstrap_draws, args.seed))


if __name__ == "__main__":
    main()
