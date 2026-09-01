#!/usr/bin/env python3
"""Layerwise ontology scores on real within-patient CXR transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from corrected_sgta.run_huatuo_natural_counterfactual_probe_v1 import prompt, sha256
from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
)


VERSION = "huatuo-natural-counterfactual-layerwise-v1"
FINDINGS = ("pleural effusion", "pneumothorax", "edema", "consolidation", "pneumonia")


@torch.inference_mode()
def score_layers(bot, ids: dict[str, int], image_path: Path, finding: str) -> dict[str, float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt(finding), tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    values = layer_logits(bot, hidden, tuple(range(len(hidden))), ids)
    return {
        str(layer): float(logits["supported"] - logits["refuted"])
        for layer, logits in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row['patient_id']}:{row['prior_study']}:{row['current_study']}"
        by_pair[key].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "model": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n_pairs": len(by_pair),
        "ontology": list(FINDINGS),
        "measurement": "all hidden-state output-head Yes-minus-No margins, including final layer",
        "specificity_control": "mean of ontology findings absent from all silver changes for the same pair",
        "boundary": "silver mechanism screen; off-claims are unmentioned, not expert-verified stable",
        "code_sha256": sha256(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume")
        old = json.loads(config_path.read_text())
        for key in ("version", "model", "manifest_sha256", "n_pairs", "ontology", "measurement"):
            if old.get(key) != config.get(key):
                raise RuntimeError(f"resume configuration drift: {key}")
        config = old
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    if raw_path.exists() and args.resume:
        done = {
            row["pair_key"]
            for row in map(json.loads, raw_path.read_text().splitlines())
            if row.get("status") == "ok"
        }

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    for index, pair_key in enumerate(sorted(by_pair)):
        if pair_key in done:
            continue
        pair_rows = by_pair[pair_key]
        exemplar = pair_rows[0]
        output = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_key": pair_key,
            "patient_id": exemplar["patient_id"],
            "prior_study": exemplar["prior_study"],
            "current_study": exemplar["current_study"],
            "prior_image": exemplar["prior_image"],
            "current_image": exemplar["current_image"],
            "target_claims": [
                {
                    "record_key": row["record_key"],
                    "finding": row["finding"],
                    "direction": row["direction"],
                    "direction_name": row["direction_name"],
                }
                for row in pair_rows
            ],
            "control_findings": [
                finding
                for finding in FINDINGS
                if finding not in {row["finding"] for row in pair_rows}
            ],
            "status": "error",
        }
        try:
            output["scores"] = {
                phase: {
                    finding: score_layers(bot, ids, Path(exemplar[f"{phase}_image"]), finding)
                    for finding in FINDINGS
                }
                for phase in ("prior", "current")
            }
            output["status"] = "ok"
        except Exception as error:
            output["error"] = repr(error)
            output["traceback"] = traceback.format_exc()
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        with raw_path.open("a") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
        print(json.dumps({"progress": f"{index + 1}/{len(by_pair)}", "pair_key": pair_key, "status": output["status"]}), flush=True)


if __name__ == "__main__":
    main()
