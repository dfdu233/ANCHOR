#!/usr/bin/env python3
"""Off-claim control for the real prior/current Huatuo score probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from corrected_sgta.run_huatuo_natural_counterfactual_probe_v1 import score, sha256
from corrected_sgta.run_huatuo_vindr_commitment_probe import import_huatuo, label_ids


VERSION = "huatuo-natural-counterfactual-specificity-v1"
FINDINGS = ("pleural effusion", "pneumothorax", "edema", "consolidation", "pneumonia")


def stable_control(pair_key: str, occupied: set[str], seed: int) -> str:
    candidates = [finding for finding in FINDINGS if finding not in occupied]
    if not candidates:
        raise ValueError(f"no off-claim control for {pair_key}")
    return min(
        candidates,
        key=lambda finding: hashlib.sha256(f"{seed}:{pair_key}:{finding}".encode()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair_key = f"{row['patient_id']}:{row['prior_study']}:{row['current_study']}"
        by_pair[pair_key].append(row)
    controls = {
        pair: stable_control(pair, {row["finding"] for row in pair_rows}, args.seed)
        for pair, pair_rows in by_pair.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "model": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "seed": args.seed,
        "n_target_rows": len(rows),
        "n_unique_pair_controls": len(controls),
        "control_selection": "stable hash among ontology findings absent from every changed claim for that pair",
        "boundary": "an unmentioned claim is a negative control, not a verified stable clinical finding",
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume")
        old = json.loads(config_path.read_text())
        if old != config:
            raise RuntimeError("resume configuration drift")
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
        exemplar = by_pair[pair_key][0]
        finding = controls[pair_key]
        output = {
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "pair_key": pair_key,
            "patient_id": exemplar["patient_id"],
            "prior_study": exemplar["prior_study"],
            "current_study": exemplar["current_study"],
            "prior_image": exemplar["prior_image"],
            "current_image": exemplar["current_image"],
            "control_finding": finding,
            "target_record_keys": [row["record_key"] for row in by_pair[pair_key]],
            "target_findings": sorted({row["finding"] for row in by_pair[pair_key]}),
            "status": "error",
        }
        try:
            output["scores"] = {
                "prior": score(bot, ids, Path(exemplar["prior_image"]), finding),
                "current": score(bot, ids, Path(exemplar["current_image"]), finding),
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
