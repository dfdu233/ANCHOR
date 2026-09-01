#!/usr/bin/env python3
"""Score an eight-claim ontology on each VinDr common-mode canary image."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    dicom_to_pil,
    hidden_trajectory,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    prompt_for,
)


VERSION = "huatuo-vindr-claim-common-mode-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def score(bot, ids: dict[str, int], image, finding: str) -> dict[str, float]:
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(
        bot, prompt_for(finding), tensor
    )
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    values = layer_logits(bot, hidden, (), ids)[len(hidden) - 1]
    return {
        "yes": float(values["supported"]),
        "no": float(values["refuted"]),
        "maybe": float(values["undetermined"]),
        "yes_minus_no": float(values["supported"] - values["refuted"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    gate = json.loads(args.gate.read_text())
    ontology = list(gate["ontology"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "model": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "gate": str(args.gate.resolve()),
        "gate_sha256": sha256(args.gate),
        "ontology": ontology,
        "n": len(rows),
        "prompt": "Does this chest X-ray show <finding>? Answer exactly Yes, No, or Maybe.",
        "measurement": "native final hidden state and output head; Yes-minus-No",
        "code_sha256": sha256(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume")
        old = json.loads(config_path.read_text())
        for key in ("version", "model", "manifest_sha256", "gate_sha256", "ontology", "n", "prompt", "measurement"):
            if old.get(key) != config.get(key):
                raise RuntimeError(f"resume configuration drift: {key}")
        config = old
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    if raw_path.exists() and args.resume:
        done = {
            row["record_key"]
            for row in map(json.loads, raw_path.read_text().splitlines())
            if row.get("status") == "ok"
        }

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device=args.device)
    ids = label_ids(bot)
    for index, row in enumerate(rows):
        if row["record_key"] in done:
            continue
        output = {
            **row,
            "version": VERSION,
            "fingerprint": config["fingerprint"],
            "status": "error",
        }
        try:
            image = dicom_to_pil(Path(row["image_path"]))
            output["ontology_scores"] = {
                finding: score(bot, ids, image, finding) for finding in ontology
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
        print(json.dumps({"progress": f"{index + 1}/{len(rows)}", "record_key": row["record_key"], "status": output["status"]}), flush=True)


if __name__ == "__main__":
    main()
