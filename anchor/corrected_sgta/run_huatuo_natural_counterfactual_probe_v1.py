#!/usr/bin/env python3
"""Score real prior/current CXRs for report-derived atomic claim transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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


VERSION = "huatuo-natural-counterfactual-probe-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt(finding: str) -> str:
    return (
        f"Does this chest X-ray show {finding}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    )


@torch.inference_mode()
def score(bot, ids: dict[str, int], image_path: Path, finding: str) -> dict[str, object]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = torch.stack(bot.get_image_tensors([image])).to(
        device=bot.model.device, dtype=torch.bfloat16
    )
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt(finding), tensor)
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    values = layer_logits(bot, hidden, (), ids)[len(hidden) - 1]
    logits = np.asarray(
        [values["supported"], values["refuted"], values["undetermined"]], dtype=float
    )
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return {
        "yes": values["supported"],
        "no": values["refuted"],
        "maybe": values["undetermined"],
        "yes_minus_no": values["supported"] - values["refuted"],
        "probabilities": {
            "yes": float(probabilities[0]),
            "no": float(probabilities[1]),
            "maybe": float(probabilities[2]),
        },
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n": len(rows),
        "prompt": "Does this chest X-ray show <finding>? Yes/No/Maybe",
        "measurement": "final-layer one-token Yes-minus-No margin on each image separately",
        "label_boundary": "report-derived silver natural transitions; not clinical gold",
        "command": " ".join(sys.argv),
        "code_sha256": sha256(Path(__file__)),
    }
    config["fingerprint"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume")
        old = json.loads(config_path.read_text())
        for key in ("version", "model", "manifest_sha256", "n", "prompt", "measurement"):
            if old.get(key) != config.get(key):
                raise RuntimeError(f"resume configuration drift: {key}")
        config = old
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    if raw_path.exists() and args.resume:
        done = {
            json.loads(line)["record_key"]
            for line in raw_path.read_text().splitlines()
            if line.strip()
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
            output["scores"] = {
                "prior": score(bot, ids, Path(row["prior_image"]), row["finding"]),
                "current": score(bot, ids, Path(row["current_image"]), row["finding"]),
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
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(rows)}",
                    "record_key": row["record_key"],
                    "status": output["status"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
