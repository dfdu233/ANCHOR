#!/usr/bin/env python3
"""Cache normalized BiomedCLIP image features for MMed-RAG source reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256
from corrected_sgta.run_rule_biomedclip_bilinear_anchor import _load_biomedclip

VERSION = "mmedrag-source-report-bank-v1"


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiology-json", type=Path, required=True)
    parser.add_argument("--iu-root", type=Path, required=True)
    parser.add_argument("--harvard-json", type=Path, required=True)
    parser.add_argument("--harvard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-harvard", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    radiology = json.loads(args.radiology_json.read_text())
    harvard = json.loads(args.harvard_json.read_text())
    rows = []
    for row in radiology:
        if not str(row.get("id", "")).startswith("CXR"):
            continue
        path = args.iu_root / str(row["image_path"][0])
        if path.is_file():
            rows.append(
                {
                    "domain": "radiology_iuxray",
                    "id": str(row["id"]),
                    "image": str(path.resolve()),
                    "report": str(row["report"]),
                }
            )
    selected_harvard = sorted(
        harvard,
        key=lambda row: stable_key(args.seed, str(row["id"])),
    )[: args.max_harvard]
    for row in selected_harvard:
        path = args.harvard_root / str(row["image_path"])
        if path.is_file():
            rows.append(
                {
                    "domain": "ophthalmology_harvard",
                    "id": str(row["id"]),
                    "image": str(path.resolve()),
                    "report": str(row["report"]),
                }
            )
    if not rows:
        raise RuntimeError("source report bank is empty")

    device = torch.device("cuda")
    model, preprocess, _ = _load_biomedclip()
    model = model.to(device).eval()
    features = []
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(rows), args.batch_size), desc="source-bank"):
                batch = rows[start : start + args.batch_size]
                images = []
                for row in batch:
                    with Image.open(row["image"]) as handle:
                        images.append(preprocess(handle.convert("RGB")))
                encoded = F.normalize(
                    model.encode_image(torch.stack(images).to(device)), dim=-1
                )
                features.append(encoded.half().cpu())
    finally:
        del model
        torch.cuda.empty_cache()

    provenance = {
        "version": VERSION,
        "radiology_json_sha256": file_sha256(args.radiology_json),
        "harvard_json_sha256": file_sha256(args.harvard_json),
        "weights_sha256": file_sha256(
            Path("/root/autodl-tmp/BiomedCLIP/open_clip_pytorch_model.bin")
        ),
        "rows": len(rows),
        "domains": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in sorted({row["domain"] for row in rows})
        },
        "seed": args.seed,
    }
    payload = {
        **provenance,
        "fingerprint": stable_json_sha256(provenance),
        "rows_data": rows,
        "features": torch.cat(features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({k: payload[k] for k in ("fingerprint", "rows", "domains")}, indent=2))


if __name__ == "__main__":
    main()
