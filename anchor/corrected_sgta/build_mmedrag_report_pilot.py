#!/usr/bin/env python3
"""Build a deterministic MMed-RAG report pilot with source-report neighbors."""
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

VERSION = "mmedrag-report-pilot-v1"


def stable_key(seed: int, dataset: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--iuxray-json", type=Path, required=True)
    parser.add_argument("--iuxray-root", type=Path, required=True)
    parser.add_argument("--mimic-json", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--harvard-json", type=Path, required=True)
    parser.add_argument("--harvard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    bank = torch.load(args.source_bank, map_location="cpu", weights_only=False)
    bank_features = F.normalize(bank["features"].float(), dim=-1)
    bank_rows = bank["rows_data"]
    specs = {
        "iuxray": (args.iuxray_json, args.iuxray_root, "radiology_iuxray"),
        "mimic": (args.mimic_json, args.mimic_root, "radiology_iuxray"),
        "harvard": (
            args.harvard_json,
            args.harvard_root,
            "ophthalmology_harvard",
        ),
    }
    selected = []
    availability = {}
    for dataset, (path, root, source_domain) in specs.items():
        rows = json.loads(path.read_text())
        rows = sorted(
            rows,
            key=lambda row: stable_key(args.seed, dataset, str(row["id"])),
        )
        chosen = 0
        missing = 0
        for row in rows:
            rel = row["image_path"][0] if isinstance(row["image_path"], list) else row["image_path"]
            image = root / str(rel)
            if not image.is_file():
                missing += 1
                continue
            selected.append(
                {
                    "dataset": dataset,
                    "id": str(row["id"]),
                    "image": str(image.resolve()),
                    "reference": str(row["report"]),
                    "source_domain": source_domain,
                }
            )
            chosen += 1
            if chosen == args.samples_per_dataset:
                break
        if chosen != args.samples_per_dataset:
            raise RuntimeError(
                f"{dataset}: requested {args.samples_per_dataset} available images, "
                f"found {chosen} after skipping {missing} missing files"
            )
        availability[dataset] = {
            "selected": chosen,
            "missing_before_last_selected": missing,
            "test_rows": len(rows),
        }

    device = torch.device("cuda")
    model, preprocess, _ = _load_biomedclip()
    model = model.to(device).eval()
    try:
        with torch.inference_mode():
            for row in tqdm(selected, desc="retrieve-source-report"):
                with Image.open(row["image"]) as handle:
                    image = preprocess(handle.convert("RGB")).unsqueeze(0).to(device)
                query = F.normalize(model.encode_image(image), dim=-1).float().cpu()[0]
                indices = [
                    i
                    for i, source in enumerate(bank_rows)
                    if source["domain"] == row["source_domain"]
                    and source["id"] != row["id"]
                ]
                similarities = bank_features[indices] @ query
                top = similarities.topk(min(args.neighbors, len(indices))).indices.tolist()
                row["neighbors"] = [
                    {
                        **bank_rows[indices[offset]],
                        "similarity": float(similarities[offset]),
                    }
                    for offset in top
                ]
    finally:
        del model
        torch.cuda.empty_cache()

    provenance = {
        "version": VERSION,
        "source_bank_sha256": file_sha256(args.source_bank),
        "test_json_sha256": {
            name: file_sha256(spec[0]) for name, spec in specs.items()
        },
        "samples_per_dataset": args.samples_per_dataset,
        "neighbors": args.neighbors,
        "seed": args.seed,
        "selection": (
            "sha256(seed,dataset,id), first samples_per_dataset rows whose "
            "resolved local image exists"
        ),
        "availability": availability,
        "source_policy": {
            "iuxray": "IU-Xray training reports",
            "mimic": "IU-Xray training reports (cross-domain)",
            "harvard": "Harvard-FairVLMed training reports",
        },
    }
    payload = {
        **provenance,
        "fingerprint": stable_json_sha256(provenance),
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fingerprint": payload["fingerprint"],
                "n": len(selected),
                "datasets": {
                    name: sum(row["dataset"] == name for row in selected)
                    for name in specs
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
