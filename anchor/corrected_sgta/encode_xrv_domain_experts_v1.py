#!/usr/bin/env python3
"""CPU encoder for multiple official source-specific XRV specialists."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from anchor.corrected_sgta.screen_external_visual_increment_v1 import load_claims, sha256_file
from anchor.corrected_sgta.screen_xrv_visual_increment_v1 import XRV_LABELS, chunks, dicom_tensor, load_xrv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--models-source", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("DOMAIN", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) not in ("", "-1"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' to preserve the baseline GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    image_ids = sorted(
        {
            row["image_id"]
            for manifest in args.manifests
            for row in load_claims(manifest, "unused", "unused")
        }
    )
    checkpoints = [(domain, Path(path)) for domain, path in args.checkpoint]
    models = [(domain, load_xrv(args.models_source, path)) for domain, path in checkpoints]
    encoded = {domain: [] for domain, _ in models}
    with torch.inference_mode():
        for batch_ids in chunks(image_ids, args.batch_size):
            images = torch.stack([dicom_tensor(args.image_root / f"{image_id}.dicom") for image_id in batch_ids])
            for domain, model in models:
                logits = model.classifier(model.features2(images)).cpu().numpy().astype(np.float32)
                encoded[domain].append(logits)
    domains = [domain for domain, _ in checkpoints]
    tensor = np.stack([np.concatenate(encoded[domain], axis=0) for domain in domains], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=np.asarray(image_ids),
        domains=np.asarray(domains),
        labels=np.asarray(XRV_LABELS),
        logits=tensor,
        provenance=np.asarray(
            json.dumps(
                {
                    "protocol": "xrv-domain-experts-encoding-v1",
                    "models_source_sha256": sha256_file(args.models_source),
                    "checkpoints": {domain: sha256_file(path) for domain, path in checkpoints},
                    "image_count": len(image_ids),
                    "renderer": "XRV [-1024,1024], MONOCHROME1 fix, center crop, 224 bilinear",
                },
                sort_keys=True,
            )
        ),
    )
    print(json.dumps({"status": "complete", "domains": domains, "images": len(image_ids), "shape": tensor.shape}))


if __name__ == "__main__":
    main()
