#!/usr/bin/env python3
"""Build a deterministic report-generation manifest without test-time retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256


VERSION = "mmedrag-generation-manifest-v1"


def stable_key(seed: int, dataset: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for dataset in ("iuxray", "mimic", "harvard"):
        parser.add_argument(f"--{dataset}-json", required=True, type=Path)
        parser.add_argument(f"--{dataset}-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-dataset", type=int, default=128)
    for dataset in ("iuxray", "mimic", "harvard"):
        parser.add_argument(f"--{dataset}-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    specs = {
        name: (
            getattr(args, f"{name}_json"),
            getattr(args, f"{name}_root"),
        )
        for name in ("iuxray", "mimic", "harvard")
    }
    records = []
    availability = {}
    for dataset, (path, root) in specs.items():
        target = getattr(args, f"{dataset}_samples") or args.samples_per_dataset
        rows = sorted(
            json.loads(path.read_text()),
            key=lambda row: stable_key(args.seed, dataset, str(row["id"])),
        )
        chosen = 0
        missing = 0
        for row in rows:
            relative = (
                row["image_path"][0]
                if isinstance(row["image_path"], list)
                else row["image_path"]
            )
            image = root / str(relative)
            if not image.is_file():
                missing += 1
                continue
            records.append(
                {
                    "dataset": dataset,
                    "id": str(row["id"]),
                    "image": str(image.resolve()),
                    "reference": str(row["report"]),
                }
            )
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise RuntimeError(
                f"{dataset}: requested {target} available "
                f"images, found {chosen}"
            )
        availability[dataset] = {
            "selected": chosen,
            "missing_before_last_selected": missing,
            "test_rows": len(rows),
        }

    provenance = {
        "version": VERSION,
        "test_json_sha256": {
            name: file_sha256(path) for name, (path, _) in specs.items()
        },
        "samples_per_dataset": args.samples_per_dataset,
        "per_dataset_samples": {
            dataset: getattr(args, f"{dataset}_samples") or args.samples_per_dataset
            for dataset in specs
        },
        "seed": args.seed,
        "selection": (
            "sha256(seed,dataset,id), first samples_per_dataset rows whose "
            "resolved local image exists"
        ),
        "availability": availability,
        "code_sha256": file_sha256(Path(__file__)),
    }
    payload = {
        **provenance,
        "fingerprint": stable_json_sha256(provenance),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fingerprint": payload["fingerprint"],
                "n": len(records),
                "availability": availability,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
