#!/usr/bin/env python3
"""Audit prior repository references to a locked VinDr holdout.

This is deliberately a provenance audit, not a model evaluation.  It prevents
calling an endpoint-held-out split "image-unseen" when its image identifiers
already occur in unrelated historical experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable


PROTOCOL = "vindr-holdout-prior-exposure-audit-v1"
# VinDr image IDs are 32 lower-case hexadecimal characters (not UUIDs).
IMAGE_ID = re.compile(rb"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["image_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def referenced_ids(root: Path, candidates: set[str]) -> set[str]:
    found: set[str] = set()
    if not root.exists():
        return found
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        # Archives and binary tensors are neither searchable provenance nor
        # necessary here; all three audited experiment families retain text receipts.
        if path.suffix.lower() in {".tar", ".gz", ".zip", ".npz", ".npy", ".pt", ".pth", ".safetensors"}:
            continue
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    for value in IMAGE_ID.findall(block):
                        image_id = value.decode("ascii")
                        if image_id in candidates:
                            found.add(image_id)
        except OSError:
            continue
    return found


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--direct-ce-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    holdout = jsonl_ids(args.holdout_manifest)
    excluded = jsonl_ids(args.direct_ce_exclusion_manifest)
    direct_overlap = holdout & excluded
    by_root = {
        str(root.resolve()): sorted(referenced_ids(root, holdout))
        for root in args.prior_root
    }
    any_prior = set().union(*(set(values) for values in by_root.values()))
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "holdout_manifest": str(args.holdout_manifest.resolve()),
        "holdout_manifest_sha256": sha256(args.holdout_manifest),
        "holdout_unique_images": len(holdout),
        "direct_ce_exclusion_manifest": str(args.direct_ce_exclusion_manifest.resolve()),
        "direct_ce_exclusion_manifest_sha256": sha256(args.direct_ce_exclusion_manifest),
        "direct_ce_overlap_count": len(direct_overlap),
        "direct_ce_exclusion_verified": len(direct_overlap) == 0,
        "audited_prior_roots": {
            root: {"overlap_count": len(values), "overlap_ids_sha256": hashlib.sha256("\n".join(values).encode()).hexdigest()}
            for root, values in by_root.items()
        },
        "any_audited_prior_reference_count": len(any_prior),
        "any_audited_prior_reference_fraction": len(any_prior) / len(holdout),
        "image_unseen_across_repository": False,
        "endpoint_prospective": True,
        "claim_boundary": "prospective endpoint-held-out relative to the prior direct-CE manifest; not image-unseen",
        "interpretation": (
            "Selection and confirmation for the new incremental-decoding endpoint were frozen before its outcomes were read. "
            "Image identifiers may have appeared in unrelated historical experiments, so external independent confirmation is required for a definitive claim."
        ),
        "audit_code_sha256": sha256(Path(__file__)),
    }
    if not payload["direct_ce_exclusion_verified"]:
        raise ValueError("locked holdout overlaps the prior direct-CE manifest")
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
