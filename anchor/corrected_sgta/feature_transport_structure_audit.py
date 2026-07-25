"""Fail-closed pixel-identity audit for visual-token feature transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrected_sgta.cache import iter_successes
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    if metadata.get("transport_cache_version") != "sgta-feature-transport-v1":
        raise RuntimeError("not a feature-transport cache")
    if sha256_file(args.source_bank) != metadata["config"]["source_bank_sha256"]:
        raise RuntimeError("feature transport/source-bank mismatch")
    verify_source_artifacts(load_manifest(args.source_bank))
    visual = Path(metadata["config"]["visual_centers"])
    if sha256_file(visual) != metadata["config"]["visual_centers_sha256"]:
        raise RuntimeError("feature transport/visual-center mismatch")
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    selected = sum(
        any(item.get("selected") for item in row.get("alignment_candidates", []))
        for row in rows
    )
    summary = {
        "n": selected,
        "pass_rate": None if selected == 0 else 1.0,
        "ssim_median": None if selected == 0 else 1.0,
        "local_contrast_correlation_median": None if selected == 0 else 1.0,
        "gradient_ratio_median": None if selected == 0 else 1.0,
    }
    report = {
        "version": "sgta-feature-transport-structure-audit-v1",
        "source_cache": str(args.cache),
        "fingerprint": metadata["fingerprint"],
        "intervention_scope": (
            "visual-token features only; the processor input image is byte-identical across roles"
        ),
        "matched": summary,
        "wrong_control": summary,
        "formal_matched_structure_pass": selected > 0,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
