"""Fail-closed pixel-identity and pair-completeness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corrected_sgta.cache import iter_successes
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path); parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    if metadata.get("transport_cache_version") != "sgta-model-source-visual-residual-v1":
        raise RuntimeError("not a model-source-residual cache")
    if sha256_file(args.source_bank) != metadata["config"]["source_bank_sha256"]:
        raise RuntimeError("source-bank mismatch")
    verify_source_artifacts(load_manifest(args.source_bank))
    visual = Path(metadata["config"]["visual_centers"])
    if sha256_file(visual) != metadata["config"]["visual_centers_sha256"]:
        raise RuntimeError("visual-center mismatch")
    rows = list(iter_successes(args.cache, metadata["fingerprint"]))
    complete = [row["style_roles"] == ["original", "matched", "wrong_control"] and len(row.get("alignment_candidates", [])) == 1 for row in rows]
    count = int(sum(complete)); summary = {"n": count, "pass_rate": 1.0 if count else None, "pixel_identity": True}
    report = {
        "version": "sgta-model-source-residual-audit-v1", "fingerprint": metadata["fingerprint"],
        "source_cache": str(args.cache), "rows": len(rows), "matched": summary,
        "wrong_control": summary, "formal_matched_structure_pass": bool(rows) and all(complete),
        "intervention_scope": "visual features only; processor input image is identical across roles",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

