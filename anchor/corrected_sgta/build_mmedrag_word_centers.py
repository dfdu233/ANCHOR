#!/usr/bin/env python3
"""Build content-free robust word-count centers from source report banks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from corrected_sgta.rule_source_preference import file_sha256, stable_json_sha256


VERSION = "mmedrag-source-word-centers-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiology-json", required=True, type=Path)
    parser.add_argument("--iu-root", required=True, type=Path)
    parser.add_argument("--harvard-json", required=True, type=Path)
    parser.add_argument("--harvard-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    radiology = [
        row
        for row in json.loads(args.radiology_json.read_text())
        if str(row.get("id", "")).startswith("CXR")
        and (args.iu_root / str(row["image_path"][0])).is_file()
    ]
    harvard = [
        row
        for row in json.loads(args.harvard_json.read_text())
        if (args.harvard_root / str(row["image_path"])).is_file()
    ]
    if not radiology or not harvard:
        raise ValueError("a source report domain is empty")
    centers = {
        "radiology": {
            "source_reports": len(radiology),
            "median_words": int(
                statistics.median(len(str(row["report"]).split()) for row in radiology)
            ),
        },
        "ophthalmology": {
            "source_reports": len(harvard),
            "median_words": int(
                statistics.median(len(str(row["report"]).split()) for row in harvard)
            ),
        },
    }
    provenance = {
        "version": VERSION,
        "radiology_json_sha256": file_sha256(args.radiology_json),
        "harvard_json_sha256": file_sha256(args.harvard_json),
        "centers": centers,
        "content_retained": False,
        "estimator": "coordinatewise sample median; empirical L1-risk minimizer",
        "code_sha256": file_sha256(Path(__file__)),
    }
    payload = {
        **provenance,
        "fingerprint": stable_json_sha256(provenance),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
