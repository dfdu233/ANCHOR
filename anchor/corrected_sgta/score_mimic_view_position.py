"""Score frontal/lateral MIMIC view position with the local BiomedCLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.corrected_sgta.filter_anchor_dg_chest_sources import (
    DEFAULT_MODEL_ROOT,
)
from anchor.corrected_sgta.prepare_center_native_pubmed import score_frontal
from anchor.corrected_sgta.run_visual_evidence_chord_probe import unique_cases


VERSION = "mimic-biomedclip-view-position-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cases = unique_cases(args.questions, args.image_manifest, args.limit)
    rows = [
        {
            "id": case["case_id"],
            "image_bytes": Path(case["image"]).read_bytes(),
        }
        for case in cases
    ]
    scores = score_frontal(
        rows, args.model_root, args.batch_size, device="cuda"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for case in cases:
            result = scores[case["case_id"]]
            handle.write(
                json.dumps(
                    {
                        "version": VERSION,
                        "case_id": case["case_id"],
                        "image_relative": case["image_relative"],
                        **result,
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "n": len(cases),
                "frontal_top1": sum(
                    score["predicted_category"]
                    == "a frontal chest radiograph"
                    for score in scores.values()
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
