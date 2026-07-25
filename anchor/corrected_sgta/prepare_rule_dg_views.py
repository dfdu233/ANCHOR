"""Generate deterministic RULE IU-Xray views toward its test-disjoint train center."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from corrected_sgta.frequency_alignment_source_spectrum_release2 import source_spectrum_alignment_release2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--rule-test", type=Path, default=Path("/root/autodl-tmp/RULE/data/test/iuxray_test.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("/root/autodl-tmp/MedHEval/images/IU-Xray"))
    parser.add_argument("--center", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.02, 0.05, 0.1))
    args = parser.parse_args()
    pilot_ids = {str(row["qid"]) for row in json.loads(args.pilot.read_text())}
    rows = [json.loads(line) for line in args.rule_test.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if str(row["question_id"]) in pilot_ids]
    rows.sort(key=lambda row: int(row["question_id"]))
    if len(rows) != len(pilot_ids):
        raise RuntimeError(f"pilot qid mismatch: rows={len(rows)} qids={len(pilot_ids)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    qfile = args.output_root / "questions.pilot.jsonl"
    qfile.write_text("".join(json.dumps(row) + "\n" for row in rows))
    center = np.load(args.center)
    images = sorted({row["image"] for row in rows})
    manifest = {"questions": len(rows), "unique_images": len(images), "alphas": args.alphas, "views": {}}
    for alpha in args.alphas:
        view_root = args.output_root / f"alpha_{alpha:g}"
        for relative in tqdm(images, desc=f"DG alpha={alpha:g}"):
            destination = view_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(args.image_root / relative) as source:
                view = source_spectrum_alignment_release2(source, center, low_frequency_ratio=alpha, source_ratio=0.0)
            view.save(destination)
        manifest["views"][f"{alpha:g}"] = str(view_root.resolve())
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
