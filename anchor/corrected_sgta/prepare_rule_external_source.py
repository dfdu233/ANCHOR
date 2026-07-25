"""Prepare leakage-audited external Yes/No data for RULE adapter pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "rule-external-source-prep-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def answer(row: dict) -> str | None:
    value = str(row.get("answer", row.get("gt_ans", ""))).strip().lower().rstrip(".")
    return value.capitalize() + "." if value in {"yes", "no"} else None


def image_name(row: dict) -> str:
    return str(row.get("img_name") or row.get("image") or row.get("img_id") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    test_rows = [json.loads(line) for line in args.locked_test.read_text().splitlines() if line.strip()]
    test_images = {str(row.get("image", "")) for row in test_rows}
    converted = []
    missing = 0
    invalid_answer = 0
    overlaps = []
    for index, row in enumerate(rows):
        target = answer(row)
        if target is None:
            invalid_answer += 1
            continue
        image = image_name(row)
        if not image or not (args.image_root / image).is_file():
            missing += 1
            continue
        if image in test_images:
            overlaps.append(image)
            continue
        question = str(row.get("question", "")).replace("<image>", "").strip()
        if not question:
            continue
        converted.append({
            "id": f"external-{index}",
            "image": image,
            "source_dataset": str(args.input),
            "conversations": [
                {"from": "human", "value": question + "\n<image>"},
                {"from": "gpt", "value": target},
            ],
        })
    if overlaps:
        raise RuntimeError(f"external source overlaps locked test images: {overlaps[:3]}")
    if not converted:
        raise RuntimeError("no valid external Yes/No examples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, indent=2))
    manifest = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "locked_test": str(args.locked_test.resolve()),
        "locked_test_sha256": sha256(args.locked_test),
        "image_root": str(args.image_root.resolve()),
        "n_input": len(rows),
        "n_output": len(converted),
        "missing_images": missing,
        "invalid_answers": invalid_answer,
        "train_test_image_overlap": 0,
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
