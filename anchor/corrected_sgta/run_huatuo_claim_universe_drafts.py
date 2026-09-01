#!/usr/bin/env python3
"""Generate Huatuo greedy drafts for a frozen grouped claim universe."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from corrected_sgta.evaluate_medheval_answers import parse_answer
from corrected_sgta.run_huatuo_rule_feddg import generate_with_nll
from corrected_sgta.run_huatuo_vindr_commitment_probe import import_huatuo, sha256_file


VERSION = "huatuo-bare-claim-drafts-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = json.loads(args.questions.read_text())
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "image_root": str(args.image_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "draft_prompt": "exact source question; no appended answer-format instruction",
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    constructor = import_huatuo(Path("/home/dbw/HuatuoGPT-Vision"))
    bot = constructor(str(args.model_path), device="cuda:0")
    raw_path = args.output_dir / "raw.jsonl"
    completed = 0
    for index, row in enumerate(rows):
        record = {
            "version": VERSION,
            "question_id": int(row["qid"]),
            "image": str(row["img_name"]),
            "truth": str(row["answer"]),
            "status": "error",
        }
        try:
            with Image.open(args.image_root / row["img_name"]) as opened:
                image = opened.convert("RGB")
            draft = generate_with_nll(
                bot, str(row["question"]), image,
                max_new_tokens=8, repetition_penalty=1.0,
            )
            parsed = parse_answer(draft["text"], answer_type="binary")
            record.update({
                "status": "ok",
                "draft": {
                    **draft,
                    "prediction": parsed.labels[0] if parsed.labels else "invalid",
                    "parse_status": parsed.status,
                },
            })
            completed += 1
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
        with raw_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[{index + 1}/{len(rows)}] qid={row['qid']} {record['status']}", flush=True)
    summary = {
        "version": VERSION,
        "status": "complete" if completed == len(rows) else "partial",
        "n": len(rows),
        "completed": completed,
        "errors": len(rows) - completed,
        "config": config,
        "raw_sha256": sha256_file(raw_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
