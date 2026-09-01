#!/usr/bin/env python3
"""Run and summarize a small canonical-vs-mitigation LLaVA generation audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "there",
    "in", "on", "at", "to", "of", "for", "from", "with", "and", "or",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def canonical(args: argparse.Namespace) -> None:
    from corrected_sgta.models_oe import LlavaMedOEAdapter

    rows = json.loads(args.manifest.read_text())[: args.limit]
    adapter = LlavaMedOEAdapter(conv_mode="mistral_instruct")
    answers = []
    for index, row in enumerate(rows):
        with Image.open(args.image_root / row["img_name"]) as source:
            image = source.convert("RGB")
        generation, _ = adapter.generate_oe(
            image=image,
            prompt=str(row["question"]),
            candidates=0,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
            candidate_batch=1,
        )
        answers.append({
            "question_id": str(row["qid"]),
            "text": generation.text,
            "gt_ans": str(row["answer"]),
            "metadata": {"generated_token_count": generation.token_count},
        })
    write_jsonl(args.output, answers)


def summarize(args: argparse.Namespace) -> None:
    payload = {"conditions": {}}
    for spec in args.answers:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        rows = load_jsonl(path)
        texts = [str(row.get("text", "")).strip() for row in rows]
        normalized = [text.lower().strip(".,:;!?()[]{}") for text in texts]
        token_counts = [
            int(row.get("metadata", {}).get(
                "generated_token_count",
                row.get("metadata", {}).get("decoded_sequence_token_count", len(text.split())),
            ))
            for row, text in zip(rows, texts)
        ]
        payload["conditions"][name] = {
            "answers": str(path.resolve()),
            "n": len(rows),
            "texts": texts,
            "token_counts": token_counts,
            "function_word_only_fraction": (
                sum(text in FUNCTION_WORDS for text in normalized) / max(len(texts), 1)
            ),
            "unique_predictions": len(set(texts)),
            "most_common": Counter(texts).most_common(5),
        }
    canonical_row = payload["conditions"].get("canonical")
    port_rows = [
        row for name, row in payload["conditions"].items() if name.startswith("port_")
    ]
    payload["port_failure_confirmed"] = bool(
        canonical_row
        and canonical_row["function_word_only_fraction"] < 0.5
        and any(row["function_word_only_fraction"] >= 0.5 for row in port_rows)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("canonical")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--image-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int, default=4)
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--seed", type=int, default=42)
    report = subparsers.add_parser("summarize")
    report.add_argument("--answers", nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical(args) if args.command == "canonical" else summarize(args)


if __name__ == "__main__":
    main()
