"""Blind semantic adjudication for paired MIMIC question-frame responses."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path

import certifi


VERSION = "mimic-question-frame-opencode-judge-v1"
SYSTEM = """You evaluate answers to binary medical-image questions.
For each item, infer whether the MODEL RESPONSE semantically answers YES or NO
to the QUESTION. Return AMBIGUOUS only if the response does not commit.
Do not judge medical correctness and do not use outside knowledge.
Return only a JSON array with objects: {"id": string, "answer": "yes"|"no"|"ambiguous"}.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_content(content: str) -> list[dict]:
    stripped = content.strip().removeprefix("```json").removeprefix("```")
    stripped = stripped.removesuffix("```").strip()
    start, end = stripped.find("["), stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON array")
    rows = json.loads(stripped[start : end + 1])
    if not isinstance(rows, list):
        raise ValueError("judge response is not a list")
    return rows


def judge_chunk(
    rows: list[dict],
    endpoint: str,
    model: str,
    api_key: str,
    cache_dir: Path,
) -> list[dict]:
    items = [
        {
            "id": row["id"],
            "question": row["question"],
            "model_response": row["text"],
        }
        for row in rows
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(items)},
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    key = hashlib.sha256(encoded).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.exists():
        response = json.loads(cache.read_text())
    else:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "opencode/1.18.9",
            },
            method="POST",
        )
        error = None
        for attempt in range(3):
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(
                    request, timeout=180, context=context
                ) as handle:
                    response = json.loads(handle.read())
                cache.write_text(json.dumps(response, indent=2))
                break
            except Exception as caught:  # pragma: no cover - network retries
                error = caught
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"judge request failed: {error}")
    content = response["choices"][0]["message"]["content"]
    judged = parse_content(content)
    expected = {row["id"] for row in rows}
    actual = {str(row["id"]) for row in judged}
    if actual != expected:
        raise ValueError(f"judge IDs differ: expected {expected}, got {actual}")
    return judged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="https://opencode.ai/zen/go/v1/chat/completions",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="OPENCODE_API_KEY")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing environment variable {args.api_key_env}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output.parent / "judge_cache"
    cache_dir.mkdir(exist_ok=True)
    rows = read_jsonl(args.input)
    chunks = [
        rows[start : start + args.chunk_size]
        for start in range(0, len(rows), args.chunk_size)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(
                judge_chunk,
                chunk,
                args.endpoint,
                args.model,
                api_key,
                cache_dir,
            )
            for chunk in chunks
        ]
        judged = [
            row
            for future in futures
            for row in future.result()
        ]
    mapping = {str(row["id"]): row["answer"].lower() for row in judged}
    with args.output.open("w") as handle:
        for row in rows:
            answer = mapping[row["id"]]
            record = {
                "version": VERSION,
                "id": row["id"],
                "judge_prediction": (
                    answer if answer in {"yes", "no"} else None
                ),
                "judge_raw": answer,
                "model": args.model,
                "endpoint": args.endpoint,
            }
            handle.write(json.dumps(record) + "\n")
    print(
        json.dumps(
            {
                "n": len(rows),
                "parseable": sum(
                    answer in {"yes", "no"} for answer in mapping.values()
                ),
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
