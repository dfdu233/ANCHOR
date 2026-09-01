#!/usr/bin/env python3
"""Reference-grounded pairwise judge for medical report generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are evaluating two generated chest X-ray reports against a
reference report. You cannot see the image. Judge only reference-grounded
clinical factual alignment. Do not reward verbosity. Penalize unsupported
findings, contradictions, wrong laterality/severity, and omissions of major
reference findings. Return one JSON object and no prose:
{"a_factuality": integer 0-10, "b_factuality": integer 0-10,
 "a_hallucination": integer 0-5, "b_hallucination": integer 0-5,
 "preference": "A" or "B" or "tie",
 "a_unsupported": [short strings], "b_unsupported": [short strings],
 "a_omissions": [short strings], "b_omissions": [short strings],
 "reason": "one concise comparison"}
Factuality 10 means fully aligned. Hallucination 0 means no unsupported claim."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qid(row: dict[str, Any]) -> str:
    return str(row.get("question_id", row.get("qid", row.get("item_id", ""))))


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate(result: dict[str, Any]) -> None:
    for key in ("a_factuality", "b_factuality"):
        if not isinstance(result.get(key), int) or not 0 <= result[key] <= 10:
            raise ValueError(f"invalid {key}")
    for key in ("a_hallucination", "b_hallucination"):
        if not isinstance(result.get(key), int) or not 0 <= result[key] <= 5:
            raise ValueError(f"invalid {key}")
    if result.get("preference") not in {"A", "B", "tie"}:
        raise ValueError("invalid preference")


def request_judgment(
    endpoint: str,
    api_key: str,
    model: str,
    reference: str,
    answer_a: str,
    answer_b: str,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    user_prompt = (
        f"REFERENCE REPORT:\n{reference}\n\n"
        f"REPORT A:\n{answer_a}\n\nREPORT B:\n{answer_b}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ANCHOR-medical-evaluation/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read())
    content = raw["choices"][0]["message"]["content"]
    result = extract_json(content)
    validate(result)
    provenance = {
        "response_model": raw.get("model"),
        "response_id": raw.get("id"),
        "usage": raw.get("usage"),
        "request_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    return result, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-a", type=Path, required=True)
    parser.add_argument("--answers-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-a", default="native")
    parser.add_argument("--name-b", required=True)
    parser.add_argument("--maximum", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--endpoint", default="https://opencode.ai/zen/go/v1/chat/completions"
    )
    parser.add_argument("--api-key-env", default="OPENCODE_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    rows_a = load_jsonl(args.answers_a)
    rows_b = load_jsonl(args.answers_b)
    by_b = {qid(row): row for row in rows_b}
    selected = rows_a[: args.maximum or None]
    output: list[dict[str, Any]] = []
    for index, row_a in enumerate(selected):
        identifier = qid(row_a)
        row_b = by_b.get(identifier)
        if row_b is None:
            raise ValueError(f"missing paired qid {identifier}")
        reference = str(
            row_a.get("ground_truth", row_a.get("gt_answer", ""))
        ).strip()
        answer_a = str(row_a.get("model_answer", row_a.get("text", ""))).strip()
        answer_b = str(row_b.get("model_answer", row_b.get("text", ""))).strip()
        error = None
        for attempt in range(1, args.retries + 1):
            try:
                result, provenance = request_judgment(
                    args.endpoint,
                    api_key,
                    args.model,
                    reference,
                    answer_a,
                    answer_b,
                    args.timeout,
                )
                output.append(
                    {
                        "question_id": identifier,
                        "name_a": args.name_a,
                        "name_b": args.name_b,
                        "reference": reference,
                        "answer_a": answer_a,
                        "answer_b": answer_b,
                        "judge": result,
                        "provenance": provenance,
                    }
                )
                error = None
                break
            except (ValueError, KeyError, urllib.error.URLError) as exc:
                error = repr(exc)
                if attempt < args.retries:
                    time.sleep(2**attempt)
        if error is not None:
            output.append(
                {
                    "question_id": identifier,
                    "name_a": args.name_a,
                    "name_b": args.name_b,
                    "error": error,
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
        )
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(selected)}",
                    "question_id": identifier,
                    "error": error,
                    "preference": (
                        output[-1].get("judge", {}).get("preference")
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
