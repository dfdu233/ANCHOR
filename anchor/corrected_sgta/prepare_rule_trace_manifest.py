"""Build a deterministic error/control manifest for mechanistic RULE traces.

This module is deliberately CPU-only.  It selects baseline errors first and
matches each to a correctly answered example with the same binary target and
the highest question-token Jaccard similarity.  The manifest freezes the
cohort before any expensive internal-state tracing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from corrected_sgta.evaluate_medheval_answers import rule_pope_prediction


VERSION = "rule-mechanistic-trace-manifest-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    return rows


def qid(row: dict[str, Any]) -> str:
    value = row.get("question_id", row.get("qid"))
    if value is None:
        raise ValueError("row has no question_id/qid")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_tokens(row: dict[str, Any]) -> set[str]:
    value = str(row.get("question", row.get("prompt", row.get("text", ""))))
    return set(TOKEN_RE.findall(value.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def baseline_text(row: dict[str, Any]) -> str:
    for key in ("base_text", "answer", "text"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"baseline row qid={qid(row)} has no answer text")


def build_manifest(
    questions: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    max_errors: int,
    forced_qids: set[str],
) -> list[dict[str, Any]]:
    question_by_qid = {qid(row): row for row in questions}
    baseline_by_qid = {qid(row): row for row in baselines}
    common = sorted(set(question_by_qid) & set(baseline_by_qid), key=lambda x: int(x))
    status = {}
    for key in common:
        question = question_by_qid[key]
        baseline = baseline_by_qid[key]
        gt = rule_pope_prediction(
            question.get("answer", question.get("gt_answer", baseline.get("gt_answer")))
        )
        pred = rule_pope_prediction(baseline_text(baseline))
        status[key] = {"gt": gt, "prediction": pred, "correct": gt == pred}

    errors = [key for key in common if not status[key]["correct"]]
    forced_errors = [key for key in sorted(forced_qids, key=int) if key in errors]
    remaining = [key for key in errors if key not in forced_qids]
    # Stable hash prevents selecting a convenient-looking subset.
    remaining.sort(key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    chosen_errors = (forced_errors + remaining)[:max_errors]

    unused_controls = {key for key in common if status[key]["correct"]}
    manifest: list[dict[str, Any]] = []
    for error_qid in chosen_errors:
        error = question_by_qid[error_qid]
        target = status[error_qid]["gt"]
        candidates = [
            key for key in unused_controls if status[key]["gt"] == target
        ]
        if not candidates:
            raise ValueError(f"no same-label control for error qid={error_qid}")
        error_tokens = question_tokens(error)
        control_qid = max(
            candidates,
            key=lambda key: (
                jaccard(error_tokens, question_tokens(question_by_qid[key])),
                -int(key),
            ),
        )
        unused_controls.remove(control_qid)
        similarity = jaccard(
            error_tokens, question_tokens(question_by_qid[control_qid])
        )
        for role, key, pair_key in (
            ("error", error_qid, control_qid),
            ("matched_correct_control", control_qid, error_qid),
        ):
            question = question_by_qid[key]
            manifest.append(
                {
                    "question_id": key,
                    "pair_question_id": pair_key,
                    "role": role,
                    "match_jaccard": similarity,
                    "image": question.get("image"),
                    "question": question.get("question", question.get("text")),
                    "gt_answer": question.get(
                        "answer", question.get("gt_answer")
                    ),
                    "baseline_text": baseline_text(baseline_by_qid[key]),
                    "baseline_prediction": status[key]["prediction"],
                }
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-errors", type=int, default=12)
    parser.add_argument("--force-qid", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(
        load_rows(args.questions),
        load_rows(args.baseline),
        args.max_errors,
        set(args.force_qid),
    )
    payload = {
        "version": VERSION,
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256(args.questions),
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "selection": {
            "max_errors": args.max_errors,
            "forced_qids": sorted(set(args.force_qid), key=int),
            "control_match": "same RULE target; maximum question-token Jaccard; no replacement",
        },
        "n_rows": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
