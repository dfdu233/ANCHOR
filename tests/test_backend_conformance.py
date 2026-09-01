import json
from pathlib import Path

from anchor.medeval.evaluate_backend_conformance import evaluate_conformance


def write(path: Path, values: list[str], with_tokens: bool = False) -> None:
    path.write_text("".join(
        json.dumps({
            "question_id": str(index),
            "text": value,
            "metadata": ({"generated_token_ids": [index, len(value)]} if with_tokens else {}),
        }) + "\n"
        for index, value in enumerate(values)
    ))


def test_identical_backend_passes(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    values = ["right side", "not visible", "left lower lobe"] * 7
    write(canonical, values)
    write(candidate, values)
    result = evaluate_conformance(canonical, candidate)
    assert result["passed"]
    assert result["normalized_exact"] == 1.0


def test_function_word_port_fails_even_when_aligned(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write(canonical, ["right side", "not visible"] * 10)
    write(candidate, ["The", "In"] * 10)
    result = evaluate_conformance(canonical, candidate)
    assert result["aligned"]
    assert not result["passed"]
    assert "candidate_function_word_only_rate_too_high" in result["failure_reasons"]


def test_trace_certified_gate_requires_exact_token_ids(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    values = ["right side", "not visible"] * 10
    write(canonical, values, with_tokens=True)
    write(candidate, values, with_tokens=True)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    rows[0]["metadata"]["generated_token_ids"] = [999]
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = evaluate_conformance(canonical, candidate, require_token_exact=True)
    assert not result["passed"]
    assert result["generated_token_exact_rate"] == 0.95
    assert "generated_token_ids_not_exact" in result["failure_reasons"]


def test_conformance_compares_a_frozen_prefix_without_rewriting(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write(canonical, ["one", "two", "canonical-only"])
    write(candidate, ["one", "two"])
    result = evaluate_conformance(canonical, candidate, limit=2)
    assert result["passed"]
    assert result["prefix_limit"] == 2
    assert result["n"] == 2
