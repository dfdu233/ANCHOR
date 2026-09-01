"""Target-label-free analysis for the matched retrieval-polarity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.evaluate_medheval_answers import parse_answer


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("present", "absent", "neutral", "random_deletion", "plain")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_label(text: object) -> str:
    parsed = parse_answer(text, answer_type="ternary")
    return parsed.labels[0] if parsed.labels else "invalid"


def value(label: str) -> float | None:
    return {"yes": 1.0, "maybe": 0.5, "no": 0.0}.get(label)


def paired_delta(
    pairs: list[dict[str, Any]], left: str, right: str, *, replicates: int, seed: int
) -> dict[str, Any]:
    complete = [row for row in pairs if value(row[left]) is not None and value(row[right]) is not None]
    deltas = [value(row[left]) - value(row[right]) for row in complete]
    assert all(item is not None for item in deltas)
    if not deltas:
        return {"estimate": None, "ci95": [None, None], "n_complete": 0, "replicates": replicates}
    observed = statistics.fmean(float(item) for item in deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        chosen = rng.choices(deltas, k=len(deltas))
        samples.append(statistics.fmean(float(item) for item in chosen))
    samples.sort()
    def quantile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        return samples[lower] * (1 - fraction) + samples[upper] * fraction
    return {
        "estimate": observed,
        "ci95": [quantile(0.025), quantile(0.975)],
        "n_complete": len(complete),
        "replicates": replicates,
    }


def symmetry_residual(pairs: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    complete = [
        row for row in pairs
        if value(row["present"]) is not None
        and value(row["absent"]) is not None
        and value(row["plain"]) is not None
    ]
    residuals = [
        float(value(row["present"])) + float(value(row["absent"])) - 2 * float(value(row["plain"]))
        for row in complete
    ]
    if not residuals:
        return {"estimate": None, "ci95": [None, None], "n_complete": 0, "replicates": replicates}
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(residuals, k=len(residuals))) for _ in range(replicates)
    )
    def quantile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        return samples[lower] * (1 - fraction) + samples[upper] * fraction
    return {
        "estimate": statistics.fmean(residuals),
        "ci95": [quantile(0.025), quantile(0.975)],
        "n_complete": len(complete),
        "replicates": replicates,
        "note": "zero means present and absent effects are symmetric around plain",
    }


def summarize(pairs: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    counts = {arm: dict(Counter(row[arm] for row in pairs)) for arm in ARMS}
    return {
        "n_pairs": len(pairs),
        "answer_counts": counts,
        "parse_rate": {
            arm: sum(row[arm] != "invalid" for row in pairs) / len(pairs) if pairs else None
            for arm in ARMS
        },
        "present_minus_absent": paired_delta(pairs, "present", "absent", replicates=replicates, seed=seed),
        "present_minus_neutral": paired_delta(pairs, "present", "neutral", replicates=replicates, seed=seed + 1),
        "present_minus_random_deletion": paired_delta(
            pairs, "present", "random_deletion", replicates=replicates, seed=seed + 2
        ),
        "neutral_minus_random_deletion": paired_delta(
            pairs, "neutral", "random_deletion", replicates=replicates, seed=seed + 3
        ),
        "present_minus_plain": paired_delta(
            pairs, "present", "plain", replicates=replicates, seed=seed + 4
        ),
        "absent_minus_plain": paired_delta(
            pairs, "absent", "plain", replicates=replicates, seed=seed + 5
        ),
        "symmetry_residual_present_plus_absent_minus_2plain": symmetry_residual(
            pairs, replicates=replicates, seed=seed + 6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot", type=Path,
        default=ROOT / "corrected_runs/matched_retrieval_polarity_pilot_v1/target_blind_pilot_v2.json",
    )
    parser.add_argument(
        "--answer-root", type=Path,
        default=ROOT / "corrected_runs/matched_retrieval_polarity_pilot_v1/generated_answers",
    )
    parser.add_argument("--models", nargs="+", default=["huatuo", "hulu"])
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifests = json.loads(args.pilot.read_text())
    if not isinstance(manifests, list) or len(manifests) != 160:
        raise ValueError("pilot must contain the frozen 160 arm rows")
    manifest_by_qid = {str(row["qid"]): row for row in manifests}
    if len(manifest_by_qid) != len(manifests):
        raise ValueError("pilot qids are not unique")
    expected_pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifests:
        expected_pairs[str(row["pair_id"])][str(row["arm"])] = row
    if any(set(arms) != set(ARMS) for arms in expected_pairs.values()):
        raise ValueError("every pair must contain exactly the four frozen arms")

    result: dict[str, Any] = {
        "protocol": "matched-retrieval-polarity-pilot-analysis-v1",
        "evidence_level": "target_blind_fast_causal_canary_not_confirmation",
        "ordinal_readout": {"yes": 1.0, "uncertain": 0.5, "no": 0.0, "invalid": None},
        "pilot": str(args.pilot.relative_to(ROOT)),
        "pilot_sha256": sha256(args.pilot),
        "models": {},
        "limitations": [
            "The neutral arm removes the target-state sentence from the present donor; there is no separate absent-donor neutral arm in this fast pilot.",
            "The 32-pair pilot is a directional screen; the remaining 76 preregistered pairs are the confirmation set.",
        ],
    }
    all_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(args.models):
        path = args.answer_root / model / "answers.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        answers = load_jsonl(path)
        by_qid = {str(row.get("question_id") or row.get("qid")): row for row in answers}
        if set(by_qid) != set(manifest_by_qid):
            raise ValueError(f"answer coverage mismatch for {model}")
        pairs: list[dict[str, Any]] = []
        for pair_id, arm_rows in expected_pairs.items():
            row: dict[str, Any] = {
                "model": model,
                "pair_id": pair_id,
                "finding": arm_rows["present"]["finding"],
                "source_qid": arm_rows["present"]["source_qid"],
            }
            for arm in ARMS:
                qid = str(arm_rows[arm]["qid"])
                row[arm] = parse_label(by_qid[qid].get("text", ""))
            pairs.append(row)
            all_rows.append(row)
        by_finding: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pairs:
            by_finding[row["finding"]].append(row)
        result["models"][model] = {
            "overall": summarize(pairs, replicates=args.bootstrap_replicates, seed=args.seed + model_index * 100),
            "by_finding": {
                finding: summarize(rows, replicates=args.bootstrap_replicates, seed=args.seed + model_index * 100 + offset * 10)
                for offset, (finding, rows) in enumerate(sorted(by_finding.items()))
            },
            "answers": str(path.relative_to(ROOT)),
            "answers_sha256": sha256(path),
        }
    rows_path = args.answer_root.parent / "analysis_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    result["analysis_rows"] = str(rows_path.relative_to(ROOT))
    result["analysis_rows_sha256"] = sha256(rows_path)
    output = args.answer_root.parent / "analysis_result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
