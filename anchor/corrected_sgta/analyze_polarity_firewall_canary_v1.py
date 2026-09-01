"""Frozen five-arm analysis for the target-blind Polarity Firewall canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from anchor.corrected_sgta.evaluate_medheval_answers import (
    normalize_binary_reference,
    parse_answer,
)


ROOT = Path(__file__).resolve().parents[2]
ARMS = (
    "raw_rag",
    "no_context",
    "depolarized_rag",
    "token_matched_neutral_rag",
    "query_term_only_neutral_rag",
)


def manifest_name(arm: str, model: str) -> str:
    if arm == "token_matched_neutral_rag":
        return f"{model}_token_matched_neutral_rag"
    return arm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"expected list: {path}")
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_qid(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or row.get("qid") or row.get("id"))


def prediction(row: dict[str, Any]) -> str:
    parsed = parse_answer(row.get("text", ""), answer_type="binary")
    return parsed.labels[0] if parsed.labels else "invalid"


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = Counter((row["truth"], row["prediction"]) for row in rows)
    tp = confusion[("yes", "yes")]
    tn = confusion[("no", "no")]
    fp = confusion[("no", "yes")] + confusion[("no", "invalid")]
    fn = confusion[("yes", "no")] + confusion[("yes", "invalid")]
    n_yes = sum(row["truth"] == "yes" for row in rows)
    n_no = sum(row["truth"] == "no" for row in rows)
    correct = tp + tn
    tpr = tp / n_yes if n_yes else None
    tnr = tn / n_no if n_no else None
    return {
        "n": len(rows),
        "accuracy_invalid_as_error": correct / len(rows) if rows else None,
        "balanced_accuracy_invalid_as_error": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "tp": tp,
        "tn": tn,
        "fp_including_invalid_negative_truth": fp,
        "fn_including_invalid_positive_truth": fn,
        "invalid": sum(row["prediction"] == "invalid" for row in rows),
        "state_alignment_all_denominator": sum(row["state_aligned"] for row in rows) / len(rows) if rows else None,
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
    }


def cluster_bootstrap_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    lmap = {row["qid"]: row for row in left}
    rmap = {row["qid"]: row for row in right}
    if lmap.keys() != rmap.keys():
        raise ValueError("paired arm QID mismatch")
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for qid, row in lmap.items():
        by_cluster[row["patient_id"]].append(qid)
    clusters = sorted(by_cluster)
    observed = statistics.fmean(value(lmap[q]) - value(rmap[q]) for q in lmap)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        chosen = rng.choices(clusters, k=len(clusters))
        qids = [qid for cluster in chosen for qid in by_cluster[cluster]]
        samples.append(statistics.fmean(value(lmap[q]) - value(rmap[q]) for q in qids))
    samples.sort()
    def quantile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        return samples[lower] * (1 - fraction) + samples[upper] * fraction
    return {
        "estimate": float(observed),
        "ci95": [quantile(0.025), quantile(0.975)],
        "clusters": len(clusters),
        "replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "corrected_runs/polarity_firewall_canary_v1")
    parser.add_argument("--models", nargs="+", default=["huatuo", "hulu"])
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shared_manifests = {
        arm: load_json(args.root / f"{arm}.json")
        for arm in ARMS
        if arm != "token_matched_neutral_rag"
    }
    qids = [row_qid(row) for row in shared_manifests["raw_rag"]]
    for arm, rows in shared_manifests.items():
        if [row_qid(row) for row in rows] != qids:
            raise ValueError(f"manifest order mismatch: {arm}")
    meta = {row_qid(row): row for row in shared_manifests["raw_rag"]}
    result: dict[str, Any] = {
        "protocol": "polarity-firewall-five-arm-canary-analysis-v1",
        "evidence_level": "development_canary_not_confirmatory",
        "arms": list(ARMS),
        "models": {},
        "manifest_sha256": {},
    }
    all_rows: list[dict[str, Any]] = []
    for model in args.models:
        model_manifests = {
            arm: load_json(args.root / f"{manifest_name(arm, model)}.json") for arm in ARMS
        }
        for arm, rows in model_manifests.items():
            if [row_qid(row) for row in rows] != qids:
                raise ValueError(f"manifest order mismatch: {model}/{arm}")
        result["manifest_sha256"][model] = {
            arm: sha256(args.root / f"{manifest_name(arm, model)}.json") for arm in ARMS
        }
        answer_paths: dict[str, Path] = {}
        for arm in ARMS:
            cached = args.root / "cached_answers" / model / arm / "answers.jsonl"
            generated = args.root / "generated_answers" / model / arm / "answers.jsonl"
            answer_paths[arm] = cached if cached.is_file() else generated
            if not answer_paths[arm].is_file():
                raise FileNotFoundError(f"missing answer arm for {model}/{arm}: {answer_paths[arm]}")
        answers = {arm: load_jsonl(path) for arm, path in answer_paths.items()}
        indexed = {arm: {row_qid(row): row for row in rows} for arm, rows in answers.items()}
        for arm in ARMS:
            if set(indexed[arm]) != set(qids):
                raise ValueError(f"answer coverage mismatch: {model}/{arm}")
        raw_truth = {
            qid: normalize_binary_reference(indexed["raw_rag"][qid].get("gt_ans"))
            for qid in qids
        }
        if any(value is None for value in raw_truth.values()):
            raise ValueError(f"non-binary or missing frozen truth for {model}")
        model_rows: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            rows = []
            for qid in qids:
                info = meta[qid]
                pred = prediction(indexed[arm][qid])
                retrieval = info["selection_retrieval_polarity"]
                aligned_label = "yes" if retrieval == "positive" else "no"
                row = {
                    "model": model,
                    "arm": arm,
                    "qid": qid,
                    "patient_id": str(info["patient_id"]),
                    "finding_group": info["selection_group"],
                    "retrieval_polarity": retrieval,
                    "truth": raw_truth[qid],
                    "prediction": pred,
                    "state_aligned": int(pred == aligned_label),
                    "correct": int(pred == raw_truth[qid]),
                }
                rows.append(row)
                all_rows.append(row)
            model_rows[arm] = rows
        raw_by_qid = {row["qid"]: row for row in model_rows["raw_rag"]}
        arm_result: dict[str, Any] = {}
        for offset, arm in enumerate(ARMS):
            rows = model_rows[arm]
            changes = Counter()
            for row in rows:
                old = raw_by_qid[row["qid"]]
                if old["correct"] == 0 and row["correct"] == 1:
                    changes["rescue"] += 1
                elif old["correct"] == 1 and row["correct"] == 0:
                    changes["harm"] += 1
                else:
                    changes["unchanged_correctness"] += 1
            arm_result[arm] = {
                "metrics": metric(rows),
                "raw_to_arm_changes": dict(changes),
                "raw_minus_arm_state_alignment": cluster_bootstrap_delta(
                    model_rows["raw_rag"], rows, lambda row: float(row["state_aligned"]),
                    replicates=args.bootstrap_replicates, seed=args.seed + offset,
                ),
                "answer_path": str(answer_paths[arm].relative_to(ROOT)),
                "answer_sha256": sha256(answer_paths[arm]),
            }
        result["models"][model] = arm_result
    rows_path = args.root / "analysis_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    result["rows"] = str(rows_path.relative_to(ROOT))
    result["rows_sha256"] = sha256(rows_path)
    out = args.root / "analysis_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
