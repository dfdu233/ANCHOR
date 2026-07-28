#!/usr/bin/env python3
"""Analyze ANCHOR-Riemann gate outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from corrected_sgta.evaluate_medheval_answers import parse_answer
from corrected_sgta.oe_metrics_v2 import lexical_metrics


ANALYSIS_VERSION = "anchor-riemann-gate-analysis-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_output(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"no records in {path}")
    return payload


def first_label(text: str, prompt: str, reference: str) -> tuple[str | None, str]:
    parsed = parse_answer(text, answer_type="binary", prompt=prompt, ground_truth=reference)
    if parsed.labels is None:
        parsed = parse_answer(text, answer_type="choice", prompt=prompt, ground_truth=reference)
    if parsed.labels is None:
        return None, parsed.status
    return parsed.labels[0], parsed.status


def ce_correct(text: str, reference: str, prompt: str) -> tuple[bool, bool, str | None, str | None, str]:
    pred, status = first_label(text, prompt, reference)
    truth, truth_status = first_label(reference, prompt, reference)
    parseable = pred is not None
    return bool(parseable and truth is not None and pred == truth), parseable, pred, truth, status if parseable else f"pred_{status};truth_{truth_status}"


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def auc_from_scores(scores: list[float], labels: list[int]) -> float | None:
    """AUC where lower score should indicate positive/correct.

    Returns None if only one class is present.
    """

    if len(scores) != len(labels) or not scores:
        return None
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = ties = total = 0
    for p in positives:
        for n in negatives:
            total += 1
            if p < n:
                wins += 1
            elif p == n:
                ties += 1
    return float((wins + 0.5 * ties) / total)


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array, dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) < 3 or len(set(labels)) < 2:
        return None
    x = rankdata(scores)
    y = rankdata([float(value) for value in labels])
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ce_metrics(records: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    correct = parseable = 0
    details: list[dict[str, Any]] = []
    for row in records:
        if selector == "greedy":
            index = 0
        else:
            index = int(row[f"{selector}_selected_index"])
        candidate = row["candidates"][index]
        ok, parsed, pred, truth, status = ce_correct(
            candidate["text"], row["reference"], row["prompt"]
        )
        correct += int(ok)
        parseable += int(parsed)
        details.append(
            {
                "id": row["id"],
                "selected_index": index,
                "correct": ok,
                "parseable": parsed,
                "prediction": pred,
                "truth": truth,
                "parse_status": status,
                "text": candidate["text"],
            }
        )
    n = len(records)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "parseable": parseable,
        "parse_rate": parseable / n if n else 0.0,
        "details": details,
    }


def ce_candidate_oracle(records: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = greedy = parseable_any = 0
    rescue = harm = 0
    candidate_scores: dict[str, list[float]] = {
        "sequence_nll": [],
        "source_manifold_distance": [],
        "dirichlet_energy": [],
        "riemann_energy": [],
        "random_manifold_energy": [],
    }
    candidate_labels: list[int] = []
    for row in records:
        statuses = []
        for cand in row["candidates"]:
            ok, parsed, *_ = ce_correct(cand["text"], row["reference"], row["prompt"])
            statuses.append(bool(ok))
            candidate_labels.append(int(ok))
            for key in candidate_scores:
                candidate_scores[key].append(float(cand[key]))
        greedy_ok = statuses[0]
        oracle_ok = any(statuses)
        greedy += int(greedy_ok)
        oracle += int(oracle_ok)
        parseable_any += int(
            any(ce_correct(c["text"], row["reference"], row["prompt"])[1] for c in row["candidates"])
        )
        rescue += int((not greedy_ok) and oracle_ok)
        harm += int(greedy_ok and not oracle_ok)
    n = len(records)
    auc = {
        key: auc_from_scores(values, candidate_labels)
        for key, values in candidate_scores.items()
    }
    corr = {
        key: spearman(values, candidate_labels)
        for key, values in candidate_scores.items()
    }
    return {
        "n": n,
        "greedy_accuracy": greedy / n if n else 0.0,
        "candidate_oracle_accuracy": oracle / n if n else 0.0,
        "oracle_headroom": (oracle - greedy) / n if n else 0.0,
        "candidate_any_parse_rate": parseable_any / n if n else 0.0,
        "oracle_rescue": rescue,
        "oracle_harm": harm,
        "candidate_level_auc_lower_is_better": auc,
        "candidate_level_spearman_score_vs_correct": corr,
    }


def oe_metric_for(text: str, reference: str) -> dict[str, float]:
    return lexical_metrics(text, reference)


def oe_metrics(records: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "bleu4": [],
        "rouge_1": [],
        "rouge_2": [],
        "rouge_l": [],
        "token_f1": [],
        "length": [],
    }
    details = []
    for row in records:
        index = 0 if selector == "greedy" else int(row[f"{selector}_selected_index"])
        candidate = row["candidates"][index]
        scores = oe_metric_for(candidate["text"], row["reference"])
        for key, value in scores.items():
            values[key].append(float(value))
        values["length"].append(float(len(candidate["text"].split())))
        details.append(
            {
                "id": row["id"],
                "selected_index": index,
                "metrics": scores,
                "text": candidate["text"],
            }
        )
    return {
        "n": len(records),
        "mean": {key: mean(value) for key, value in values.items()},
        "details": details,
        "clinical_metrics_available": False,
        "clinical_metrics_note": "Gate uses lexical OE metrics only unless RadGraph/CheXbert/RaTEScore caches are supplied.",
    }


def oe_candidate_oracle(records: list[dict[str, Any]]) -> dict[str, Any]:
    greedy_values: dict[str, list[float]] = {"rouge_l": [], "token_f1": []}
    oracle_values: dict[str, list[float]] = {"rouge_l": [], "token_f1": []}
    candidate_scores: dict[str, list[float]] = {
        "sequence_nll": [],
        "source_manifold_distance": [],
        "dirichlet_energy": [],
        "riemann_energy": [],
        "random_manifold_energy": [],
    }
    candidate_labels: list[int] = []
    for row in records:
        per_candidate = [oe_metric_for(c["text"], row["reference"]) for c in row["candidates"]]
        greedy_values["rouge_l"].append(per_candidate[0]["rouge_l"])
        greedy_values["token_f1"].append(per_candidate[0]["token_f1"])
        oracle_index = int(np.argmax([m["rouge_l"] for m in per_candidate]))
        oracle_values["rouge_l"].append(per_candidate[oracle_index]["rouge_l"])
        oracle_values["token_f1"].append(per_candidate[oracle_index]["token_f1"])
        # Candidate-level weak label: best ROUGE-L candidate(s) per item.
        best = max(m["rouge_l"] for m in per_candidate)
        for cand, metrics in zip(row["candidates"], per_candidate):
            candidate_labels.append(int(metrics["rouge_l"] >= best - 1e-12))
            for key in candidate_scores:
                candidate_scores[key].append(float(cand[key]))
    return {
        "n": len(records),
        "greedy_mean": {key: mean(value) for key, value in greedy_values.items()},
        "candidate_oracle_mean": {
            key: mean(value) for key, value in oracle_values.items()
        },
        "oracle_headroom": {
            key: mean(oracle_values[key]) - mean(greedy_values[key])
            for key in greedy_values
        },
        "candidate_level_auc_lower_is_better_weak_rouge_oracle": {
            key: auc_from_scores(values, candidate_labels)
            for key, values in candidate_scores.items()
        },
        "candidate_level_spearman_score_vs_weak_correct": {
            key: spearman(values, candidate_labels)
            for key, values in candidate_scores.items()
        },
    }


def rescue_harm(
    records: list[dict[str, Any]],
    task: str,
    selector: str,
) -> dict[str, Any]:
    if task != "ce":
        return {"available": False, "reason": "OE uses continuous metrics"}
    rescue = harm = changed = 0
    examples = []
    for row in records:
        greedy_ok = ce_correct(row["candidates"][0]["text"], row["reference"], row["prompt"])[0]
        sel = int(row[f"{selector}_selected_index"])
        selected_ok = ce_correct(row["candidates"][sel]["text"], row["reference"], row["prompt"])[0]
        changed += int(sel != 0)
        rescue += int((not greedy_ok) and selected_ok)
        harm += int(greedy_ok and not selected_ok)
        if sel != 0 and len(examples) < 10:
            examples.append(
                {
                    "id": row["id"],
                    "selected_index": sel,
                    "greedy_correct": greedy_ok,
                    "selected_correct": selected_ok,
                    "greedy_text": row["candidates"][0]["text"],
                    "selected_text": row["candidates"][sel]["text"],
                    "reference": row["reference"],
                }
            )
    return {
        "available": True,
        "changed": changed,
        "rescue": rescue,
        "harm": harm,
        "examples": examples,
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    records = payload["records"]
    selectors = ["greedy", "nll", "random", "riemann"]
    if task == "ce":
        metrics = {name: ce_metrics(records, name) for name in selectors}
        oracle = ce_candidate_oracle(records)
        primary_geometry_auc = oracle["candidate_level_auc_lower_is_better"].get(
            "riemann_energy"
        )
        nll_auc = oracle["candidate_level_auc_lower_is_better"].get("sequence_nll")
        gate_pass = bool(
            oracle["oracle_headroom"] >= 0.05
            and primary_geometry_auc is not None
            and primary_geometry_auc >= 0.70
            and (nll_auc is None or primary_geometry_auc > nll_auc)
            and metrics["riemann"]["accuracy"] >= metrics["greedy"]["accuracy"]
        )
    else:
        metrics = {name: oe_metrics(records, name) for name in selectors}
        oracle = oe_candidate_oracle(records)
        primary_geometry_auc = oracle[
            "candidate_level_auc_lower_is_better_weak_rouge_oracle"
        ].get("riemann_energy")
        nll_auc = oracle[
            "candidate_level_auc_lower_is_better_weak_rouge_oracle"
        ].get("sequence_nll")
        gate_pass = bool(
            oracle["oracle_headroom"]["rouge_l"] > 0.01
            and primary_geometry_auc is not None
            and primary_geometry_auc >= 0.65
            and (nll_auc is None or primary_geometry_auc > nll_auc)
            and metrics["riemann"]["mean"]["rouge_l"] >= metrics["greedy"]["mean"]["rouge_l"]
        )
    compact = {
        name: (
            {
                "accuracy": metrics[name]["accuracy"],
                "parse_rate": metrics[name]["parse_rate"],
            }
            if task == "ce"
            else metrics[name]["mean"]
        )
        for name in selectors
    }
    return {
        "version": ANALYSIS_VERSION,
        "input_fingerprint": payload["fingerprint"],
        "task": task,
        "n": len(records),
        "compact_metrics": compact,
        "candidate_oracle": oracle,
        "rescue_harm": {
            name: rescue_harm(records, task, name)
            for name in ("nll", "random", "riemann")
        },
        "gate": {
            "pass": gate_pass,
            "primary_geometry_auc": primary_geometry_auc,
            "sequence_nll_auc": nll_auc,
            "decision": "continue_to_pilot" if gate_pass else "stop_or_demote_riemann",
            "notes": [
                "CE/OE use complete generated sentences; canonical label logits are not final predictions.",
                "OE gate uses lexical metrics unless clinical metric caches are supplied.",
                "Target labels/references are used only for post-hoc gate evaluation, not generation or selection.",
            ],
        },
        "full_metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_output(args.input)
    report = summarize(payload)
    atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "task": report["task"],
        "n": report["n"],
        "compact_metrics": report["compact_metrics"],
        "gate": report["gate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
