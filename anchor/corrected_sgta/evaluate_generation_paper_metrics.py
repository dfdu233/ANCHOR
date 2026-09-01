#!/usr/bin/env python3
"""Paper-style text metrics for report generation outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

from corrected_sgta.oe_metrics import word_tokens


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference_of(row: dict) -> str:
    return str(row.get("gt_ans") or row.get("gt_answer") or row.get("answer") or row.get("ground_truth") or "").strip()


def prediction_of(row: dict) -> str:
    return str(row.get("text") or row.get("model_answer") or row.get("answer") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = []
    for row in load_jsonl(args.answers):
        pred = prediction_of(row)
        ref = reference_of(row)
        if pred and ref:
            pairs.append((pred, ref))
    if not pairs:
        raise SystemExit("no valid prediction/reference pairs")

    smooth = SmoothingFunction().method4
    refs_tok = [[word_tokens(ref)] for _, ref in pairs]
    preds_tok = [word_tokens(pred) for pred, _ in pairs]
    weights = {
        "bleu1": (1.0, 0.0, 0.0, 0.0),
        "bleu2": (0.5, 0.5, 0.0, 0.0),
        "bleu3": (1 / 3, 1 / 3, 1 / 3, 0.0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    corpus = {
        key: float(corpus_bleu(refs_tok, preds_tok, weights=value, smoothing_function=smooth))
        for key, value in weights.items()
    }
    sent = {
        key: sum(
            float(sentence_bleu([word_tokens(ref)], word_tokens(pred), weights=value, smoothing_function=smooth))
            for pred, ref in pairs
        )
        / len(pairs)
        for key, value in weights.items()
    }
    rouge = Rouge()
    rouge_l = sum(
        float(rouge.get_scores(pred.lower()[:2048], ref.lower())[0]["rouge-l"]["f"])
        for pred, ref in pairs
    ) / len(pairs)
    meteor = None
    try:
        meteor = sum(
            float(single_meteor_score(reference=word_tokens(ref), hypothesis=word_tokens(pred)))
            for pred, ref in pairs
        ) / len(pairs)
    except LookupError:
        meteor = None

    pred_texts = [pred for pred, _ in pairs]
    payload = {
        "n": len(pairs),
        "metrics_fraction": {
            **{f"corpus_{key}": value for key, value in corpus.items()},
            "corpus_avg_bleu": sum(corpus.values()) / 4,
            **{f"sentence_{key}": value for key, value in sent.items()},
            "sentence_avg_bleu": sum(sent.values()) / 4,
            "rouge_l": rouge_l,
            "meteor": meteor,
        },
        "metrics_percent": {},
        "diagnostics": {
            "unique_predictions": len(set(pred_texts)),
            "top_predictions": Counter(pred_texts).most_common(10),
            "avg_prediction_tokens": sum(len(word_tokens(pred)) for pred in pred_texts) / len(pred_texts),
            "avg_reference_tokens": sum(len(word_tokens(ref)) for _, ref in pairs) / len(pairs),
            "short_normal_fraction": sum(
                ("normal" in pred.lower() and len(word_tokens(pred)) < 30) for pred in pred_texts
            )
            / len(pred_texts),
        },
    }
    payload["metrics_percent"] = {key: (None if value is None else value * 100) for key, value in payload["metrics_fraction"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), "n": len(pairs), "metrics_percent": payload["metrics_percent"], "diagnostics": payload["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
