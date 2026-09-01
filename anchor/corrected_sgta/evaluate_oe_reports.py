#!/usr/bin/env python3
"""Unified task-aware evaluator for medical OE/report generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import nltk

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from corrected_sgta.oe_metrics import token_f1, word_tokens
from corrected_sgta.report_protocol import (
    VERSION as REPORT_PROTOCOL_VERSION,
    has_unnegated_abnormal_finding,
    infer_report_task,
    is_normal_template,
)

VERSION = "anchor-oe-evaluator-v3-paper-metrics"
PINNED_NLTK_DATA = Path("/home/dbw/nltk_data")
if PINNED_NLTK_DATA.is_dir() and str(PINNED_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(PINNED_NLTK_DATA))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLINICAL_PYTHON = Path("/root/autodl-tmp/envs/medheval-report-eval/bin/python")
DEFAULT_METRIC_MANIFESTS = (
    REPO_ROOT / "docs/medheval_report_metric_manifest.json",
    Path("/root/autodl-tmp/Hulu-Med/MedUniEval/docs/medheval_report_metric_manifest.json"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_payload(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("records", "rows", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    if not isinstance(payload, list):
        raise ValueError(f"unsupported OE payload in {path}")
    return payload


def _prediction_from(row: Mapping[str, Any], field: str) -> str:
    if field != "auto":
        value: Any = row
        for part in field.split("."):
            value = value[part]
        if isinstance(value, Mapping):
            value = value.get("text", "")
        return str(value).strip()
    for key in ("model_answer", "prediction", "selected_text", "text"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    if isinstance(row.get("greedy"), Mapping):
        return str(row["greedy"].get("text", "")).strip()
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates:
        selected = row.get("selected_index", row.get("anchor_flow_selected_index", 0))
        try:
            return str(candidates[int(selected)].get("text", "")).strip()
        except (IndexError, TypeError, ValueError):
            return str(candidates[0].get("text", "")).strip()
    raise ValueError("could not find a prediction field")


def normalize_rows(inputs: Iterable[Path], prediction_field: str, real_view_only: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for index, row in enumerate(load_payload(path)):
            if real_view_only and str(row.get("view", "real")) != "real":
                continue
            reference = str(row.get("ground_truth") or row.get("reference") or row.get("answer") or "").strip()
            if not reference:
                raise ValueError(f"{path}:{index + 1} has no reference")
            prediction = _prediction_from(row, prediction_field)
            # Empty outputs are model failures, not missing rows.  Keep them in
            # every lexical and clinical denominator.
            item_id = str(row.get("item_id") or row.get("id") or row.get("qid") or row.get("question_id") or index)
            method = str(row.get("method") or row.get("selected_method") or "greedy")
            protocol_parts = [str(row.get(key, "")).strip() for key in ("conv_mode", "prompt_mode")]
            protocol_parts = [value for value in protocol_parts if value and value != "unknown"]
            if protocol_parts:
                method = "|".join([method, *protocol_parts])
            task = infer_report_task(row)
            key = stable_sha256({"input": str(path.resolve()), "item_id": item_id, "method": method, "prediction": prediction})
            if key in seen:
                raise ValueError(f"duplicate normalized prediction: {item_id}/{method}")
            seen.add(key)
            output.append({
                "item_id": item_id,
                "patient_id": str(row.get("patient_id") or row.get("subject_id") or row.get("study_id") or item_id),
                "dataset": task.dataset,
                "task": task.task,
                "modality": task.modality,
                "clinical_metric_family": task.clinical_metric_family,
                "method": method,
                "model_answer": prediction,
                "ground_truth": reference,
                "source_file": str(path.resolve()),
                "source_row": index,
                "source_fingerprint": row.get("fingerprint"),
            })
    if not output:
        raise ValueError("no OE rows remained after normalization")
    return output


def safe_sentence_bleu(
    hyp_tokens: list[str], ref_tokens: list[str], order: int = 4
) -> float:
    """Return sentence BLEU with a Python-3.12-safe fallback.

    Some older NLTK releases call ``Fraction(..., _normalize=False)``, which
    is incompatible with Python 3.12.  BLEU is a secondary lexical metric here,
    so exact matches get 1.0 and failures fall back to token F1 rather than
    breaking report evaluation.
    """
    if not hyp_tokens or not ref_tokens:
        return 0.0
    try:
        smooth = SmoothingFunction().method4
        weights = tuple([1.0 / order] * order + [0.0] * (4 - order))
        return float(
            sentence_bleu(
                [ref_tokens], hyp_tokens, weights=weights,
                smoothing_function=smooth,
            )
        )
    except TypeError:
        if hyp_tokens == ref_tokens:
            return 1.0
        return float(token_f1(" ".join(hyp_tokens), " ".join(ref_tokens)))


def score_text_pair(hypothesis: str, reference: str) -> dict[str, float]:
    hyp_tokens, ref_tokens = word_tokens(hypothesis), word_tokens(reference)
    meteor = meteor_score([ref_tokens], hyp_tokens) if hyp_tokens and ref_tokens else 0.0
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    ).score(reference, hypothesis)
    result = {
        f"bleu_{order}": safe_sentence_bleu(hyp_tokens, ref_tokens, order)
        for order in range(1, 5)
    }
    for public, internal in (("rouge_1", "rouge1"), ("rouge_2", "rouge2"), ("rouge_l", "rougeL")):
        value = rouge[internal]
        result[f"{public}_precision"] = float(value.precision)
        result[f"{public}_recall"] = float(value.recall)
        result[f"{public}_f1"] = float(value.fmeasure)
    # Backward-compatible aliases remain bound to BLEU-4 and ROUGE-L F1.
    result.update(
        bleu=result["bleu_4"],
        rouge_l=result["rouge_l_f1"],
        meteor=float(meteor),
        token_f1=float(token_f1(hypothesis, reference)),
    )
    return result


def cluster_bootstrap_means(group: list[dict[str, Any]], names: tuple[str, ...], replicates: int = 5000, seed: int = 42) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        clusters[str(row["patient_id"])].append(row)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, dtype=np.float64) for name in names}
    for replicate in range(replicates):
        selected = rng.integers(0, len(keys), size=len(keys))
        sample = [row for index in selected for row in clusters[keys[int(index)]]]
        for name in names:
            draws[name][replicate] = np.mean([row["text_metrics"][name] for row in sample])
    return {
        name: {
            "estimate": float(np.mean([row["text_metrics"][name] for row in group])),
            "ci95_lower": float(np.quantile(draws[name], 0.025)),
            "ci95_upper": float(np.quantile(draws[name], 0.975)),
            "clusters": len(keys),
            "replicates": replicates,
            "seed": seed,
        }
        for name in names
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        scored.append({**row, "text_metrics": score_text_pair(row["model_answer"], row["ground_truth"]),
                       "prediction_words": len(word_tokens(row["model_answer"])),
                       "reference_words": len(word_tokens(row["ground_truth"])),
                       "normal_template": is_normal_template(row["model_answer"]),
                       "abnormal_finding": has_unnegated_abnormal_finding(row["model_answer"])})

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        metric_names = (
            "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
            "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
            "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
            "meteor", "token_f1",
        )
        return {
            "n": len(group),
            "n_patients_or_studies": len({row["patient_id"] for row in group}),
            **{name: float(np.mean([row["text_metrics"][name] for row in group])) for name in metric_names},
            "bootstrap_ci95": cluster_bootstrap_means(group, metric_names),
            "mean_prediction_words": float(np.mean([row["prediction_words"] for row in group])),
            "mean_reference_words": float(np.mean([row["reference_words"] for row in group])),
            "unique_output_rate": len({row["model_answer"] for row in group}) / len(group),
            "normal_template_rate": float(np.mean([row["normal_template"] for row in group])),
            "abnormal_finding_rate": float(np.mean([row["abnormal_finding"] for row in group])),
        }

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["dataset"], row["modality"], row["method"])].append(row)
    return {"records": scored, "overall_text_only": aggregate(scored),
            "by_dataset_modality_method": {"|".join(key): aggregate(value) for key, value in sorted(groups.items())},
            "warning": "The overall aggregate mixes modalities and is diagnostic only; paper tables must report each dataset/modality separately."}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_manifest(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    for path in DEFAULT_METRIC_MANIFESTS:
        if path.is_file():
            return path
    raise FileNotFoundError("no pinned MedHEval clinical metric manifest found")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def run_clinical(records: list[dict[str, Any]], output_dir: Path, python: Path, manifest: Path, cache: Path, validate_directions: bool) -> dict[str, Any]:
    eligible = [row for row in records if row["task"] == "report_generation" and row["clinical_metric_family"] == "chest_radiograph"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(row["dataset"], row["method"])].append(row)
    outputs: dict[str, Any] = {}
    for (dataset, method), rows in sorted(groups.items()):
        slug = _slug(f"{dataset}-{method}")
        pair_path = output_dir / "clinical_pairs" / f"{slug}.jsonl"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_path.write_text("".join(json.dumps({"item_id": row["item_id"], "patient_id": row["patient_id"], "ground_truth": row["ground_truth"], "model_answer": row["model_answer"]}, ensure_ascii=False) + "\n" for row in rows))
        target = output_dir / "clinical_metrics" / slug
        # The clinical runner owns the resume contract: it compares its full
        # fingerprint (input, checkpoints, configuration and code) before
        # reusing any records.  Always request that safe path so a process
        # interruption after ``run_manifest.json`` does not deadlock every
        # later scoring-monitor pass.
        command = [str(python), "-m", "corrected_sgta.evaluate_medheval_report_clinical", "--input", str(pair_path), "--output-dir", str(target), "--metric-manifest", str(manifest), "--cache", str(cache), "--batch-size", "8", "--resume"]
        if validate_directions and len(rows) >= 100:
            command.extend(["--validate-directions", "--min-direction-pairs", "100"])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO_ROOT / "anchor")
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        outputs[f"{dataset}|{method}"] = json.loads((target / "aggregate.json").read_text())
    return {"eligible_n": len(eligible), "excluded_n": len(records) - len(eligible),
            "policy": "clinical metrics are restricted to chest-radiograph report generation", "groups": outputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-field", default="auto")
    parser.add_argument("--include-counterfactual-views", action="store_true")
    parser.add_argument("--clinical", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--clinical-python", type=Path, default=DEFAULT_CLINICAL_PYTHON)
    parser.add_argument("--metric-manifest", type=Path)
    parser.add_argument("--clinical-cache", type=Path, default=Path("/home/dbw/model_cache/report_metrics"))
    parser.add_argument("--validate-directions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = normalize_rows(args.input, args.prediction_field, not args.include_counterfactual_views)
    summary = summarize(rows)
    config = {"version": VERSION, "report_protocol_version": REPORT_PROTOCOL_VERSION,
              "inputs": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.input],
              "prediction_field": args.prediction_field, "real_view_only": not args.include_counterfactual_views,
              "clinical": args.clinical, "validate_directions": args.validate_directions,
              "clinical_cache": str(args.clinical_cache.resolve()),
              "code_sha256": file_sha256(Path(__file__))}
    clinical, clinical_error = None, None
    if args.clinical != "off":
        try:
            manifest = resolve_manifest(args.metric_manifest)
            if not args.clinical_python.is_file():
                raise FileNotFoundError(args.clinical_python)
            clinical = run_clinical(summary["records"], args.output_dir, args.clinical_python, manifest, args.clinical_cache, args.validate_directions)
        except Exception as error:
            if args.clinical == "required":
                raise
            clinical_error = f"{type(error).__name__}: {error}"
    output = {"version": VERSION, "fingerprint": stable_sha256(config), "config": config,
              "task_validity": {"report_generation_n": sum(row["task"] == "report_generation" for row in rows),
                                "open_vqa_n": sum(row["task"] == "open_vqa" for row in rows),
                                "mixed_task_warning": len({row["task"] for row in rows}) > 1},
              "text_metrics": {key: value for key, value in summary.items() if key != "records"},
              "clinical_metrics": clinical, "clinical_error": clinical_error,
              "paper_policy": {"primary_for_chest_reports": ["radgraph", "ratescore", "chexbert"],
                               "secondary_comparability": ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_2", "rouge_l", "meteor"],
                               "ophthalmology_or_pathology": "report separately; chest-radiograph clinical metrics are invalid"}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "normalized_records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary["records"]))
    atomic_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
