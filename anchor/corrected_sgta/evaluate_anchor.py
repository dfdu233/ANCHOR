#!/usr/bin/env python3
"""Fit and evaluate ANCHOR from lambda-independent full-sentence caches."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from corrected_sgta.anchor_transport import (
    anchor_sequence_score,
    conformal_candidate_indices,
    file_sha256,
    load_json_or_jsonl,
    normalize_manifest_record,
    stable_json_sha256,
)
from corrected_sgta.evaluate_medheval_answers import (
    ParsedAnswer,
    infer_answer_type,
    parse_answer,
    rule_pope_prediction,
)
from corrected_sgta.evaluate_mmedrag_sequence_anchor import score_one


EVAL_VERSION = "anchor-full-text-evaluation-v1"
DEFAULT_LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_lambda_grid(value: str) -> tuple[float, ...]:
    grid = tuple(float(item) for item in value.split(",") if item.strip())
    if not grid or any(item < 0 or not math.isfinite(item) for item in grid):
        raise ValueError("lambda grid must contain finite non-negative values")
    return tuple(sorted(set(grid)))


def load_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("lambda_independent_candidate_cache") is not True:
        raise RuntimeError(f"not an ANCHOR lambda-independent cache: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"cache has no records: {path}")
    for record in records:
        if record.get("target_labels_used_for_generation_or_selection") is not False:
            raise RuntimeError("cache reports target-label use")
        if record.get("source_answer_text_retrieved") is not False:
            raise RuntimeError("cache reports source-answer retrieval")
        if len(record.get("candidates", ())) != 8:
            raise RuntimeError("all ANCHOR records must contain eight candidates")
    return payload


def full_text_correct(
    prediction: str,
    ground_truth: str,
    prompt: str,
) -> tuple[bool, bool, dict[str, Any]]:
    binary_gt = parse_answer(
        ground_truth, answer_type="binary", prompt=prompt,
        ground_truth=ground_truth,
    )
    answer_type = (
        "binary" if binary_gt.labels is not None
        else infer_answer_type(ground_truth)
    )
    gt = parse_answer(
        ground_truth,
        answer_type=answer_type,
        prompt=prompt,
        ground_truth=ground_truth,
    )
    if answer_type == "binary" and gt.labels is None:
        label = rule_pope_prediction(ground_truth)
        gt = (
            ParsedAnswer((label,), "parsed", "official_binary_ground_truth")
            if label
            else gt
        )
    pred = parse_answer(
        prediction,
        answer_type=answer_type,
        prompt=prompt,
        ground_truth=ground_truth,
    )
    strict_pred = pred
    if answer_type == "binary":
        official_label = rule_pope_prediction(prediction)
        pred = ParsedAnswer(
            (official_label,),
            "parsed",
            "rule_pope_first_sentence",
        )
    parseable = pred.labels is not None and gt.labels is not None
    correct = bool(parseable and pred.labels == gt.labels)
    detail = {
        "answer_type": answer_type,
        "prediction": list(pred.labels) if pred.labels else None,
        "ground_truth": list(gt.labels) if gt.labels else None,
        "parse_status": pred.status,
        "parser": pred.parser,
    }
    if answer_type == "binary":
        detail["strict_prediction"] = (
            list(strict_pred.labels) if strict_pred.labels else None
        )
        detail["strict_parse_status"] = strict_pred.status
        detail["strict_parser"] = strict_pred.parser
    return correct, parseable, detail


def join_cache_manifest(
    cache_path: Path,
    manifest_path: Path,
    evaluator: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if evaluator not in {"vqa", "report"}:
        raise ValueError(f"unknown evaluator: {evaluator}")
    cache = load_cache(cache_path)
    manifest = [
        normalize_manifest_record(
            row, require_answer=True, default_domain="unknown"
        )
        for row in load_json_or_jsonl(manifest_path)
    ]
    cache_ids = [row["id"] for row in cache["records"]]
    manifest_ids = [row["id"] for row in manifest]
    if cache_ids != manifest_ids:
        raise RuntimeError(
            "cache/manifest ids and order differ; evaluation requires an exact join"
        )
    rows = []
    for cached, source in zip(cache["records"], manifest):
        rows.append(
            {
                "id": source["id"],
                "domain": source["domain"],
                "patient_id": source["patient_id"],
                "prompt": source["prompt"],
                "ground_truth": source["answer"],
                "candidates": cached["candidates"],
                "cache_fingerprint": cache["fingerprint"],
                "evaluator": evaluator,
            }
        )
    return cache, rows


def load_bundle(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("bundle must be a non-empty list or {entries:[...]}")
    output = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("bundle entries must be objects")
        evaluator = str(entry.get("evaluator", ""))
        cache_path = Path(entry["cache"])
        manifest_path = Path(entry["manifest"])
        cache, rows = join_cache_manifest(cache_path, manifest_path, evaluator)
        output.append(
            {
                "cache_path": cache_path,
                "manifest_path": manifest_path,
                "evaluator": evaluator,
                "cache": cache,
                "rows": rows,
            }
        )
    models = {entry["cache"]["model"] for entry in output}
    if len(models) != 1:
        raise RuntimeError("one lambda configuration may bind only one VLM")
    return output


def attach_utilities(
    entries: list[dict[str, Any]],
    *,
    clinical: bool,
    clinical_cache: Path | None,
    clinical_batch_size: int,
) -> dict[str, Any] | None:
    clinical_contract = None
    report_candidates: list[tuple[dict[str, Any], str, str]] = []
    for entry in entries:
        for row in entry["rows"]:
            for candidate in row["candidates"]:
                if entry["evaluator"] == "vqa":
                    correct, parseable, detail = full_text_correct(
                        candidate["text"], row["ground_truth"], row["prompt"]
                    )
                    candidate["_utility"] = float(correct)
                    candidate["_admissible"] = bool(correct)
                    candidate["_parseable"] = bool(parseable)
                    candidate["_evaluation"] = detail
                else:
                    lexical = score_one(
                        candidate["text"], row["ground_truth"]
                    )
                    candidate["_lexical"] = lexical
                    candidate["_parseable"] = True
                    if not clinical:
                        candidate["_utility"] = float(lexical["rougeL"])
                        candidate["_evaluation"] = lexical
                    report_candidates.append(
                        (candidate, candidate["text"], row["ground_truth"])
                    )
    if clinical:
        if clinical_cache is None:
            raise ValueError("--clinical requires --clinical-cache")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(clinical_cache / "hf_home")
        os.environ["XDG_CACHE_HOME"] = str(clinical_cache / "xdg")
        from corrected_sgta.evaluate_medheval_report_clinical import (
            ClinicalScorers, checkpoint_contract,
        )

        metric_manifest = Path("docs/medheval_report_metric_manifest.json")
        clinical_contract = checkpoint_contract(clinical_cache, metric_manifest)
        clinical_contract["metric_manifest_sha256"] = file_sha256(metric_manifest)
        clinical_contract["packages"] = {
            name: importlib.metadata.version(package)
            for name, package in (
                ("radgraph", "radgraph"),
                ("ratescore", "RaTEScore"),
                ("chexbert", "f1chexbert"),
            )
        }
        scorer = ClinicalScorers(clinical_cache, clinical_batch_size)
        for start in range(0, len(report_candidates), clinical_batch_size):
            batch = report_candidates[start : start + clinical_batch_size]
            values = scorer.score(
                [item[1] for item in batch], [item[2] for item in batch]
            )
            for (candidate, _, _), metrics in zip(batch, values):
                candidate["_clinical"] = metrics
                candidate["_utility"] = float(metrics["radgraph_complete"])
                candidate["_evaluation"] = metrics
    return clinical_contract


def candidate_scores(row: dict[str, Any], lambda_value: float) -> list[float]:
    return [
        anchor_sequence_score(
            float(candidate["mean_image_log_probability"]),
            float(candidate["source_distance"]),
            lambda_value,
        )
        for candidate in row["candidates"]
    ]


def selected_index(row: dict[str, Any], lambda_value: float) -> int:
    return int(np.argmax(candidate_scores(row, lambda_value)))


def summarize_at_lambda(
    rows: list[dict[str, Any]],
    lambda_value: float,
    bootstrap: int = 0,
    seed: int = 20260726,
) -> dict[str, Any]:
    selected = []
    baseline = []
    oracle = []
    parseable = []
    changes = 0
    per_domain: dict[str, list[float]] = defaultdict(list)
    details = []
    for row in rows:
        index = selected_index(row, lambda_value)
        values = [float(candidate["_utility"]) for candidate in row["candidates"]]
        selected.append(values[index])
        baseline.append(values[0])
        oracle.append(max(values))
        parseable.append(bool(row["candidates"][index]["_parseable"]))
        changes += int(index != 0)
        per_domain[row["domain"]].append(values[index])
        details.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "selected_index": index,
                "selected_text": row["candidates"][index]["text"],
                "baseline_text": row["candidates"][0]["text"],
                "selected_utility": values[index],
                "baseline_utility": values[0],
                "oracle_utility": max(values),
                "selected_parseable": parseable[-1],
                "candidate_scores": candidate_scores(row, lambda_value),
            }
        )
    selected_array = np.asarray(selected, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    oracle_array = np.asarray(oracle, dtype=np.float64)
    delta = selected_array - baseline_array
    result = {
        "lambda": lambda_value,
        "n": len(rows),
        "selected_mean_utility": float(selected_array.mean()),
        "baseline_mean_utility": float(baseline_array.mean()),
        "delta": float(delta.mean()),
        "candidate_oracle_mean_utility": float(oracle_array.mean()),
        "candidate_oracle_headroom": float(
            (oracle_array - baseline_array).mean()
        ),
        "parse_rate": float(np.mean(parseable)),
        "changed_from_greedy": changes,
        "per_domain": {
            domain: {
                "n": len(values),
                "mean_utility": float(np.mean(values)),
            }
            for domain, values in sorted(per_domain.items())
        },
        "details": details,
    }
    if bootstrap:
        rng = np.random.default_rng(seed)
        patient_rows = defaultdict(list)
        for row_index, row in enumerate(rows):
            patient_rows[row["patient_id"]].append(row_index)
        patients = sorted(patient_rows)
        samples = np.empty(bootstrap, dtype=np.float64)
        for index in range(bootstrap):
            sampled_patients = rng.choice(patients, size=len(patients), replace=True)
            positions = [
                row_index for patient in sampled_patients
                for row_index in patient_rows[str(patient)]
            ]
            samples[index] = delta[np.asarray(positions, dtype=np.int64)].mean()
        result["paired_bootstrap_delta_ci95"] = [
            float(value) for value in np.quantile(samples, (0.025, 0.975))
        ]
        result["bootstrap_unit"] = "patient_id"
    return result


def robust_objective(
    rows: list[dict[str, Any]], lambda_value: float
) -> tuple[float, float, float]:
    summary = summarize_at_lambda(rows, lambda_value)
    domain_values = [
        value["mean_utility"] for value in summary["per_domain"].values()
    ]
    return (
        min(domain_values),
        float(np.mean(domain_values)),
        summary["selected_mean_utility"],
    )


def fit_lambda(args: argparse.Namespace) -> None:
    entries = load_bundle(args.bundle)
    clinical_contract = attach_utilities(
        entries,
        clinical=args.clinical,
        clinical_cache=args.clinical_cache,
        clinical_batch_size=args.clinical_batch_size,
    )
    rows = [row for entry in entries for row in entry["rows"]]
    grid = parse_lambda_grid(args.lambda_grid)
    table = []
    for value in grid:
        minimum, macro, micro = robust_objective(rows, value)
        summary = summarize_at_lambda(rows, value)
        table.append(
            {
                "lambda": value,
                "minimum_domain_utility": minimum,
                "macro_domain_utility": macro,
                "micro_utility": micro,
                "candidate_oracle_headroom": summary[
                    "candidate_oracle_headroom"
                ],
                "parse_rate": summary["parse_rate"],
            }
        )
    chosen = max(
        table,
        key=lambda row: (
            row["minimum_domain_utility"],
            row["macro_domain_utility"],
            row["micro_utility"],
            -row["lambda"],
        ),
    )["lambda"]

    domains = sorted({row["domain"] for row in rows})
    lodo = {}
    if len(domains) >= 2:
        for heldout in domains:
            train = [row for row in rows if row["domain"] != heldout]
            test = [row for row in rows if row["domain"] == heldout]
            local = max(
                grid,
                key=lambda value: (*robust_objective(train, value), -value),
            )
            lodo[heldout] = {
                "selected_lambda_without_domain": local,
                "heldout": {
                    key: value
                    for key, value in summarize_at_lambda(test, local).items()
                    if key != "details"
                },
            }
    model = entries[0]["cache"]["model"]
    config = {
        "version": EVAL_VERSION,
        "kind": "source_only_model_level_lambda",
        "model": model,
        "lambda": chosen,
        "lambda_grid": list(grid),
        "selection_objective": (
            "lexicographic maximum of minimum-domain, macro-domain, and "
            "micro full-text utility; smallest lambda breaks exact ties"
        ),
        "clinical_metric_contract": clinical_contract,
        "primary_report_utility": (
            "radgraph_complete" if args.clinical else "rougeL_diagnostic"
        ),
        "source_inputs": [
            {
                "cache": str(entry["cache_path"].resolve()),
                "cache_sha256": file_sha256(entry["cache_path"]),
                "cache_fingerprint": entry["cache"]["fingerprint"],
                "manifest": str(entry["manifest_path"].resolve()),
                "manifest_sha256": file_sha256(entry["manifest_path"]),
                "evaluator": entry["evaluator"],
            }
            for entry in entries
        ],
        "table": table,
        "leave_one_source_domain_out": lodo,
        "target_data_used": False,
        "code_sha256": file_sha256(Path(__file__)),
    }
    config["fingerprint"] = stable_json_sha256(config)
    atomic_json(args.output, config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": model,
                "lambda": chosen,
                "domains": domains,
                "target_data_used": False,
            },
            indent=2,
        )
    )


def load_lambda(args: argparse.Namespace, model: str) -> tuple[float, str | None]:
    if args.lambda_config:
        config = json.loads(args.lambda_config.read_text())
        if config.get("kind") != "source_only_model_level_lambda":
            raise RuntimeError("invalid lambda configuration")
        if config.get("model") != model:
            raise RuntimeError("lambda configuration is bound to another model")
        if config.get("target_data_used") is not False:
            raise RuntimeError("lambda configuration reports target-data use")
        return float(config["lambda"]), config["fingerprint"]
    if args.lambda_value is None:
        raise ValueError("provide --lambda-config or --lambda-value")
    return float(args.lambda_value), None


def evaluate(args: argparse.Namespace) -> None:
    cache, rows = join_cache_manifest(
        args.cache, args.manifest, args.evaluator
    )
    entries = [{"cache": cache, "rows": rows, "evaluator": args.evaluator}]
    clinical_contract = attach_utilities(
        entries,
        clinical=args.clinical,
        clinical_cache=args.clinical_cache,
        clinical_batch_size=args.clinical_batch_size,
    )
    lambda_value, lambda_fingerprint = load_lambda(args, cache["model"])
    summary = summarize_at_lambda(
        rows, lambda_value, bootstrap=args.bootstrap, seed=args.seed
    )
    conformal = None
    if args.conformal_config:
        config = json.loads(args.conformal_config.read_text())
        if config.get("lambda") != lambda_value:
            raise RuntimeError("conformal and evaluation lambda differ")
        threshold = float(config["threshold"])
        if args.evaluator == "report":
            expected_metric = (
                "radgraph_complete" if args.clinical else "rougeL_diagnostic"
            )
            declared = config["report_admissibility"]
            if declared["metric"] != expected_metric:
                raise RuntimeError("conformal report metric and evaluator differ")
            for row in rows:
                for candidate in row["candidates"]:
                    candidate["_admissible"] = bool(
                        candidate["_utility"] >= declared["threshold"]
                    )
        set_sizes = []
        covered = []
        for row, detail in zip(rows, summary["details"]):
            scores = detail["candidate_scores"]
            indices = (
                list(range(len(scores)))
                if math.isinf(threshold)
                else conformal_candidate_indices(scores, threshold)
            )
            unique_indices = []
            seen_texts = set()
            for index in indices:
                text_key = row["candidates"][index]["text"].strip()
                if text_key not in seen_texts:
                    unique_indices.append(index)
                    seen_texts.add(text_key)
            indices = unique_indices
            set_sizes.append(len(indices))
            covered.append(
                any(row["candidates"][index]["_admissible"] for index in indices)
            )
            detail["conformal_candidate_indices"] = indices
        conformal = {
            "config_fingerprint": config["fingerprint"],
            "coverage_target": config["coverage"],
            "empirical_coverage": float(np.mean(covered)),
            "mean_set_size": float(np.mean(set_sizes)),
            "vacuous_guarantee": bool(config["vacuous_guarantee"]),
            "validity_scope": config["validity_scope"],
        }
    result = {
        "version": EVAL_VERSION,
        "kind": "anchor_full_text_evaluation",
        "cache": str(args.cache.resolve()),
        "cache_sha256": file_sha256(args.cache),
        "cache_fingerprint": cache["fingerprint"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "model": cache["model"],
        "evaluator": args.evaluator,
        "clinical_metric_contract": clinical_contract,
        "primary_utility": (
            "decoded_full_text_accuracy"
            if args.evaluator == "vqa"
            else ("radgraph_complete" if args.clinical else "rougeL_diagnostic")
        ),
        "lambda": lambda_value,
        "lambda_config_fingerprint": lambda_fingerprint,
        "summary": summary,
        "conformal": conformal,
        "selection_used_complete_generated_text": True,
        "single_token_logits_used_as_prediction": False,
        "code_sha256": file_sha256(Path(__file__)),
    }
    result["fingerprint"] = stable_json_sha256(
        {key: value for key, value in result.items() if key != "summary"}
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": cache["model"],
                "lambda": lambda_value,
                "n": summary["n"],
                "baseline": summary["baseline_mean_utility"],
                "anchor": summary["selected_mean_utility"],
                "delta": summary["delta"],
                "oracle_headroom": summary["candidate_oracle_headroom"],
                "parse_rate": summary["parse_rate"],
            },
            indent=2,
        )
    )


def _finite_or_infinite_quantile(
    values: list[float], coverage: float
) -> float:
    if not values or any(math.isnan(value) or value < 0 for value in values):
        raise ValueError("invalid conformal calibration scores")
    ordered = sorted(values)
    rank = min(len(ordered), int(math.ceil((len(ordered) + 1) * coverage)))
    return float(ordered[rank - 1])


def fit_conformal(args: argparse.Namespace) -> None:
    entries = load_bundle(args.bundle)
    clinical_contract = attach_utilities(
        entries,
        clinical=args.clinical,
        clinical_cache=args.clinical_cache,
        clinical_batch_size=args.clinical_batch_size,
    )
    model = entries[0]["cache"]["model"]
    lambda_args = argparse.Namespace(
        lambda_config=args.lambda_config,
        lambda_value=None,
    )
    lambda_value, lambda_fingerprint = load_lambda(lambda_args, model)
    by_domain: dict[str, list[float]] = defaultdict(list)
    no_admissible = 0
    for entry in entries:
        for row in entry["rows"]:
            if entry["evaluator"] == "report":
                for candidate in row["candidates"]:
                    candidate["_admissible"] = bool(
                        candidate["_utility"] >= args.report_admissibility_threshold
                    )
            scores = candidate_scores(row, lambda_value)
            admissible_scores = [
                score
                for score, candidate in zip(scores, row["candidates"])
                if candidate.get("_admissible", False)
            ]
            if admissible_scores:
                nonconformity = max(scores) - max(admissible_scores)
            else:
                nonconformity = float("inf")
                no_admissible += 1
            by_domain[row["domain"]].append(float(nonconformity))
    per_domain = {
        domain: _finite_or_infinite_quantile(values, args.coverage)
        for domain, values in sorted(by_domain.items())
    }
    threshold = max(per_domain.values())
    output = {
        "version": EVAL_VERSION,
        "kind": "anchor_worst_source_conformal",
        "model": model,
        "lambda": lambda_value,
        "lambda_config_fingerprint": lambda_fingerprint,
        "coverage": args.coverage,
        "threshold": threshold,
        "per_domain_threshold": per_domain,
        "no_admissible_calibration_instances": no_admissible,
        "vacuous_guarantee": not math.isfinite(threshold),
        "validity_scope": (
            "source-mixture/Wasserstein uncertainty envelope only; no "
            "distribution-free claim for arbitrary unseen domains"
        ),
        "clinical_metric_contract": clinical_contract,
        "report_admissibility": (
            {
                "metric": (
                    "radgraph_complete" if args.clinical else "rougeL_diagnostic"
                ),
                "threshold": args.report_admissibility_threshold,
            }
        ),
        "source_bundle_sha256": file_sha256(args.bundle),
        "target_data_used": False,
        "code_sha256": file_sha256(Path(__file__)),
    }
    output["fingerprint"] = stable_json_sha256(output)
    atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "coverage": args.coverage,
                "threshold": threshold,
                "vacuous_guarantee": output["vacuous_guarantee"],
                "no_admissible": no_admissible,
            },
            indent=2,
        )
    )


def add_clinical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clinical", action="store_true")
    parser.add_argument("--clinical-cache", type=Path)
    parser.add_argument("--clinical-batch-size", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-only lambda fitting, full-text evaluation, and worst-source "
            "conformal calibration for ANCHOR."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit-lambda")
    fit.add_argument("--bundle", type=Path, required=True)
    fit.add_argument(
        "--lambda-grid",
        default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID),
    )
    fit.add_argument("--output", type=Path, required=True)
    add_clinical_arguments(fit)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--cache", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument(
        "--evaluator", choices=("vqa", "report"), required=True
    )
    group = evaluation.add_mutually_exclusive_group(required=True)
    group.add_argument("--lambda-config", type=Path)
    group.add_argument("--lambda-value", type=float)
    evaluation.add_argument("--conformal-config", type=Path)
    evaluation.add_argument("--bootstrap", type=int, default=10_000)
    evaluation.add_argument("--seed", type=int, default=20260726)
    evaluation.add_argument("--output", type=Path, required=True)
    add_clinical_arguments(evaluation)

    conformal = subparsers.add_parser("fit-conformal")
    conformal.add_argument("--bundle", type=Path, required=True)
    conformal.add_argument("--lambda-config", type=Path, required=True)
    conformal.add_argument("--coverage", type=float, default=0.90)
    conformal.add_argument(
        "--report-admissibility-threshold", type=float, default=0.30
    )
    conformal.add_argument("--output", type=Path, required=True)
    add_clinical_arguments(conformal)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "clinical_batch_size", 1) <= 0:
        raise ValueError("clinical batch size must be positive")
    if args.command == "fit-lambda":
        fit_lambda(args)
    elif args.command == "evaluate":
        if args.bootstrap <= 0:
            raise ValueError("bootstrap must be positive")
        evaluate(args)
    elif args.command == "fit-conformal":
        if not 0.0 < args.coverage < 1.0:
            raise ValueError("coverage must lie strictly between zero and one")
        fit_conformal(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
