#!/usr/bin/env python3
"""Reproducible CPU runner for MedHEval clinical report metrics.

The input is an existing JSONL with ``ground_truth`` and ``model_answer``.
This module never generates reports and intentionally imports the heavyweight
metric packages only after pinning the offline cache environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from unittest.mock import patch

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from corrected_sgta.report_protocol import (
    has_unnegated_abnormal_finding,
    is_normal_template,
)


VERSION = "medheval-report-clinical-runner-v1"
DEFAULT_CACHE = Path("/root/autodl-tmp/model_cache/report_metrics")
NORMAL_TEMPLATE = (
    "No acute cardiopulmonary abnormality. The lungs are clear without focal "
    "opacity, pleural effusion, or pneumothorax. Heart size is normal."
)
ABNORMAL_TEMPLATE = (
    "There is severe bilateral airspace consolidation, a large pleural "
    "effusion, pneumothorax, and marked cardiomegaly."
)
NORMAL_PATTERNS = (
    r"\bno acute (?:cardiopulmonary )?(?:abnormalit(?:y|ies)|disease|process)\b",
    r"\bno significant abnormalit",
    r"\bclear lungs\b.*\bnormal cardiomediastinal silhouette\b",
    r"\bnormal chest\b",
    r"\bno focal (?:airspace )?(?:opacity|disease|consolidation)\b",
)
STRONG_ABNORMAL_PATTERNS = (
    r"\bpneumonia\b",
    r"\bpneumothorax\b(?!\s+(?:is|are)?\s*(?:not|absent))",
    r"\bcardiomegaly\b",
    r"\b(?:moderate|large|small|bilateral|right|left) pleural effusion",
    r"\bconsolidation\b(?!\s+(?:is|are)?\s*(?:not|absent))",
    r"\batelectasis\b",
    r"\bedema\b",
    r"\bopacity\b",
    r"\bfracture\b",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def load_pairs(path: Path, maximum: int = 0) -> list[dict[str, str]]:
    """Load and strictly normalize existing report/reference pairs."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        reference = str(raw.get("ground_truth", "")).strip()
        hypothesis = str(raw.get("model_answer", "")).strip()
        identifier = str(
            raw.get("item_id", raw.get("qid", raw.get("question_id", index)))
        ).strip()
        patient_id = str(raw.get("patient_id") or raw.get("subject_id") or raw.get("study_id") or identifier).strip()
        # Empty hypotheses are valid model failures and must remain in the
        # denominator.  Only the identifier and frozen reference are required.
        if not identifier or not reference:
            raise ValueError(f"invalid report pair at JSONL line {index + 1}")
        if identifier in seen:
            raise ValueError(f"duplicate report identifier: {identifier}")
        seen.add(identifier)
        rows.append(
            {
                "item_id": identifier,
                "patient_id": patient_id,
                "ground_truth": reference,
                "model_answer": hypothesis,
                "pair_sha256": stable_json_sha256(
                    {
                        "item_id": identifier,
                        "patient_id": patient_id,
                        "ground_truth": reference,
                        "model_answer": hypothesis,
                    }
                ),
            }
        )
        if maximum and len(rows) >= maximum:
            break
    if not rows:
        raise ValueError("input contains no valid report pairs")
    return rows


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def example_binary_f1(reference: Sequence[int], hypothesis: Sequence[int]) -> float:
    ref = np.asarray(reference, dtype=np.int64)
    hyp = np.asarray(hypothesis, dtype=np.int64)
    if ref.shape != (14,) or hyp.shape != (14,):
        raise ValueError("CheXbert labels must have 14 entries")
    tp = int(((ref == 1) & (hyp == 1)).sum())
    fp = int(((ref == 0) & (hyp == 1)).sum())
    fn = int(((ref == 1) & (hyp == 0)).sum())
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator


def is_normal_report(report: str) -> bool:
    """Return a conservative normal-template flag for metric sanity checks."""
    return is_normal_template(report) and not has_unnegated_abnormal_finding(report)


def contradiction_for(reference: str) -> tuple[str, str]:
    if is_normal_report(reference):
        return ABNORMAL_TEMPLATE, "normal"
    return NORMAL_TEMPLATE, "abnormal"


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty metric records")
    scalar_keys = (
        "radgraph_simple",
        "radgraph_partial",
        "radgraph_complete",
        "ratescore",
        "chexbert_example_f1_14",
        "chexbert_exact_match_5",
    )
    scalar = {
        key: float(np.mean([float(row["metrics"][key]) for row in records]))
        for key in scalar_keys
    }
    references = np.asarray(
        [row["metrics"]["chexbert_reference_labels_14"] for row in records],
        dtype=np.int64,
    )
    hypotheses = np.asarray(
        [row["metrics"]["chexbert_hypothesis_labels_14"] for row in records],
        dtype=np.int64,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        references,
        hypotheses,
        average="macro",
        zero_division=0,
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        references.reshape(-1),
        hypotheses.reshape(-1),
        average="binary",
        zero_division=0,
    )
    output = {
        "n": len(records),
        **scalar,
        "chexbert_macro_f1_14": float(macro_f1),
        "chexbert_micro_f1_14": float(micro_f1),
    }
    for name, value in output.items():
        if name != "n" and (
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"non-finite/out-of-range aggregate metric: {name}")
    return output


def bootstrap_records(
    records: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    """Patient-cluster bootstrap over already computed per-report metrics."""
    clusters: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        clusters.setdefault(str(row["patient_id"]), []).append(index)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    names = (
        "radgraph_simple",
        "radgraph_partial",
        "radgraph_complete",
        "ratescore",
        "chexbert_example_f1_14",
        "chexbert_exact_match_5",
        "chexbert_macro_f1_14",
        "chexbert_micro_f1_14",
    )
    draws = {name: np.empty(replicates, dtype=np.float64) for name in names}
    point = summarize_records(records)
    for replicate in range(replicates):
        selected = rng.integers(0, len(keys), size=len(keys))
        indices = [index for cluster_index in selected for index in clusters[keys[int(cluster_index)]]]
        summary = summarize_records([records[index] for index in indices])
        for name in names:
            draws[name][replicate] = float(summary[name])
    return {
        name: {
            "estimate": float(point[name]),
            "ci95_lower": float(np.quantile(draws[name], 0.025)),
            "ci95_upper": float(np.quantile(draws[name], 0.975)),
            "clusters": len(keys),
            "replicates": replicates,
            "seed": seed,
        }
        for name in names
    }


def summarize_directions(
    records: Sequence[Mapping[str, Any]], minimum: int
) -> dict[str, Any]:
    if len(records) < minimum:
        raise ValueError(
            f"direction validation needs >={minimum} pairs, got {len(records)}"
        )
    metric_names = (
        "radgraph_simple",
        "radgraph_partial",
        "radgraph_complete",
        "ratescore",
        "chexbert_example_f1_14",
        "chexbert_exact_match_5",
    )

    def group_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"n": len(group), "metrics": {}}
        if not group:
            output["passed"] = False
            output["reason"] = "no examples in direction group"
            return output
        for metric in metric_names:
            matched = np.asarray(
                [row["matched"][metric] for row in group], dtype=np.float64
            )
            contradicted = np.asarray(
                [row["contradicted"][metric] for row in group],
                dtype=np.float64,
            )
            finite = bool(
                np.isfinite(matched).all() and np.isfinite(contradicted).all()
            )
            delta = float(np.mean(matched - contradicted))
            output["metrics"][metric] = {
                "matched_mean": float(matched.mean()),
                "contradicted_mean": float(contradicted.mean()),
                "mean_delta": delta,
                "finite": finite,
                "passed": finite and delta > 0.0,
            }
        output["passed"] = bool(group) and all(
            value["passed"] for value in output["metrics"].values()
        )
        return output

    normal = [row for row in records if row["reference_type"] == "normal"]
    abnormal = [row for row in records if row["reference_type"] == "abnormal"]
    all_summary = group_summary(records)
    normal_summary = group_summary(normal)
    abnormal_summary = group_summary(abnormal)
    checks = {
        "normal_no_finding_direction": normal_summary["passed"],
        "abnormal_finding_direction": abnormal_summary["passed"],
        "critical_contradiction_direction": all_summary["passed"],
    }
    return {
        "n_validation": len(records),
        "groups": {
            "normal": normal_summary,
            "abnormal": abnormal_summary,
            "all": all_summary,
        },
        "direction_checks": checks,
        "passed": all(checks.values()),
    }


class ClinicalScorers:
    """One-load wrapper preserving each official package's argument order."""

    def __init__(self, cache: Path, batch_size: int):
        # RaTEScore's PyRuSH dependency defaults to token-level DEBUG output.
        # Silence that namespace only; warnings and metric failures still surface.
        from loguru import logger

        logger.disable("PyRuSH")
        from RaTEScore import RaTEScore
        from f1chexbert import F1CheXbert
        from radgraph import F1RadGraph
        from transformers import AutoConfig, BertTokenizer

        self.radgraph = F1RadGraph(
            reward_level="all",
            model_type="modern-radgraph-xl",
            batch_size=batch_size,
            cuda=-1,
            model_cache_dir=str(cache / "radgraph"),
            tokenizer_cache_dir=str(cache / "modernbert-base"),
        )
        self.ratescore = RaTEScore(
            bert_model=str(cache / "rate-ner-deberta"),
            eval_model=str(cache / "biolord-2023-c"),
            batch_size=batch_size,
            use_gpu=False,
        )
        # f1chexbert 0.0.2 hard-codes the Hub id ``bert-base-uncased``
        # and exposes no tokenizer argument.  Resolve that exact tokenizer
        # from the pinned offline snapshot without changing scorer code.
        tokenizer_snapshot = (
            cache
            / "hf_home/hub/models--bert-base-uncased/snapshots"
            / "86b5e0934494bd15c9632b12f734a8a67f723594"
        )
        if not (tokenizer_snapshot / "vocab.txt").is_file():
            raise FileNotFoundError(tokenizer_snapshot / "vocab.txt")
        local_tokenizer = BertTokenizer.from_pretrained(
            str(tokenizer_snapshot), local_files_only=True
        )
        local_config = AutoConfig.from_pretrained(
            str(tokenizer_snapshot), local_files_only=True
        )
        with (
            patch.object(
                BertTokenizer, "from_pretrained", return_value=local_tokenizer
            ),
            patch.object(
                AutoConfig, "from_pretrained", return_value=local_config
            ),
        ):
            self.chexbert = F1CheXbert(device="cpu")

    def score(
        self, hypotheses: Sequence[str], references: Sequence[str]
    ) -> list[dict[str, Any]]:
        if len(hypotheses) != len(references) or not hypotheses:
            raise ValueError("non-empty equal hypothesis/reference lists required")
        # Official APIs intentionally use different positional orders.
        _, radgraph_rewards, _, _ = self.radgraph(
            list(references), list(hypotheses)
        )
        simple, partial, complete = radgraph_rewards
        rate = self.ratescore.compute_score(
            list(hypotheses), list(references)
        )
        hypothesis_labels = [
            self.chexbert.get_label(text) for text in hypotheses
        ]
        reference_labels = [
            self.chexbert.get_label(text) for text in references
        ]
        if not all(
            len(values) == len(hypotheses)
            for values in (simple, partial, complete, rate)
        ):
            raise RuntimeError("metric package returned an unexpected batch size")
        five = tuple(int(index) for index in self.chexbert.target_names_5_index)
        output = []
        for index in range(len(hypotheses)):
            hyp_labels = [int(value) for value in hypothesis_labels[index]]
            ref_labels = [int(value) for value in reference_labels[index]]
            exact_five = float(
                all(hyp_labels[position] == ref_labels[position] for position in five)
            )
            values = {
                "radgraph_simple": float(simple[index]),
                "radgraph_partial": float(partial[index]),
                "radgraph_complete": float(complete[index]),
                "ratescore": float(rate[index]),
                "chexbert_example_f1_14": example_binary_f1(
                    ref_labels, hyp_labels
                ),
                "chexbert_exact_match_5": exact_five,
                "chexbert_hypothesis_labels_14": hyp_labels,
                "chexbert_reference_labels_14": ref_labels,
            }
            if any(
                not math.isfinite(float(value))
                for key, value in values.items()
                if not key.endswith("_labels_14")
            ):
                raise FloatingPointError("metric package returned non-finite score")
            output.append(values)
        return output


def checkpoint_contract(cache: Path, metric_manifest: Path) -> dict[str, Any]:
    declared = json.loads(metric_manifest.read_text())["checkpoints"]
    artifacts = {
        "modern-radgraph-xl": cache / "radgraph/modern-radgraph-xl.tar.gz",
        # F1CheXbert resolves this path from XDG_CACHE_HOME at import time.
        "chexbert.pth": cache / "xdg/chexbert/chexbert.pth",
        "RaTE-NER-Deberta": cache / "rate-ner-deberta/model.safetensors",
        "BioLORD-2023-C": cache / "biolord-2023-c/model.safetensors",
    }
    hashes = {}
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        expected = declared[name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch for {name}")
        hashes[name] = actual
    tokenizer_snapshot = (
        cache
        / "hf_home/hub/models--bert-base-uncased/snapshots"
        / "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    tokenizer_hashes = {
        path.name: file_sha256(path)
        for path in sorted(tokenizer_snapshot.iterdir())
        if path.is_file()
    }
    if not {"vocab.txt", "tokenizer_config.json"} <= set(tokenizer_hashes):
        raise RuntimeError("incomplete local bert-base-uncased tokenizer")
    return {
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "sha256": hashes,
        "chexbert_tokenizer_snapshot": str(tokenizer_snapshot),
        "chexbert_tokenizer_sha256": tokenizer_hashes,
        "weights_sha256": stable_json_sha256(hashes),
    }


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("pair_sha256", ""))
        if len(key) != 64 or key in output:
            raise ValueError(f"invalid/duplicate resume row at line {line_number}")
        output[key] = row
    return output


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metric-manifest",
        type=Path,
        default=Path("docs/medheval_report_metric_manifest.json"),
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-directions", action="store_true")
    parser.add_argument("--min-direction-pairs", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_pairs < 0 or args.bootstrap_replicates <= 0:
        raise ValueError("batch-size must be positive and max-pairs nonnegative")
    if not args.input.is_file() or not args.metric_manifest.is_file():
        raise FileNotFoundError("input or metric manifest is missing")
    args.cache = args.cache.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(args.cache / "hf_home")
    os.environ["XDG_CACHE_HOME"] = str(args.cache / "xdg")

    pairs = load_pairs(args.input, args.max_pairs)
    checkpoints = checkpoint_contract(args.cache, args.metric_manifest)
    config = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "selected_pair_sha256": [row["pair_sha256"] for row in pairs],
        "metric_manifest": str(args.metric_manifest.resolve()),
        "metric_manifest_sha256": file_sha256(args.metric_manifest),
        "checkpoint_contract": checkpoints,
        "batch_size": args.batch_size,
        "max_pairs": args.max_pairs,
        "validate_directions": args.validate_directions,
        "min_direction_pairs": args.min_direction_pairs,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "code_sha256": file_sha256(Path(__file__)),
        "preprocessing": "strip_only; no section removal, truncation, or lowercasing",
        "device": "cpu",
        "network": "offline",
    }
    fingerprint = stable_json_sha256(config)
    state_path = args.output_dir / "run_manifest.json"
    records_path = args.output_dir / "records.jsonl"
    direction_path = args.output_dir / "direction_records.jsonl"
    aggregate_path = args.output_dir / "aggregate.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not args.resume:
            raise FileExistsError("output exists; pass --resume for identical run")
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
    elif any(path.exists() for path in (records_path, direction_path, aggregate_path)):
        raise RuntimeError("orphan output artifact without run manifest")
    else:
        state = {
            "version": VERSION,
            "fingerprint": fingerprint,
            "status": "in_progress",
            "config": config,
            "scorers": {
                "radgraph": importlib.metadata.version("radgraph"),
                "ratescore": importlib.metadata.version("RaTEScore"),
                "chexbert": importlib.metadata.version("f1chexbert"),
            },
            "n_input": len(pairs),
            "n_complete": 0,
        }
        atomic_json(state_path, state)

    scorer = ClinicalScorers(args.cache, args.batch_size)
    completed = load_completed(records_path)
    pending = [row for row in pairs if row["pair_sha256"] not in completed]
    for batch in batched(pending, args.batch_size):
        values = scorer.score(
            [row["model_answer"] for row in batch],
            [row["ground_truth"] for row in batch],
        )
        output = [
            {
                **row,
                "runner_fingerprint": fingerprint,
                "metrics": metric,
            }
            for row, metric in zip(batch, values)
        ]
        append_jsonl(records_path, output)
        completed.update({row["pair_sha256"]: row for row in output})
        state["n_complete"] = len(completed)
        atomic_json(state_path, state)

    ordered = [completed[row["pair_sha256"]] for row in pairs]
    direction_summary = None
    if args.validate_directions:
        direction_completed = load_completed(direction_path)
        direction_pending = [
            row for row in pairs if row["pair_sha256"] not in direction_completed
        ]
        for batch in batched(direction_pending, args.batch_size):
            contradictions = [
                contradiction_for(row["ground_truth"]) for row in batch
            ]
            matched = scorer.score(
                [row["ground_truth"] for row in batch],
                [row["ground_truth"] for row in batch],
            )
            contradicted = scorer.score(
                [value[0] for value in contradictions],
                [row["ground_truth"] for row in batch],
            )
            output = []
            for row, kind, good, bad in zip(
                batch, contradictions, matched, contradicted
            ):
                output.append(
                    {
                        "item_id": row["item_id"],
                        "pair_sha256": row["pair_sha256"],
                        "reference_type": kind[1],
                        "contradiction": kind[0],
                        "matched": {
                            key: value
                            for key, value in good.items()
                            if not key.endswith("_labels_14")
                        },
                        "contradicted": {
                            key: value
                            for key, value in bad.items()
                            if not key.endswith("_labels_14")
                        },
                        "runner_fingerprint": fingerprint,
                    }
                )
            append_jsonl(direction_path, output)
            direction_completed.update(
                {row["pair_sha256"]: row for row in output}
            )
        direction_ordered = [
            direction_completed[row["pair_sha256"]] for row in pairs
        ]
        direction_summary = summarize_directions(
            direction_ordered, args.min_direction_pairs
        )

    aggregate = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "input": str(args.input),
        "input_sha256": config["input_sha256"],
        "metric_manifest_sha256": config["metric_manifest_sha256"],
        "checkpoint_sha256": checkpoints["sha256"],
        "metrics": summarize_records(ordered),
        "bootstrap_ci95": bootstrap_records(ordered, args.bootstrap_replicates, args.bootstrap_seed),
        "direction_validation": direction_summary,
        "all_scores_finite": True,
    }
    atomic_json(aggregate_path, aggregate)
    state.update(
        {
            "status": "complete",
            "n_complete": len(ordered),
            "records_sha256": file_sha256(records_path),
            "aggregate_sha256": file_sha256(aggregate_path),
            "direction_records_sha256": (
                file_sha256(direction_path)
                if args.validate_directions
                else None
            ),
            "n_validation": (
                direction_summary["n_validation"]
                if direction_summary is not None
                else 0
            ),
            "direction_checks": (
                direction_summary["direction_checks"]
                if direction_summary is not None
                else {}
            ),
            "weights_sha256": checkpoints["weights_sha256"],
        }
    )
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "fingerprint": fingerprint,
                "metrics": aggregate["metrics"],
                "direction_validation": direction_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
