#!/usr/bin/env python3
"""Paired LET evaluation for unrestricted chest-radiograph reports.

The runner intentionally shares no CE parser or label interface.  Baseline and
LET use the same image, report prompt, conversation template, greedy decoding
budget, and standard ``model.generate`` backend.  The output schema is directly
compatible with the repository's lexical and clinical report evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import repair_truncated_jsonl_tail
from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.report_protocol import is_normal_template, report_prompt
from corrected_sgta.run_anchor_let_rule75 import (
    generate_layer_expert_standard,
    sha256_file,
)


VERSION = "anchor-let-report-v1"
CHEST_DATASETS = ("iuxray", "mimic")


def stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_records(
    manifest: Path, datasets: tuple[str, ...], per_dataset: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest.read_text())
    source = payload.get("records")
    if not isinstance(source, list):
        raise ValueError("manifest must contain a records list")
    selected: list[dict[str, Any]] = []
    counts = {name: 0 for name in datasets}
    for row in source:
        dataset = str(row.get("dataset", "")).lower()
        if dataset not in counts:
            continue
        if per_dataset and counts[dataset] >= per_dataset:
            continue
        image = Path(str(row.get("image", "")))
        reference = str(row.get("reference", row.get("answer", ""))).strip()
        if not image.is_file() or not reference:
            continue
        selected.append(dict(row))
        counts[dataset] += 1
    missing = {
        name: per_dataset - count
        for name, count in counts.items()
        if per_dataset and count < per_dataset
    }
    if missing:
        raise ValueError(f"insufficient valid report rows: {missing}")
    return payload, selected


def completed_rows(path: Path, fingerprint: str) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    repair_truncated_jsonl_tail(path)
    output: dict[tuple[str, str], dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("fingerprint") != fingerprint:
            raise ValueError(f"fingerprint mismatch at line {line_number}")
        key = (str(row["dataset"]), str(row["id"]))
        if key in output:
            raise ValueError(f"duplicate report row: {key}")
        output[key] = row
    return output


def build_payload(
    rows: list[dict[str, Any]], fingerprint: str, config: dict[str, Any]
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["dataset"], row["id"]))
    return {
        "version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "ground_truth_used_for_generation_or_selection": False,
        "records": [
            {
                "dataset": row["dataset"],
                "id": row["id"],
                "image": row["image"],
                "ground_truth": row["ground_truth"],
                "prompt": row["prompt"],
                "candidates": {
                    "baseline": row["baseline_text"],
                    "let": row["let_text"],
                },
                "normal_template": {
                    "baseline": row["baseline_normal_template"],
                    "let": row["let_normal_template"],
                },
                "token_count": {
                    "baseline": row["baseline_token_count"],
                    "let": row["let_token_count"],
                },
            }
            for row in ordered
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=CHEST_DATASETS, default=list(CHEST_DATASETS)
    )
    parser.add_argument("--per-dataset", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--layer", type=int, default=-12)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.per_dataset < 0 or args.max_new_tokens <= 0:
        raise ValueError("invalid sample or token budget")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    datasets = tuple(args.datasets)
    manifest_payload, records = load_records(
        args.manifest, datasets, args.per_dataset
    )
    config = {
        "version": VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_fingerprint": manifest_payload.get("fingerprint"),
        "datasets": datasets,
        "per_dataset": args.per_dataset,
        "model": "microsoft/llava-med-v1.5-mistral-7b",
        "conversation": "vicuna_v1",
        "prompt_mode": "official_zero_shot",
        "generation_backend": "standard_model_generate",
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "layer": args.layer,
        "alpha": args.alpha,
        "seed": args.seed,
        "primary_metric": "RadGraph F1 simple",
        "secondary_metrics": ["RaTEScore", "CheXbert", "ROUGE-L"],
        "code_sha256": sha256_file(Path(__file__)),
        "ce_parser_used": False,
        "label_logits_used": False,
    }
    fingerprint = stable_sha256(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "let_report.raw.jsonl"
    predictions_path = args.output_dir / "predictions.json"
    manifest_path = args.output_dir / "run_manifest.json"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(raw_path)
    done = completed_rows(raw_path, fingerprint) if args.resume else {}

    adapter = LlavaMedOEAdapter(conv_mode="vicuna_v1")
    output_rows = list(done.values())
    with raw_path.open("a" if args.resume else "w") as handle:
        for index, row in enumerate(tqdm(records, desc="LET report")):
            key = (str(row["dataset"]), str(row["id"]))
            if key in done:
                continue
            sample = {
                "dataset": row["dataset"],
                "task": "report_generation",
                "modality": "chest_radiograph",
            }
            prompt = report_prompt(sample, mode="official_zero_shot")
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            baseline = generate_layer_expert_standard(
                adapter,
                image,
                prompt,
                alpha=0.0,
                expert_layer=args.layer,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + index,
            )
            let = generate_layer_expert_standard(
                adapter,
                image,
                prompt,
                alpha=args.alpha,
                expert_layer=args.layer,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + index,
            )
            result = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "dataset": row["dataset"],
                "id": str(row["id"]),
                "image": str(row["image"]),
                "ground_truth": str(row.get("reference", row.get("answer"))),
                "prompt": prompt,
                "baseline_text": baseline.text,
                "let_text": let.text,
                "baseline_token_count": baseline.token_count,
                "let_token_count": let.token_count,
                "baseline_normal_template": is_normal_template(baseline.text),
                "let_normal_template": is_normal_template(let.text),
                "layer": args.layer,
                "alpha": args.alpha,
                "ground_truth_used_for_generation_or_selection": False,
                "uses_ce_parser": False,
                "uses_label_logits_for_prediction": False,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            output_rows.append(result)
            predictions_path.write_text(
                json.dumps(
                    build_payload(output_rows, fingerprint, config),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )

    payload = build_payload(output_rows, fingerprint, config)
    predictions_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "fingerprint": fingerprint,
                "status": "complete",
                "config": config,
                "n": len(output_rows),
                "raw": str(raw_path),
                "predictions": str(predictions_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n": len(output_rows),
                "predictions": str(predictions_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
