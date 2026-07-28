"""Generate full text with source-style subspace projection.

This is a probe for ANCHOR style nuisance removal. It does not choose among
Yes/No logits and does not tune thresholds on target labels. It applies a fixed
visual-feature projection during generation and evaluates generated text.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.analyze_style_nuisance_subspace import (
    DEFAULT_IUXRAY_IMAGE_ROOT,
    DEFAULT_IUXRAY_REPORT,
    DEFAULT_MIMIC_CE,
    DEFAULT_MIMIC_IMAGE_ROOT,
    DEFAULT_MIMIC_REPORT,
    fit_mean_subspace,
    read_json_or_jsonl,
    report_records,
    stable_key,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_surface import load_adapter

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "anchor-style-projection-generation-probe-v1"


def parse_binary_text(text: str) -> str | None:
    value = " ".join(str(text).strip().split())
    if not value:
        return None
    first = re.split(r"[.!?]", value, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z]+", first.lower())
    if any(tok in {"no", "not", "none", "without", "absent"} for tok in tokens):
        return "No."
    if any(tok in {"yes", "present", "evidence"} for tok in tokens):
        return "Yes."
    # RULE-compatible fallback: look at the whole short answer, but only text.
    tokens_all = re.findall(r"[A-Za-z]+", value.lower())
    if "yes" in tokens_all and "no" not in tokens_all:
        return "Yes."
    if "no" in tokens_all and "yes" not in tokens_all:
        return "No."
    return None


def normalize_binary(text: object) -> str:
    value = str(text).strip().lower()
    return "No." if value.startswith("no") else "Yes."


def rouge_l(prediction: str, reference: str) -> float:
    pred = re.findall(r"\w+", prediction.lower())
    ref = re.findall(r"\w+", reference.lower())
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for token in pred:
        curr = [0]
        for j, ref_token in enumerate(ref, start=1):
            curr.append(prev[j - 1] + 1 if token == ref_token else max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def ce_rows(path: Path, image_root: Path, max_samples: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_json_or_jsonl(path):
        image = str(row.get("image", "")).strip()
        question = str(row.get("question", "")).replace("<image>", "").strip()
        answer = row.get("answer")
        full = image_root / image
        if not image or not question or not full.exists():
            continue
        rows.append({
            "task": "ce",
            "id": str(row.get("question_id")),
            "image_path": str(full.resolve()),
            "question": question,
            "reference": normalize_binary(answer),
            "prompt": f"{question}\nAnswer the medical question in one concise sentence.",
        })
    rows.sort(key=lambda item: stable_key(item["id"], seed))
    return rows[:max_samples] if max_samples else rows


def report_task_rows(path: Path, image_root: Path, source: str, max_samples: int, seed: int) -> list[dict[str, Any]]:
    out = []
    for item in report_records(path, image_root, source, max_images=0, seed=seed):
        # Recover reference report by id from source JSON.
        pass
    source_rows = read_json_or_jsonl(path)
    by_id = {str(row.get("id")): row for row in source_rows}
    for item in report_records(path, image_root, source, max_images=0, seed=seed):
        ref = by_id.get(str(item["id"]), {}).get("report", "")
        if not ref:
            continue
        out.append({
            "task": "report",
            "source": source,
            "id": str(item["id"]),
            "image_path": item["image_path"],
            "reference": ref,
            "prompt": "Generate a concise radiology report for this chest X-ray.",
        })
    out.sort(key=lambda item: stable_key(f"{source}:{item['id']}", seed))
    return out[:max_samples] if max_samples else out


def load_subspace(prevalidation_dir: Path, source_domains: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = np.load(prevalidation_dir / "features.npz")["features"].astype(np.float64)
    meta = json.loads((prevalidation_dir / "features.meta.json").read_text())
    records = meta["records"]
    idx = [i for i, row in enumerate(records) if row["split"] == "source" and row["domain"] in source_domains]
    sub = fit_mean_subspace(features[idx], [records[i]["domain"] for i in idx])
    return sub["mean"].astype(np.float32), sub["basis"].astype(np.float32), {"domains": sub["domains"], "rank": sub["rank"], "singular_values": sub["singular_values"]}


@contextmanager
def projected_mm_projector(model, mean: np.ndarray, basis: np.ndarray, alpha: float):
    if alpha == 0:
        with nullcontext():
            yield
        return
    projector = model.get_model().mm_projector
    original_forward = projector.forward
    device = next(projector.parameters()).device
    dtype = next(projector.parameters()).dtype
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
    basis_t = torch.tensor(basis, device=device, dtype=torch.float32)

    def forward(features: torch.Tensor, *args, **kwargs):
        out = original_forward(features, *args, **kwargs)
        centered = out.float() - mean_t
        projected = (centered @ basis_t.T) @ basis_t
        return out - float(alpha) * projected.to(dtype=out.dtype)

    projector.forward = forward
    try:
        yield
    finally:
        projector.forward = original_forward


def decode_one(adapter, row: dict[str, Any], max_new_tokens: int) -> str:
    with Image.open(row["image_path"]) as handle:
        image = resize_image(handle.convert("RGB"), 384)
    return adapter.decode_ce([image], row["prompt"], max_new_tokens=max_new_tokens)[0]


def evaluate_row(row: dict[str, Any], text: str) -> dict[str, Any]:
    if row["task"] == "ce":
        parsed = parse_binary_text(text)
        return {
            "parsed": parsed,
            "correct": parsed == row["reference"] if parsed is not None else False,
            "parseable": parsed is not None,
        }
    return {
        "rouge_l": rouge_l(text, row["reference"]),
        "length_words": len(re.findall(r"\w+", text)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in sorted(set(row["task"] for row in rows)):
        subset = [row for row in rows if row["task"] == task]
        if task == "ce":
            out[task] = {
                "n": len(subset),
                "accuracy": float(np.mean([row["correct"] for row in subset])) if subset else None,
                "parse_rate": float(np.mean([row["parseable"] for row in subset])) if subset else None,
            }
        else:
            out[task] = {
                "n": len(subset),
                "rouge_l": float(np.mean([row["rouge_l"] for row in subset])) if subset else None,
                "avg_length_words": float(np.mean([row["length_words"] for row in subset])) if subset else None,
            }
            by_source = {}
            for source in sorted(set(row.get("source", "unknown") for row in subset)):
                part = [row for row in subset if row.get("source", "unknown") == source]
                by_source[source] = {
                    "n": len(part),
                    "rouge_l": float(np.mean([row["rouge_l"] for row in part])),
                    "avg_length_words": float(np.mean([row["length_words"] for row in part])),
                }
            out[task]["by_source"] = by_source
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prevalidation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="llava", choices=("llava",))
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--max-ce", type=int, default=64)
    parser.add_argument("--max-report-per-source", type=int, default=32)
    parser.add_argument("--ce-max-new-tokens", type=int, default=48)
    parser.add_argument("--report-max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    mean, basis, subspace_meta = load_subspace(args.prevalidation_dir, ("rule_iuxray", "slake_xray"))
    tasks = []
    tasks.extend(ce_rows(Path(DEFAULT_MIMIC_CE), Path(DEFAULT_MIMIC_IMAGE_ROOT), args.max_ce, args.seed))
    tasks.extend(report_task_rows(Path(DEFAULT_MIMIC_REPORT), Path(DEFAULT_MIMIC_IMAGE_ROOT), "mimic_report", args.max_report_per_source, args.seed))
    tasks.extend(report_task_rows(Path(DEFAULT_IUXRAY_REPORT), Path(DEFAULT_IUXRAY_IMAGE_ROOT), "iuxray_report", args.max_report_per_source, args.seed))

    adapter = load_adapter(args.model)
    details = []
    started = time.time()
    try:
        for alpha in args.alphas:
            with projected_mm_projector(adapter.model, mean, basis, alpha):
                for row in tqdm(tasks, desc=f"generate alpha={alpha}"):
                    max_tokens = args.ce_max_new_tokens if row["task"] == "ce" else args.report_max_new_tokens
                    text = decode_one(adapter, row, max_tokens)
                    metrics = evaluate_row(row, text)
                    details.append({
                        "version": VERSION,
                        "alpha": alpha,
                        **{k: row[k] for k in row if k not in {"reference"}},
                        "reference": row["reference"],
                        "text": text,
                        **metrics,
                    })
    finally:
        adapter.close()

    summary = {}
    for alpha in args.alphas:
        subset = [row for row in details if float(row["alpha"]) == float(alpha)]
        summary[str(alpha)] = aggregate(subset)
    payload = {
        "version": VERSION,
        "elapsed_sec": time.time() - started,
        "subspace": subspace_meta,
        "config": vars(args),
        "summary": summary,
        "notes": "All outputs are full generated text. CE scores use text parsing only; no yes/no logits are used as results.",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
