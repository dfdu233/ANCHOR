#!/usr/bin/env python3
"""Sanity-audit open-ended/report generation before using it as evidence.

This runner is intentionally separate from the main OE/ConfGen pipelines.  It
answers a narrow question: does a report-generation baseline actually condition
on the image, or is it collapsing to a short normal-template prior?

Two modes are supported:

* ``--analyze-existing RAW``: CPU-only audit of an existing JSONL/JSON output.
* ``--run-generation``: small GPU audit over real/null/shuffled images across
  LLaVA-Med conversation modes and prompt templates.

The script writes a raw JSONL plus a compact summary with enough evidence to
decide whether OE/report results are admissible for a paper claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from corrected_sgta.oe_metrics import rouge_l, token_f1


VERSION = "oe-sanity-audit-v1"
DEFAULT_OUTPUT = Path("corrected_runs/oe_sanity_audit_v1")
DEFAULT_MANIFESTS = (
    Path("corrected_runs/final_anchor_riemann_gate_v1/manifests/mimic_report_oe.json"),
    Path("data/mmedrag/test/report/mimic_test.json"),
)
DEFAULT_IMAGE_ROOTS = (
    Path("data/medheval/images"),
    Path("/root/autodl-tmp/MedHEval/images"),
    Path("/root/autodl-tmp/datasets/mimic-cxr-jpg/2.1.0/files"),
    Path("/root/autodl-tmp/MIMIC-CXR-JPG/files"),
)

NORMAL_PATTERNS = (
    r"\bnormal\b",
    r"\bunremarkable\b",
    r"\bno significant abnormal",
    r"\bno acute (?:cardiopulmonary )?(?:abnormalit(?:y|ies)|disease|process|finding)",
    r"\blungs? (?:are|is) clear\b",
)
ABNORMAL_PATTERNS = (
    r"\bpneumonia\b",
    r"\bpneumothorax\b",
    r"\bcardiomegaly\b",
    r"\bpleural effusion\b",
    r"\beffusion\b",
    r"\bedema\b",
    r"\bconsolidation\b",
    r"\bopacity\b",
    r"\batelectasis\b",
    r"\bfracture\b",
    r"\bvascular congestion\b",
    r"\bmass\b",
    r"\bnodule\b",
)


def stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", str(text)))


def has_any(patterns: Iterable[str], text: str) -> bool:
    lowered = norm_text(text).lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def is_normal_template(text: str) -> bool:
    lowered = norm_text(text).lower()
    normal = has_any(NORMAL_PATTERNS, lowered)
    abnormal = has_any(ABNORMAL_PATTERNS, lowered)
    short = word_count(lowered) <= 35
    canonical = (
        "appears to be normal" in lowered
        or "no significant abnormalities detected" in lowered
        or "no acute cardiopulmonary" in lowered
    )
    return bool((canonical and short) or (normal and not abnormal and short))


def abnormal_finding(text: str) -> bool:
    lowered = norm_text(text).lower()
    return has_any(ABNORMAL_PATTERNS, lowered)


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def iter_existing_predictions(payload: Any) -> list[dict[str, Any]]:
    """Normalize several existing OE raw schemas into evaluation rows."""
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        records = payload
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        reference = (
            record.get("reference")
            or record.get("ground_truth")
            or record.get("answer")
            or ""
        )
        item_id = str(record.get("id", record.get("qid", index)))
        candidates = record.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                method = candidate.get("acquisition") or candidate.get("candidate_id") or "candidate"
                if method == "greedy" or candidate.get("candidate_id") == "candidate-0":
                    rows.append(
                        {
                            "id": item_id,
                            "condition": "existing",
                            "conv_mode": record.get("conv_mode", "unknown"),
                            "prompt_mode": "unknown",
                            "view": "real",
                            "method": "greedy",
                            "text": str(candidate.get("text", "")),
                            "reference": str(reference),
                        }
                    )
                    break
        elif isinstance(candidates, dict):
            for method, text in candidates.items():
                rows.append(
                    {
                        "id": item_id,
                        "condition": "existing",
                        "conv_mode": record.get("conv_mode", "unknown"),
                        "prompt_mode": "unknown",
                        "view": "real",
                        "method": str(method),
                        "text": str(text),
                        "reference": str(reference),
                    }
                )
        else:
            text = record.get("model_answer") or record.get("prediction") or record.get("text")
            if text is not None:
                rows.append(
                    {
                        "id": item_id,
                        "condition": "existing",
                        "conv_mode": record.get("conv_mode", "unknown"),
                        "prompt_mode": "unknown",
                        "view": "real",
                        "method": "prediction",
                        "text": str(text),
                        "reference": str(reference),
                    }
                )
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json_or_jsonl(path)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    output: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        image = row.get("image")
        if image is None and row.get("image_path"):
            image_path = row["image_path"]
            image = image_path[0] if isinstance(image_path, list) else image_path
        reference = row.get("answer") or row.get("report") or row.get("reference")
        if not image or not reference:
            continue
        dataset = row.get("dataset") or row.get("domain") or "mimic"
        output.append(
            {
                "id": str(row.get("id", row.get("source_id", index))),
                "patient_id": str(row.get("patient_id", row.get("subject_id", row.get("id", index)))),
                "dataset": str(dataset),
                "image": str(image),
                "reference": str(reference),
                "prompt": str(row.get("prompt", "")),
                "reference_abnormal": abnormal_finding(str(reference)),
                "reference_normal_template": is_normal_template(str(reference)),
            }
        )
    return output


def resolve_image(path_value: str, roots: Iterable[Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / path_value
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_value)


def prompt_for(sample: dict[str, Any], mode: str) -> str:
    if mode == "manifest" and sample.get("prompt"):
        return str(sample["prompt"]).replace("<image>", "").strip()
    dataset = sample.get("dataset", "mimic").lower()
    role = "ophthalmologist" if "harvard" in dataset else "radiologist"
    image_name = "fundus image" if "harvard" in dataset else "chest X-ray image"
    if mode in {"mmedrag", "official"}:
        return (
            f"You are a professional {role}. You are provided with a {image_name}. "
            "Please generate a report based on the image. "
            "Please only include the content of the report in your response."
        )
    if mode == "structured":
        return (
            f"You are a professional {role}. You are provided with a {image_name}. "
            "Write a concise radiology report with two sections:\n"
            "Findings:\n"
            "Impression:\n"
            "Only describe findings visible in the image."
        )
    if mode == "abnormality_focused":
        return (
            f"You are a professional {role}. Carefully inspect the {image_name}. "
            "Describe any visible abnormal findings, support devices, and relevant normal negatives. "
            "Do not answer only 'normal' unless the image truly has no abnormal findings."
        )
    raise ValueError(f"unknown prompt mode: {mode}")


def make_null_image(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    mean = tuple(int(x) for x in arr.reshape(-1, 3).mean(axis=0))
    return Image.new("RGB", rgb.size, mean)


def make_shuffled_image(image: Image.Image, seed: int) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8).copy()
    flat = arr.reshape(-1, arr.shape[-1])
    rng = np.random.default_rng(seed)
    order = rng.permutation(flat.shape[0])
    shuffled = flat[order].reshape(arr.shape)
    return Image.fromarray(shuffled, mode="RGB")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("method", "unknown")),
                str(row.get("conv_mode", "unknown")),
                str(row.get("prompt_mode", "unknown")),
                str(row.get("view", "unknown")),
            )
        ].append(row)
    summaries: dict[str, Any] = {}
    for key, values in sorted(groups.items()):
        method, conv_mode, prompt_mode, view = key
        texts = [norm_text(row.get("text", "")) for row in values]
        refs = [norm_text(row.get("reference", "")) for row in values]
        unique = len(set(texts))
        exact_counter = Counter(texts)
        metrics = []
        for text, ref in zip(texts, refs):
            if ref:
                metrics.append(
                    {
                        "rouge_l": rouge_l(text, ref),
                        "token_f1": token_f1(text, ref),
                    }
                )
        normal_template_rate = float(np.mean([is_normal_template(text) for text in texts])) if texts else 0.0
        abnormal_rate = float(np.mean([abnormal_finding(text) for text in texts])) if texts else 0.0
        summary_key = "|".join(key)
        summaries[summary_key] = {
            "method": method,
            "conv_mode": conv_mode,
            "prompt_mode": prompt_mode,
            "view": view,
            "n": len(values),
            "unique_output_count": unique,
            "unique_output_rate": unique / len(values) if values else 0.0,
            "mean_words": float(np.mean([word_count(text) for text in texts])) if texts else 0.0,
            "normal_template_rate": normal_template_rate,
            "abnormal_finding_rate": abnormal_rate,
            "rouge_l": float(np.mean([m["rouge_l"] for m in metrics])) if metrics else None,
            "token_f1": float(np.mean([m["token_f1"] for m in metrics])) if metrics else None,
            "top_outputs": [
                {"count": count, "text": text[:240]} for text, count in exact_counter.most_common(8)
            ],
        }
    # Pairwise real-vs-null/shuffled differences for generated audit rows.
    pairwise: dict[str, Any] = {}
    by_signature: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        sig = (
            str(row.get("id")),
            str(row.get("method", "unknown")),
            str(row.get("conv_mode", "unknown")),
            str(row.get("prompt_mode", "unknown")),
        )
        by_signature[sig][str(row.get("view", "unknown"))] = norm_text(row.get("text", ""))
    for view in ("null", "shuffled"):
        comparable = [views for views in by_signature.values() if "real" in views and view in views]
        if comparable:
            pairwise[f"real_vs_{view}"] = {
                "n": len(comparable),
                "exact_same_rate": float(np.mean([v["real"] == v[view] for v in comparable])),
                "mean_token_f1_between_outputs": float(
                    np.mean([token_f1(v["real"], v[view]) for v in comparable])
                ),
            }
    reference_stats = {
        "n": len({str(row.get("id")) for row in rows}),
        "mean_reference_words": float(
            np.mean([word_count(row.get("reference", "")) for row in rows if row.get("reference")])
        )
        if any(row.get("reference") for row in rows)
        else 0.0,
        "reference_abnormal_rate": float(
            np.mean([abnormal_finding(row.get("reference", "")) for row in rows if row.get("reference")])
        )
        if any(row.get("reference") for row in rows)
        else 0.0,
        "reference_normal_template_rate": float(
            np.mean([is_normal_template(row.get("reference", "")) for row in rows if row.get("reference")])
        )
        if any(row.get("reference") for row in rows)
        else 0.0,
    }
    invalid_reasons = []
    for item in summaries.values():
        if item["view"] == "real" and item["normal_template_rate"] > 0.90:
            invalid_reasons.append(
                f"{item['method']}:{item['conv_mode']}:{item['prompt_mode']} has normal_template_rate={item['normal_template_rate']:.3f}"
            )
        if item["view"] == "real" and item["unique_output_rate"] < 0.25 and item["n"] >= 8:
            invalid_reasons.append(
                f"{item['method']}:{item['conv_mode']}:{item['prompt_mode']} has low unique_output_rate={item['unique_output_rate']:.3f}"
            )
    if pairwise:
        for name, item in pairwise.items():
            if item["exact_same_rate"] > 0.75:
                invalid_reasons.append(f"{name} exact_same_rate={item['exact_same_rate']:.3f}")
    return {
        "version": VERSION,
        "n_rows": len(rows),
        "reference_stats": reference_stats,
        "summaries": summaries,
        "pairwise_image_dependency": pairwise,
        "admissible_for_report_generation_claim": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "notes": [
            "RadGraph/RaTEScore/CheXbert are not computed here; export generated real-view pairs and run evaluate_medheval_report_clinical.py for clinical metrics.",
            "ROUGE-L/token-F1 are sanity metrics only and must not be treated as clinical factuality.",
        ],
    }


def select_samples(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    abnormal = [row for row in records if row["reference_abnormal"]]
    normal = [row for row in records if not row["reference_abnormal"]]
    rng = random.Random(seed)
    rng.shuffle(abnormal)
    rng.shuffle(normal)
    if count <= 0:
        count = len(records)
    target_abnormal = min(len(abnormal), max(count // 2, count - min(len(normal), count // 2)))
    selected = abnormal[:target_abnormal] + normal[: max(0, count - target_abnormal)]
    if len(selected) < count:
        selected.extend([row for row in abnormal[target_abnormal:] if row not in selected][: count - len(selected)])
    selected = selected[:count]
    selected.sort(key=lambda row: stable_sha256({"seed": seed, "id": row["id"]}))
    return selected


def run_generation(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    from corrected_sgta.models_oe import LlavaMedOEAdapter

    manifest_path = next((path for path in args.manifest if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"none of the manifest paths exists: {args.manifest}")
    records = select_samples(load_manifest(manifest_path), args.max_samples, args.seed)
    roots = [Path(item) for item in args.image_root]
    raw_path = output_dir / "generation.raw.jsonl"
    rows: list[dict[str, Any]] = []
    config = {
        "version": VERSION,
        "mode": "generation",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "max_samples": args.max_samples,
        "conv_modes": args.conv_mode,
        "prompt_modes": args.prompt_mode,
        "views": args.view,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    fingerprint = stable_sha256(config)
    with raw_path.open("w") as handle:
        for conv_mode in args.conv_mode:
            adapter = LlavaMedOEAdapter(conv_mode=conv_mode)
            try:
                for sample_index, sample in enumerate(records):
                    image_path = resolve_image(sample["image"], roots)
                    with Image.open(image_path) as image_handle:
                        image = image_handle.convert("RGB")
                    views = {
                        "real": image,
                        "null": make_null_image(image),
                        "shuffled": make_shuffled_image(image, args.seed + sample_index),
                    }
                    for prompt_mode in args.prompt_mode:
                        prompt = prompt_for(sample, prompt_mode)
                        for view_name in args.view:
                            generation = adapter._generate_once(
                                views[view_name],
                                prompt,
                                1,
                                False,
                                1.0,
                                1.0,
                                args.max_new_tokens,
                                args.seed,
                            )[0]
                            row = {
                                "version": VERSION,
                                "fingerprint": fingerprint,
                                "id": sample["id"],
                                "patient_id": sample["patient_id"],
                                "dataset": sample["dataset"],
                                "image": str(image_path),
                                "image_sha256": file_sha256(image_path),
                                "reference": sample["reference"],
                                "reference_abnormal": sample["reference_abnormal"],
                                "method": "greedy",
                                "model": "llava-med-v1.5-mistral-7b",
                                "conv_mode": conv_mode,
                                "prompt_mode": prompt_mode,
                                "prompt": prompt,
                                "view": view_name,
                                "text": generation.text,
                                "token_count": generation.token_count,
                                "uncertainty": generation.uncertainty if math.isfinite(generation.uncertainty) else None,
                                "normal_template": is_normal_template(generation.text),
                                "abnormal_finding": abnormal_finding(generation.text),
                                "ground_truth_used_for_generation_or_selection": False,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                            rows.append(row)
            finally:
                adapter.close()
    (output_dir / "generation_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return rows


def export_clinical_pairs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    pair_path = output_dir / "clinical_pairs_real_view.jsonl"
    with pair_path.open("w") as handle:
        for row in rows:
            if row.get("view") != "real":
                continue
            handle.write(
                json.dumps(
                    {
                        "item_id": f"{row.get('id')}|{row.get('conv_mode')}|{row.get('prompt_mode')}",
                        "ground_truth": row.get("reference", ""),
                        "model_answer": row.get("text", ""),
                        "metadata": {
                            "conv_mode": row.get("conv_mode"),
                            "prompt_mode": row.get("prompt_mode"),
                            "method": row.get("method"),
                            "fingerprint": row.get("fingerprint"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analyze-existing", type=Path, action="append", default=[])
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--manifest", type=Path, nargs="*", default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--image-root", type=Path, nargs="*", default=list(DEFAULT_IMAGE_ROOTS))
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--conv-mode", nargs="*", default=["mistral_instruct", "vicuna_v1"])
    parser.add_argument(
        "--prompt-mode",
        nargs="*",
        default=["manifest", "mmedrag", "structured", "abnormality_focused"],
    )
    parser.add_argument("--view", nargs="*", default=["real", "null", "shuffled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    existing_payloads = []
    for path in args.analyze_existing:
        payload = load_json_or_jsonl(path)
        rows = iter_existing_predictions(payload)
        for row in rows:
            row["source_file"] = str(path)
        all_rows.extend(rows)
        existing_payloads.append({"path": str(path), "sha256": file_sha256(path), "rows": len(rows)})
    if args.run_generation:
        all_rows.extend(run_generation(args, args.output_dir))
    if not all_rows:
        raise SystemExit("nothing to audit: provide --analyze-existing and/or --run-generation")
    summary = summarize_rows(all_rows)
    summary["existing_inputs"] = existing_payloads
    summary["command"] = " ".join(os.sys.argv)
    summary["fingerprint"] = stable_sha256(
        {
            "version": VERSION,
            "existing": existing_payloads,
            "n_rows": len(all_rows),
            "command": summary["command"],
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "audit_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
    )
    export_clinical_pairs(all_rows, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
