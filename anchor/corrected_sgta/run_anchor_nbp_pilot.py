#!/usr/bin/env python3
"""Run ANCHOR-NBP feasibility pilot.

This script generates complete sentences only.  It applies a single additive
normal-bundle proximal delta to all LLaVA visual tokens before greedy decoding.
No target labels are used for parameter selection or output decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageFile
from tqdm import tqdm

from corrected_sgta.analyze_style_nuisance_subspace import (
    DEFAULT_IUXRAY_REPORT,
    DEFAULT_IUXRAY_IMAGE_ROOT,
    DEFAULT_MIMIC_CE,
    DEFAULT_MIMIC_IMAGE_ROOT,
    DEFAULT_MIMIC_REPORT,
    read_json_or_jsonl,
    stable_key,
)
from corrected_sgta.evaluate_medheval_answers import parse_answer, rule_pope_prediction
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_surface import load_adapter
from corrected_sgta.nbp_geometry import (
    METHODS,
    NBPConfig,
    SourceBank,
    SourceBankRecord,
    compute_delta,
    l2_normalize,
    stable_json_sha256,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

RUN_VERSION = "anchor-nbp-pilot-runner-v1"
DEFAULT_SOURCE_SPECS = (
    "rule_iuxray:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/rule_iuxray.json",
    "slake_xray:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/slake_xray.json",
    "vqa_rad_train:/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/filtered_sources_assume_all/vqa_rad_train.json",
)
DEFAULT_OUT = Path("corrected_runs/final_anchor_nbp_pilot_v1")
REPORT_PROMPT = (
    "You are a professional radiologist. Generate a concise radiology report "
    "for this chest X-ray. Only include the report text."
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(str(row.get("run_key")))
    return done


def source_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    conv = row.get("conversations")
    if isinstance(conv, list) and len(conv) >= 2:
        question = str(conv[0].get("value", "")).replace("<image>", "").strip()
        answer = str(conv[1].get("value", "")).strip()
        return question, answer
    return str(row.get("question", "")).replace("<image>", "").strip(), str(
        row.get("answer", row.get("original_answer", ""))
    ).strip()


def question_family(question: str) -> str:
    text = question.lower()
    patterns = {
        "cardiomegaly": "cardiac_size",
        "heart": "cardiac_size",
        "lung": "lung",
        "opacity": "opacity",
        "atelectasis": "atelectasis",
        "effusion": "effusion",
        "pneumothorax": "pneumothorax",
        "edema": "edema",
        "consolidation": "consolidation",
        "nodule": "nodule_mass",
        "mass": "nodule_mass",
        "fracture": "bone",
    }
    for key, value in patterns.items():
        if key in text:
            return value
    return "general"


def parse_binary_label(text: object) -> str | None:
    parsed = parse_answer(text, answer_type="binary")
    if parsed.labels:
        return parsed.labels[0]
    return rule_pope_prediction(text)


def source_success_score(answer: str, original: str | None = None) -> float:
    if not original:
        return 1.0
    pred = parse_binary_label(answer)
    gt = parse_binary_label(original)
    if pred is not None and gt is not None:
        return 1.0 if pred == gt else 0.0
    return 1.0


def read_source_records(specs: Iterable[str], max_per_domain: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()
    for spec in specs:
        domain, path_text = spec.split(":", 1)
        rows = read_json_or_jsonl(Path(path_text))
        candidates = []
        for row in rows:
            image = row.get("image") or row.get("image_path")
            if isinstance(image, list):
                image = image[0] if image else None
            if not image or not Path(str(image)).exists():
                continue
            key = str(Path(str(image)).resolve())
            if key in seen:
                continue
            seen.add(key)
            question, answer = source_question_answer(row)
            candidates.append(
                {
                    "id": str(row.get("id") or row.get("source_id") or key),
                    "domain": domain,
                    "task": "ce",
                    "modality": "xray",
                    "view": "frontal",
                    "question_family": question_family(question),
                    "image_path": key,
                    "prompt": f"{question}\nAnswer the medical question in one concise sentence.",
                    "answer": answer,
                    "original_answer": row.get("original_answer"),
                }
            )
        candidates.sort(key=lambda item: stable_key(item["id"], seed))
        records.extend(candidates[:max_per_domain] if max_per_domain else candidates)
    return records


def read_report_source_records(max_reports: int, seed: int) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(Path(DEFAULT_IUXRAY_REPORT))
    out = []
    for row in rows:
        images = row.get("image_path")
        if isinstance(images, str):
            images = [images]
        if not images:
            continue
        full = Path(DEFAULT_IUXRAY_IMAGE_ROOT) / str(images[0])
        if not full.exists():
            continue
        out.append(
            {
                "id": f"iuxray-report-{row.get('id')}",
                "domain": "iuxray_report",
                "task": "oe",
                "modality": "xray",
                "view": "frontal",
                "question_family": "report",
                "image_path": str(full.resolve()),
                "prompt": REPORT_PROMPT,
                "answer": str(row.get("report", "")).strip(),
                "original_answer": str(row.get("report", "")).strip(),
            }
        )
    out.sort(key=lambda item: stable_key(item["id"], seed))
    return out[:max_reports] if max_reports else out


def mimic_ce_rows(max_patients: int, seed: int) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(Path(DEFAULT_MIMIC_CE))
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image = str(row.get("image", "")).strip()
        if not image:
            continue
        patient = image.split("/", 2)[1] if image.startswith("p") and "/" in image else str(row.get("patient_id", image))
        full = Path(DEFAULT_MIMIC_IMAGE_ROOT) / image
        if not full.exists():
            continue
        q = str(row.get("question", "")).replace("<image>", "").strip()
        by_patient.setdefault(patient, []).append(
            {
                "id": str(row.get("question_id")),
                "patient_id": patient,
                "task": "ce",
                "domain": "mimic_rule",
                "modality": "xray",
                "view": "frontal",
                "question_family": question_family(q),
                "image_path": str(full.resolve()),
                "prompt": f"{q}\nAnswer the medical question in one concise sentence.",
                "answer": str(row.get("answer", "")).strip(),
                "question": q,
            }
        )
    positive, negative = [], []
    for patient, patient_rows in by_patient.items():
        ordered = sorted(patient_rows, key=lambda r: stable_key(r["id"], seed))
        yes = [r for r in ordered if parse_binary_label(r["answer"]) == "yes"]
        no = [r for r in ordered if parse_binary_label(r["answer"]) == "no"]
        if yes:
            positive.append((patient, yes[0]))
        if no:
            negative.append((patient, no[0]))
    positive.sort(key=lambda item: stable_key(item[0] + ":yes", seed))
    negative.sort(key=lambda item: stable_key(item[0] + ":no", seed))
    out, used = [], set()
    target_each = (max_patients // 2) if max_patients else max(len(positive), len(negative))
    for bucket in (positive[:target_each], negative[:target_each]):
        for patient, row in bucket:
            if patient in used:
                continue
            used.add(patient)
            out.append(row)
    if max_patients and len(out) < max_patients:
        leftovers = positive[target_each:] + negative[target_each:]
        leftovers.sort(key=lambda item: stable_key(item[0] + ":extra", seed))
        for patient, row in leftovers:
            if patient in used:
                continue
            used.add(patient)
            out.append(row)
            if len(out) >= max_patients:
                break
    out.sort(key=lambda r: stable_key(r["id"], seed))
    return out[:max_patients] if max_patients else out


def mimic_report_rows(max_reports: int, seed: int) -> list[dict[str, Any]]:
    rows = read_json_or_jsonl(Path(DEFAULT_MIMIC_REPORT))
    out = []
    for row in rows:
        images = row.get("image_path")
        if isinstance(images, str):
            images = [images]
        if not images:
            continue
        full = Path(DEFAULT_MIMIC_IMAGE_ROOT) / str(images[0])
        if not full.exists():
            continue
        out.append(
            {
                "id": str(row.get("id")),
                "patient_id": str(row.get("subject_id", row.get("id"))),
                "task": "oe",
                "domain": "mimic_report",
                "modality": "xray",
                "view": "frontal",
                "question_family": "report",
                "image_path": str(full.resolve()),
                "prompt": REPORT_PROMPT,
                "answer": str(row.get("report", "")).strip(),
            }
        )
    out.sort(key=lambda item: stable_key(item["id"], seed))
    return out[:max_reports] if max_reports else out


def scanner_shift(image: Image.Image, kind: str) -> Image.Image:
    if kind == "clean":
        return image
    if kind == "gamma_window":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.18)
    if kind == "blur_resample":
        small = image.resize((max(32, image.width // 2), max(32, image.height // 2)))
        return small.resize(image.size).filter(ImageFilter.GaussianBlur(radius=0.35))
    if kind == "compression_noise":
        arr = np.asarray(image).astype(np.float32)
        rng = np.random.default_rng(1337)
        arr = np.clip(arr + rng.normal(0, 3.0, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"unknown scanner shift: {kind}")


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


def evaluate_output(row: dict[str, Any], text: str) -> dict[str, Any]:
    if row["task"] == "ce":
        gt = parse_binary_label(row["answer"])
        pred = parse_binary_label(text)
        return {
            "gt": gt,
            "parsed": pred,
            "parseable": pred is not None,
            "correct": bool(pred is not None and gt is not None and pred == gt),
            "positive_gt": gt == "yes",
        }
    return {
        "rouge_l": rouge_l(text, row["answer"]),
        "length_words": len(re.findall(r"\w+", text)),
        "radgraph_f1": None,
        "ratescore": None,
        "chexbert_f1": None,
    }


def image_from_row(row: dict[str, Any], shift: str) -> Image.Image:
    with Image.open(row["image_path"]) as handle:
        image = resize_image(handle.convert("RGB"), 384)
    return scanner_shift(image, shift)


@contextmanager
def llava_nbp_projector(model: Any, bank: SourceBank, row: dict[str, Any], method: str, config: NBPConfig, weights_mode: str):
    if method == "greedy":
        with nullcontext():
            yield None
        return
    original = model.encode_images
    holder: dict[str, Any] = {}

    def encode_images_with_nbp(images):
        out = original(images)
        pooled = out.float().mean(dim=1)[0].detach().cpu().numpy().astype(np.float64)
        delta_meta = compute_delta(bank, pooled, row, method=method, config=config, weights_mode=weights_mode)
        delta = torch.tensor(delta_meta.delta, device=out.device, dtype=out.dtype).view(1, 1, -1)
        patch = delta_meta.patch
        holder["geometry"] = {
            "method": delta_meta.method,
            "delta_norm": delta_meta.delta_norm,
            "e_perp": delta_meta.e_perp,
            "tangent_energy": delta_meta.tangent_energy,
            "patch": None if patch is None else {
                "condition_level": patch.condition_level,
                "neighbor_ids": patch.neighbor_ids[:8],
                "density": patch.density,
                "k": len(patch.neighbor_ids),
            },
        }
        return out + delta

    model.encode_images = encode_images_with_nbp
    try:
        yield holder
    finally:
        model.encode_images = original


def decode_one(adapter: Any, bank: SourceBank, row: dict[str, Any], method: str, config: NBPConfig, weights_mode: str, max_new_tokens: int, shift: str) -> tuple[str, dict[str, Any] | None]:
    image = image_from_row(row, shift)
    if adapter.__class__.__name__.lower().startswith("llava"):
        with llava_nbp_projector(adapter.model, bank, row, method, config, weights_mode) as holder:
            text = adapter.decode_ce([image], row["prompt"], max_new_tokens=max_new_tokens)[0]
            geom = None if holder is None else holder.get("geometry")
        return text, geom
    raise NotImplementedError("NBP pilot currently supports LLaVA-Med only")


def build_bank(adapter: Any, source_rows: list[dict[str, Any]], output: Path, batch_size: int) -> SourceBank:
    if output.exists():
        return SourceBank.from_json(json.loads(output.read_text()))
    records: list[SourceBankRecord] = []
    for start in tqdm(range(0, len(source_rows), batch_size), desc="build-source-bank"):
        batch = source_rows[start : start + batch_size]
        images = [image_from_row(row, "clean") for row in batch]
        feats = adapter.visual_features(images) if hasattr(adapter, "visual_features") else None
        if feats is None:
            # Surface adapter lacks visual_features; call model directly for LLaVA.
            from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter

            raise RuntimeError("adapter must expose visual_features; use alignment adapter compatible class")
        for row, z in zip(batch, feats):
            score = source_success_score(row.get("answer", ""), row.get("original_answer"))
            weight = 0.1 + 0.9 * float(score)
            records.append(
                SourceBankRecord(
                    record_id=str(row["id"]),
                    domain=str(row["domain"]),
                    task=str(row["task"]),
                    modality=str(row["modality"]),
                    view=str(row["view"]),
                    question_family=str(row["question_family"]),
                    image_path=str(row["image_path"]),
                    z=np.asarray(z, dtype=np.float32).astype(float).tolist(),
                    q=l2_normalize(np.asarray(z, dtype=np.float64)).astype(float).tolist(),
                    success_score=float(score),
                    reliability_weight=float(weight),
                    answer=row.get("answer"),
                    prompt=row.get("prompt"),
                )
            )
    bank = SourceBank(records)
    atomic_json(output, bank.to_json(config={"run_version": RUN_VERSION}))
    return bank


def config_grid() -> list[NBPConfig]:
    out = []
    for k in (32, 64):
        for rank in (8, 16, 32):
            if rank >= k:
                continue
            for alpha in (0.10, 0.25, 0.50):
                out.append(NBPConfig(k=k, rank=rank, alpha=alpha))
    return out


def select_config(adapter: Any, bank: SourceBank, val_rows: list[dict[str, Any]], output: Path, max_new_tokens: int) -> NBPConfig:
    if output.exists():
        payload = json.loads(output.read_text())
        return NBPConfig(**payload["selected_config"])
    summaries = []
    # Keep source-val cheap: compare greedy/local_isotropic/nbp on synthetic shift.
    for cfg in config_grid():
        rows = []
        for method in ("greedy", "nbp"):
            for row in val_rows:
                text, _ = decode_one(adapter, bank, row, method, cfg, "reliability", max_new_tokens, "gamma_window")
                metric = evaluate_output(row, text)
                rows.append({"method": method, **metric})
        clean_rows = []
        for row in val_rows:
            text, _ = decode_one(adapter, bank, row, "nbp", cfg, "reliability", max_new_tokens, "clean")
            clean_rows.append(evaluate_output(row, text))
        shift_acc = float(np.mean([r["correct"] for r in rows if r["method"] == "nbp"]))
        greedy_shift = float(np.mean([r["correct"] for r in rows if r["method"] == "greedy"]))
        clean_acc = float(np.mean([r["correct"] for r in clean_rows]))
        summaries.append({"config": cfg.__dict__, "shift_acc": shift_acc, "greedy_shift_acc": greedy_shift, "clean_acc": clean_acc})
    eligible = [s for s in summaries if s["clean_acc"] >= max(0.0, max(x["clean_acc"] for x in summaries if x["config"]["alpha"] == 0.10) - 0.01)]
    selected = max(eligible or summaries, key=lambda s: (s["shift_acc"], -s["config"]["alpha"]))
    payload = {"version": RUN_VERSION, "summaries": summaries, "selected_config": selected["config"]}
    atomic_json(output, payload)
    return NBPConfig(**selected["config"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", choices=("llava",), default="llava")
    p.add_argument("--source-spec", action="append", default=list(DEFAULT_SOURCE_SPECS))
    p.add_argument("--max-source-per-domain", type=int, default=128)
    p.add_argument("--max-report-source", type=int, default=128)
    p.add_argument("--max-source-val", type=int, default=24)
    p.add_argument("--max-ce-patients", type=int, default=256)
    p.add_argument("--max-oe-reports", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ce-max-new-tokens", type=int, default=48)
    p.add_argument("--oe-max-new-tokens", type=int, default=96)
    p.add_argument("--methods", nargs="*", default=["greedy", "local_isotropic", "nn_interpolation", "tangent_matched", "random_matched", "nbp", "global_pca"])
    p.add_argument("--random-seeds", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--weights-modes", nargs="*", default=["reliability"], choices=("reliability", "unweighted", "shuffled", "low_reliability"))
    p.add_argument("--shifts", nargs="*", default=["clean", "gamma_window", "blur_resample", "compression_noise"])
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-source-val", action="store_true")
    p.add_argument("--fixed-k", type=int, default=32)
    p.add_argument("--fixed-rank", type=int, default=8)
    p.add_argument("--fixed-alpha", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from corrected_sgta.models_alignment import load_alignment_adapter

    adapter = load_alignment_adapter(args.model)
    try:
        source_rows = read_source_records(args.source_spec, args.max_source_per_domain, args.seed)
        source_rows.extend(read_report_source_records(args.max_report_source, args.seed))
        source_rows.sort(key=lambda r: stable_key(r["id"], args.seed))
        ce_source = [row for row in source_rows if row.get("task") == "ce"]
        val_ids = {row["id"] for row in ce_source[: args.max_source_val]}
        source_val = [row for row in source_rows if row["id"] in val_ids]
        source_train = [row for row in source_rows if row["id"] not in val_ids]
        bank = build_bank(adapter, source_train, args.output_dir / "source_bank.json", args.batch_size)
        if args.skip_source_val:
            config = NBPConfig(k=args.fixed_k, rank=args.fixed_rank, alpha=args.fixed_alpha, seed=args.seed)
            atomic_json(args.output_dir / "selected_config.json", {"version": RUN_VERSION, "selected_config": config.__dict__, "selection": "fixed_skip_source_val"})
        else:
            config = select_config(adapter, bank, source_val, args.output_dir / "selected_config.json", args.ce_max_new_tokens)

        ce_rows = mimic_ce_rows(args.max_ce_patients, args.seed)
        oe_rows = mimic_report_rows(args.max_oe_reports, args.seed)
        tasks = [("ce", ce_rows, args.ce_max_new_tokens), ("oe", oe_rows, args.oe_max_new_tokens)]
        raw_path = args.output_dir / "raw_outputs.jsonl"
        done = load_done(raw_path) if args.resume else set()
        for task, rows, max_tokens in tasks:
            for shift in args.shifts:
                if task == "oe" and shift != "clean":
                    continue
                for method in args.methods:
                    seeds = args.random_seeds if method == "random_matched" else [0]
                    method_weight_modes = args.weights_modes if method == "nbp" else ["reliability"]
                    for weights_mode in method_weight_modes:
                        for random_seed in seeds:
                            cfg = NBPConfig(k=config.k, rank=config.rank, alpha=config.alpha, seed=args.seed, random_seed=random_seed)
                            for row in tqdm(rows, desc=f"{task}:{shift}:{method}:{weights_mode}:seed{random_seed}"):
                                run_key = stable_json_sha256({"task": task, "shift": shift, "method": method, "weights_mode": weights_mode, "seed": random_seed, "id": row["id"], "cfg": cfg.__dict__})
                                if run_key in done:
                                    continue
                                try:
                                    text, geom = decode_one(adapter, bank, row, method, cfg, weights_mode, max_tokens, shift)
                                    metric = evaluate_output(row, text)
                                    append_jsonl(raw_path, {
                                        "version": RUN_VERSION,
                                        "run_key": run_key,
                                        "task": task,
                                        "split": "S-test" if shift == "clean" else "Shift-D1",
                                        "shift": shift,
                                        "method": method,
                                        "weights_mode": weights_mode,
                                        "random_seed": random_seed,
                                        "config": cfg.__dict__,
                                        "id": row["id"],
                                        "patient_id": row.get("patient_id"),
                                        "image_path": row["image_path"],
                                        "prompt": row["prompt"],
                                        "reference": row["answer"],
                                        "text": text,
                                        "geometry": geom,
                                        **metric,
                                    })
                                    done.add(run_key)
                                except Exception as exc:
                                    append_jsonl(raw_path, {
                                        "version": RUN_VERSION,
                                        "run_key": run_key,
                                        "task": task,
                                        "split": "S-test" if shift == "clean" else "Shift-D1",
                                        "shift": shift,
                                        "method": method,
                                        "weights_mode": weights_mode,
                                        "random_seed": random_seed,
                                        "config": cfg.__dict__,
                                        "id": row.get("id"),
                                        "error": repr(exc),
                                    })
                                    done.add(run_key)
        manifest = {
            "version": RUN_VERSION,
            "elapsed_sec": time.time() - started,
            "source_train_n": len(source_train),
            "source_val_n": len(source_val),
            "ce_n": len(ce_rows),
            "oe_n": len(oe_rows),
            "config": vars(args),
            "selected_config": config.__dict__,
            "raw_outputs": str(raw_path),
            "notes": "Full-sentence generation; target labels are used only for evaluation.",
        }
        atomic_json(args.output_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
