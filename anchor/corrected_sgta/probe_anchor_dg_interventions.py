"""Data-driven source-only intervention gate for ANCHOR-DG."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from corrected_sgta.anchor_dg import (
    counterfactual_view, edge_correlation, gold_token_log_probabilities,
    load_style_bank, stable_sha256, standardized_image,
)
from corrected_sgta.filter_anchor_dg_chest_sources import load_biomedclip
from corrected_sgta.infer_rule_dg_adapter import decode
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import (
    build_teacher_forcing, rule_label, rule_no_reference_prompt, sequence_forward,
)
from corrected_sgta.train_rule_source_group_adapter import normalize_source_rows, parse_named_paths

VERSION = "anchor-dg-intervention-gate-v2"
PROJECTION = "32-bin-radial-log-spectrum+32-bin-intensity-histogram"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data-driven source-only ANCHOR-DG transform gate.")
    parser.add_argument("--source-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--style-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rho", action="append", type=float, default=[])
    parser.add_argument("--beta", action="append", type=float, default=[])
    parser.add_argument("--max-vlm-candidates", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--biomedclip-root", type=Path, default=Path("/root/autodl-tmp/BiomedCLIP"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paired_bootstrap_ci(values: list[float], seed: int, replicates: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indexes].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def radial_style_features(image: Image.Image, size: int = 384, bins: int = 32) -> np.ndarray:
    array = np.asarray(standardized_image(image, size).convert("L"), dtype=np.float64) / 255.0
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(array))))
    yy, xx = np.indices(array.shape)
    cy, cx = array.shape[0] // 2, array.shape[1] // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, radius.max() + 1e-9, bins + 1)
    radial = np.asarray([spectrum[(radius >= edges[i]) & (radius < edges[i + 1])].mean() for i in range(bins)])
    histogram, _ = np.histogram(array, bins=bins, range=(0.0, 1.0), density=True)
    return np.concatenate([radial, histogram]).astype(np.float32)


def benign_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    jpeg = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    width, height = image.size
    small = image.resize((max(8, round(width * 0.9)), max(8, round(height * 0.9))), Image.Resampling.BILINEAR)
    resized = small.resize(image.size, Image.Resampling.BILINEAR)
    intensity = ImageEnhance.Brightness(image.convert("RGB")).enhance(1.05)
    return [("jpeg_q92", jpeg), ("resize_90pct", resized), ("brightness_1.05", intensity)]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def encode_biomedclip(images: list[Image.Image], root: Path, batch_size: int = 32) -> np.ndarray:
    model, preprocess, _, _ = load_biomedclip(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    outputs = []
    try:
        for start in range(0, len(images), batch_size):
            tensors = torch.stack([preprocess(image.convert("RGB")) for image in images[start:start + batch_size]]).to(device)
            with torch.inference_mode():
                outputs.append(model.encode_image(tensors, normalize=True).float().cpu().numpy())
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)


def pareto_front(configs: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    metrics = ("style_logprob_delta", "edge_delta", "biomedclip_delta")
    front = []
    for candidate in configs:
        dominated = any(
            all(float(other[key]) >= float(candidate[key]) for key in metrics)
            and any(float(other[key]) > float(candidate[key]) for key in metrics)
            for other in configs if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda row: (-float(row["style_ci_lower"]), -float(row["edge_delta"]), -float(row["biomedclip_delta"]), float(row["rho"]), float(row["beta"])))
    if len(front) < maximum:
        remaining = [row for row in configs if row not in front]
        remaining.sort(key=lambda row: (-float(row["style_ci_lower"]), -float(row["edge_delta"]), -float(row["biomedclip_delta"])))
        front.extend(remaining[:maximum - len(front)])
    return front[:maximum]


def correctness(text: str, answer: str) -> bool | None:
    prediction, gold = rule_label(text), rule_label(answer)
    return None if prediction is None or gold is None else prediction == gold


def summarize_task_safety(pairs: list[tuple[bool | None, bool | None]]) -> dict[str, object]:
    comparable = [(clean, view) for clean, view in pairs if clean is not None and view is not None]
    rescues = sum((not clean) and view for clean, view in comparable)
    harms = sum(clean and (not view) for clean, view in comparable)
    clean_accuracy = float(np.mean([clean for clean, _ in comparable])) if comparable else float("nan")
    view_accuracy = float(np.mean([view for _, view in comparable])) if comparable else float("nan")
    return {
        "n_parseable": len(comparable), "clean_accuracy": clean_accuracy,
        "view_accuracy": view_accuracy, "rescue": rescues, "harm": harms,
        "task_safe": bool(comparable and harms <= rescues and view_accuracy >= clean_accuracy),
    }


def main() -> None:
    args = parse_args()
    rhos = sorted(set(args.rho or [0.005, 0.01, 0.02]))
    betas = sorted(set(args.beta or [0.1, 0.25, 0.5]))
    if any(value <= 0 or value > 0.5 for value in rhos) or any(value <= 0 or value > 1 for value in betas):
        raise ValueError("rho must be in (0,.5] and beta in (0,1]")
    manifests = parse_named_paths(args.source_json, "--source-json")
    roots = parse_named_paths(args.source_image_root, "--source-image-root")
    if set(manifests) != set(roots):
        raise ValueError("source JSON/root names must match")
    bank = load_style_bank(args.style_bank)
    if set(manifests) != set(bank.domains):
        raise ValueError("gate source domains must exactly match style-bank domains")
    if not bank.metadata.get("filter_report_sha256"):
        raise RuntimeError("style bank lacks strict chest-filter provenance")
    groups = {name: normalize_source_rows(name, manifests[name], roots[name], 0, args.seed) for name in sorted(manifests)}
    all_rows = [row for name in sorted(groups) for row in groups[name]]
    all_rows.sort(key=lambda row: stable_sha256({"id": row["id"], "seed": args.seed}))
    rows = all_rows[:args.max_samples]
    if len(rows) < 2:
        raise RuntimeError("insufficient gate samples")

    gate_paths = {row["image"] for row in rows}
    source_features, source_labels = [], []
    for domain in sorted(groups):
        for row in groups[domain]:
            if row["image"] in gate_paths:
                continue
            with Image.open(row["image"]) as handle:
                source_features.append(radial_style_features(handle, args.view_size))
            source_labels.append(domain)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed))
    counts = {label: source_labels.count(label) for label in set(source_labels)}
    folds = min(5, min(counts.values()))
    if folds < 2:
        raise RuntimeError("source classifier needs at least two images per domain")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    classifier_accuracy = float(cross_val_score(classifier, np.stack(source_features), np.asarray(source_labels), cv=cv).mean())
    classifier.fit(np.stack(source_features), np.asarray(source_labels))
    classifier_viable = classifier_accuracy >= 0.70

    originals, benign_by_row = [], []
    for row in rows:
        with Image.open(row["image"]) as handle:
            image = handle.convert("RGB")
        originals.append(image)
        benign_by_row.append(benign_views(image))
    candidate_images: dict[str, list[tuple[Image.Image, str]]] = {}
    for rho in rhos:
        for beta in betas:
            key = f"rho={rho:g},beta={beta:g}"
            candidate_images[key] = [counterfactual_view(image, bank, row["domain"], row["id"], args.seed, index, rho, beta) for index, (image, row) in enumerate(zip(originals, rows))]

    clip_inputs = list(originals)
    for variants in benign_by_row:
        clip_inputs.extend(view for _, view in variants)
    for key in candidate_images:
        clip_inputs.extend(view for view, _ in candidate_images[key])
    clip_features = encode_biomedclip(clip_inputs, args.biomedclip_root)
    cursor = 0
    original_clip = clip_features[cursor:cursor + len(rows)]
    cursor += len(rows)
    benign_clip = []
    for _ in rows:
        benign_clip.append(clip_features[cursor:cursor + 3])
        cursor += 3
    candidate_clip = {}
    for key in candidate_images:
        candidate_clip[key] = clip_features[cursor:cursor + len(rows)]
        cursor += len(rows)

    benign_edge = [[edge_correlation(image, view) for _, view in variants] for image, variants in zip(originals, benign_by_row)]
    benign_cos = [[cosine(original_clip[index], benign_clip[index][j]) for j in range(3)] for index in range(len(rows))]
    cpu_summaries = []
    classes = list(classifier.classes_)
    for rho in rhos:
        for beta in betas:
            key = f"rho={rho:g},beta={beta:g}"
            style_deltas, edge_deltas, clip_deltas = [], [], []
            records = []
            for index, (row, image, (view, target_domain)) in enumerate(zip(rows, originals, candidate_images[key])):
                clean_feature = radial_style_features(image, args.view_size)
                view_feature = radial_style_features(view, args.view_size)
                target_index = classes.index(target_domain)
                clean_probability = float(classifier.predict_proba(clean_feature[None])[0, target_index])
                view_probability = float(classifier.predict_proba(view_feature[None])[0, target_index])
                delta = float(np.log(view_probability + 1e-12) - np.log(clean_probability + 1e-12))
                edge = edge_correlation(image, view)
                clip_similarity = cosine(original_clip[index], candidate_clip[key][index])
                edge_delta = edge - min(benign_edge[index])
                clip_delta = clip_similarity - min(benign_cos[index])
                style_deltas.append(delta)
                edge_deltas.append(edge_delta)
                clip_deltas.append(clip_delta)
                records.append({"id": row["id"], "source_domain": row["domain"], "style_domain": target_domain, "style_logprob_delta": delta, "edge_correlation": edge, "biomedclip_similarity": clip_similarity, "edge_delta_vs_worst_benign": edge_delta, "biomedclip_delta_vs_worst_benign": clip_delta})
            style_ci = paired_bootstrap_ci(style_deltas, args.seed + 11, args.bootstrap_replicates)
            edge_ci = paired_bootstrap_ci(edge_deltas, args.seed + 13, args.bootstrap_replicates)
            clip_ci = paired_bootstrap_ci(clip_deltas, args.seed + 17, args.bootstrap_replicates)
            summary = {
                "key": key, "rho": rho, "beta": beta, "n": len(rows),
                "style_logprob_delta": float(np.mean(style_deltas)), "style_ci_lower": style_ci[0], "style_ci_upper": style_ci[1],
                "edge_delta": float(np.mean(edge_deltas)), "edge_delta_ci_lower": edge_ci[0],
                "biomedclip_delta": float(np.mean(clip_deltas)), "biomedclip_delta_ci_lower": clip_ci[0],
                "content_pass": edge_ci[0] >= 0 and clip_ci[0] >= 0,
                "style_pass": classifier_viable and style_ci[0] > 0, "records": records,
            }
            summary["cpu_pass"] = bool(summary["content_pass"] and summary["style_pass"])
            cpu_summaries.append(summary)
    eligible = [row for row in cpu_summaries if row["cpu_pass"]]
    selected_cpu = pareto_front(eligible, args.max_vlm_candidates) if eligible else []

    vlm_records: dict[str, list[dict[str, object]]] = {row["key"]: [] for row in selected_cpu}
    benign_evidence_changes = []
    if selected_cpu:
        adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
        for parameter in adapter.model.parameters():
            parameter.requires_grad_(False)
        adapter.model.eval()
        try:
            for index, row in enumerate(tqdm(rows, desc="ANCHOR-DG VLM gate")):
                image = originals[index]
                prompt = rule_no_reference_prompt(row["question"])
                input_ids, labels = build_teacher_forcing(adapter, prompt, row["answer"])
                with torch.inference_mode():
                    clean_nll, clean_logits, token_ids = sequence_forward(adapter, image, input_ids, labels, None, adapter_location="post", return_token_ids=True)
                    clean_logp = gold_token_log_probabilities(clean_logits, token_ids)
                    clean_text = decode(adapter, image, prompt, args.max_new_tokens, None, module_location="post")
                    for _, benign in benign_by_row[index]:
                        _, benign_logits, benign_ids = sequence_forward(adapter, benign, input_ids, labels, None, adapter_location="post", return_token_ids=True)
                        if not torch.equal(token_ids, benign_ids):
                            raise RuntimeError("clean/benign teacher-forcing alignment failed")
                        benign_logp = gold_token_log_probabilities(benign_logits, benign_ids)
                        benign_evidence_changes.append(float((clean_logp - benign_logp).abs().mean()))
                clean_correct = correctness(clean_text, row["answer"])
                for selected in selected_cpu:
                    view, target_domain = candidate_images[selected["key"]][index]
                    with torch.inference_mode():
                        view_nll, view_logits, view_ids = sequence_forward(adapter, view, input_ids, labels, None, adapter_location="post", return_token_ids=True)
                        if not torch.equal(token_ids, view_ids):
                            raise RuntimeError("clean/view teacher-forcing token alignment failed")
                        view_logp = gold_token_log_probabilities(view_logits, view_ids)
                        view_text = decode(adapter, view, prompt, args.max_new_tokens, None, module_location="post")
                    view_correct = correctness(view_text, row["answer"])
                    vlm_records[selected["key"]].append({
                        "id": row["id"], "source_domain": row["domain"], "style_domain": target_domain,
                        "clean_nll": float(clean_nll), "view_nll": float(view_nll),
                        "mean_abs_evidence_change": float((clean_logp - view_logp).abs().mean()),
                        "clean_text": clean_text, "view_text": view_text,
                        "clean_correct": clean_correct, "view_correct": view_correct,
                        "prediction_flip": rule_label(clean_text) != rule_label(view_text),
                    })
        finally:
            adapter.close()
    benign_p95 = float(np.quantile(benign_evidence_changes, 0.95)) if benign_evidence_changes else float("nan")
    vlm_summaries = {}
    for selected in selected_cpu:
        values = vlm_records[selected["key"]]
        task_pairs = [(row["clean_correct"], row["view_correct"]) for row in values]
        task_summary = summarize_task_safety(task_pairs)
        task_deltas = [float(view) - float(clean) for clean, view in task_pairs if clean is not None and view is not None]
        accuracy_delta_ci = paired_bootstrap_ci(task_deltas, args.seed + 23, args.bootstrap_replicates)
        evidence = [float(row["mean_abs_evidence_change"]) for row in values]
        summary = {
            "n": len(values), **task_summary,
            "accuracy_delta_ci_lower": accuracy_delta_ci[0],
            "accuracy_delta_ci_upper": accuracy_delta_ci[1],
            "flip_rate": float(np.mean([row["prediction_flip"] for row in values])),
            "median_abs_evidence_change": float(np.median(evidence)),
            "benign_evidence_p95": benign_p95,
        }
        summary["evidence_nontrivial"] = bool(evidence and float(np.median(evidence)) > benign_p95)
        summary["pass"] = bool(summary["task_safe"] and summary["evidence_nontrivial"])
        vlm_summaries[selected["key"]] = summary
    passing = [row for row in selected_cpu if vlm_summaries[row["key"]]["pass"]]
    passing.sort(key=lambda row: (-vlm_summaries[row["key"]]["view_accuracy"], vlm_summaries[row["key"]]["harm"], row["beta"], row["rho"]))
    selected = passing[0] if passing else None
    output = {
        "version": VERSION, "style_bank": str(args.style_bank.resolve()),
        "filter_version": bank.metadata.get("filter_version"),
        "unverified_source_override": bool(bank.metadata.get("unverified_source_override")),
        "projection": PROJECTION,
        "source_domains": sorted(manifests), "target_data_accessed": False,
        "source_classifier_validation": "stratified held-out-image CV; domain-LODO closed-set classification is undefined",
        "source_classifier_cv_accuracy": classifier_accuracy, "source_classifier_pass": classifier_viable,
        "source_classifier_train_n": len(source_labels), "source_classifier_gate_image_overlap": 0,
        "benign_reference": ["jpeg_q92", "resize_90pct", "brightness_1.05"],
        "cpu_summaries": {row["key"]: row for row in cpu_summaries},
        "vlm_candidate_keys": [row["key"] for row in selected_cpu],
        "vlm_summaries": vlm_summaries, "vlm_records": vlm_records,
        "selected_rho": None if selected is None else selected["rho"],
        "selected_beta": None if selected is None else selected["beta"],
        "gate_pass": selected is not None,
    }
    output["fingerprint"] = stable_sha256({key: value for key, value in output.items() if key not in {"cpu_summaries", "vlm_records"}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("fingerprint", "gate_pass", "selected_rho", "selected_beta", "source_classifier_cv_accuracy", "vlm_candidate_keys", "vlm_summaries")}, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
