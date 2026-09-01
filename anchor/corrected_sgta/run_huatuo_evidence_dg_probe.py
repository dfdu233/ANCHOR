#!/usr/bin/env python3
"""Small, falsifiable evidence-DG probe for HuatuoGPT-Vision report generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from corrected_sgta.audit_public_domain_hypotheses import descriptors
from corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows
from corrected_sgta.oe_metrics import lexical_metrics


VERSION = "huatuo-evidence-dg-probe-v1"
IGNORE_INDEX = -100
REPORT_PROMPT = (
    "You are a professional radiologist. You are provided with a chest X-ray "
    "image. Please generate a report based on the image. Please only include "
    "the content of the report in your response."
)
DEFAULT_LAYERS = (7, 14, 21, 28)
DEFAULT_ALPHAS = (0.1, 0.2, 0.3)
DEFAULT_TEMPERATURES = (1.05, 1.1, 1.2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(seed: int, domain: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{domain}:{identifier}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_report_rows(repo: Path) -> dict[str, list[dict[str, Any]]]:
    image_root = repo / "data/medheval/images"
    specs: dict[str, list[dict[str, Any]]] = {}
    for domain in ("mimic", "iuxray"):
        path = repo / f"data/mmedrag/test/report/{domain}_test.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        root = image_root if domain == "mimic" else image_root / "IU-Xray"
        converted = []
        for row in rows:
            values = row.get("image_path")
            relative = values[0] if isinstance(values, list) else values
            converted.append(
                {
                    "id": str(row["id"]),
                    "domain": domain,
                    "image": root / str(relative),
                    "reference": str(row["report"]),
                }
            )
        specs[domain] = converted

    chexpert = load_jsonl(
        repo / "data/chexpert_subset_report/chexpert_subset_report_test.jsonl"
    )
    specs["chexpert_proxy"] = [
        {
            "id": str(row["question_id"]),
            "domain": "chexpert_proxy",
            "image": (
                repo
                / "data/chexpert_subset_report/processed-v1/images"
                / str(row["image"])
            ),
            "reference": str(row["report"]),
        }
        for row in chexpert
    ]
    return specs


def load_ce_rows(repo: Path) -> dict[str, list[dict[str, Any]]]:
    roots = {
        "mimic": repo / "data/medheval/images",
        "iuxray": repo / "data/medheval/images/IU-Xray",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for domain, root in roots.items():
        rows = load_jsonl(repo / f"data/rule/test/{domain}_test.jsonl")
        output[domain] = [
            {
                "id": str(row["question_id"]),
                "domain": domain,
                "image": root / str(row["image"]),
                "prompt": str(row["question"]).replace("<image>", "").strip(),
                "reference": str(row["answer"]),
            }
            for row in rows
        ]
    return output


def select_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    samples_per_domain: int,
    seed: int,
    balance_binary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for domain, rows in rows_by_domain.items():
        ordered = sorted(
            rows, key=lambda row: stable_key(seed, domain, str(row["id"]))
        )
        available = [row for row in ordered if Path(row["image"]).is_file()]
        unique_images: list[dict[str, Any]] = []
        seen_images: set[str] = set()
        for row in available:
            image = str(Path(row["image"]).resolve())
            if image not in seen_images:
                unique_images.append(row)
                seen_images.add(image)
        if balance_binary:
            if samples_per_domain % 2:
                raise ValueError("balanced CE sampling requires an even sample count")
            grouped = {
                label: [
                    row
                    for row in unique_images
                    if str(row["reference"]).strip().lower().startswith(label)
                ]
                for label in ("yes", "no")
            }
            quota = samples_per_domain // 2
            if any(len(values) < quota for values in grouped.values()):
                raise RuntimeError(
                    f"{domain}: insufficient unique balanced rows "
                    f"yes={len(grouped['yes'])}, no={len(grouped['no'])}"
                )
            chosen = sorted(
                grouped["yes"][:quota] + grouped["no"][:quota],
                key=lambda row: stable_key(seed, domain, str(row["id"])),
            )
        else:
            chosen = unique_images[:samples_per_domain]
        if len(chosen) < samples_per_domain:
            raise RuntimeError(
                f"{domain}: requested {samples_per_domain}, found "
                f"{len(unique_images)} unique available images"
            )
        selected.extend(chosen)
        audit[domain] = {
            "total_rows": len(rows),
            "available_rows": len(available),
            "unique_available_images": len(unique_images),
            "selected": len(chosen),
            "label_counts": (
                {
                    label: sum(
                        str(row["reference"]).strip().lower().startswith(label)
                        for row in chosen
                    )
                    for label in ("yes", "no")
                }
                if balance_binary
                else None
            ),
        }
    return selected, audit


def import_huatuo(root: Path):
    sys.path.insert(0, str(root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


def image_descriptors(path: Path) -> dict[str, np.ndarray]:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return descriptors(array, bins=32)


def answer_ids(bot: Any, text: str, maximum: int) -> torch.Tensor:
    values = bot.tokenizer(
        str(text).strip(), add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]
    if values.numel() == 0:
        raise RuntimeError("answer tokenization is empty")
    return values[:maximum].to(bot.model.device)


def model_inputs(
    bot: Any,
    prompt: str,
    text: str,
    image_tensor: torch.Tensor,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    prompt = bot.insert_image_placeholder(prompt, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt), return_tensors="pt"
    ).to(bot.model.device)
    if int((prompt_ids < 0).sum()) != 1:
        raise RuntimeError("teacher-forcing prompt must contain exactly one image token")
    targets = answer_ids(bot, text, maximum)
    full = torch.cat((prompt_ids, targets), dim=0)
    labels = torch.full_like(full, IGNORE_INDEX)
    labels[-targets.numel() :] = targets
    attention = torch.ones_like(full, dtype=torch.bool)

    _, position_ids, expanded_attention, _, embeddings, expanded_labels = (
        bot.model.prepare_inputs_labels_for_multimodal_new(
            [full],
            None,
            [attention],
            None,
            [labels],
            image_tensor,
        )
    )
    if embeddings is None or expanded_labels is None:
        raise RuntimeError("Huatuo multimodal expansion returned no embeddings/labels")
    output = bot.model.model(
        input_ids=None,
        attention_mask=expanded_attention,
        position_ids=position_ids,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if output.hidden_states is None:
        raise RuntimeError("Huatuo decoder returned no hidden states")
    return expanded_labels, embeddings, output.last_hidden_state, output.hidden_states


def token_measurements(
    bot: Any,
    hidden_states: tuple[torch.Tensor, ...],
    expanded_labels: torch.Tensor,
    layers: tuple[int, ...],
    alphas: tuple[float, ...],
    temperatures: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    shifted_labels = expanded_labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    targets = shifted_labels[mask]
    if targets.numel() == 0:
        raise RuntimeError("no answer targets after multimodal expansion")
    output_weight = bot.model.get_output_embeddings().weight
    final_index = len(hidden_states) - 1
    requested = sorted(set((*layers, final_index)))
    logits_by_layer: dict[int, torch.Tensor] = {}
    vectors: dict[str, np.ndarray] = {}
    trajectory: dict[str, Any] = {}

    for layer in requested:
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(
                f"layer {layer} is outside hidden-state range 0..{final_index}"
            )
        values = hidden_states[layer][:, :-1][mask]
        normalized = values if layer == final_index else bot.model.model.norm(values)
        logits = normalized.to(output_weight.dtype) @ output_weight.T
        float_logits = logits.float()
        log_probabilities = torch.log_softmax(float_logits, dim=-1)
        probabilities = torch.softmax(float_logits, dim=-1)
        target_logp = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        top2 = float_logits.topk(2, dim=-1).values
        trajectory[str(layer)] = {
            "mean_target_nll": float(-target_logp.mean().cpu()),
            "mean_entropy": float(
                (-(probabilities * log_probabilities).sum(-1)).mean().cpu()
            ),
            "mean_top1_top2_margin": float((top2[:, 0] - top2[:, 1]).mean().cpu()),
            "mean_target_top1": float(
                (float_logits.argmax(-1) == targets).float().mean().cpu()
            ),
        }
        logits_by_layer[layer] = float_logits
        vectors[f"decoder_layer_{layer}"] = (
            normalized.float().mean(0).cpu().numpy().astype(np.float32)
        )

    final_logits = logits_by_layer[final_index]
    mix_nll = {"final": trajectory[str(final_index)]["mean_target_nll"]}
    for temperature in temperatures:
        mix_nll[f"temperature_{temperature:g}"] = float(
            F.cross_entropy(final_logits / temperature, targets).cpu()
        )
    for layer in layers:
        if layer == final_index:
            continue
        for alpha in alphas:
            mixed = (1.0 - alpha) * final_logits + alpha * logits_by_layer[layer]
            key = f"layer_{layer}_alpha_{alpha:g}"
            mix_nll[key] = float(F.cross_entropy(mixed, targets).cpu())
            source = logits_by_layer[layer]
            source_centered = source - source.mean(dim=-1, keepdim=True)
            final_centered = final_logits - final_logits.mean(dim=-1, keepdim=True)
            scale = final_centered.std(dim=-1, keepdim=True) / (
                source_centered.std(dim=-1, keepdim=True) + 1e-6
            )
            norm_matched = source_centered * scale
            normalized_mix = (
                (1.0 - alpha) * final_centered + alpha * norm_matched
            )
            norm_key = f"norm_layer_{layer}_alpha_{alpha:g}"
            mix_nll[norm_key] = float(
                F.cross_entropy(normalized_mix, targets).cpu()
            )
    return {
        "token_count": int(targets.numel()),
        "final_layer": final_index,
        "layers": trajectory,
        "mix_nll": mix_nll,
    }, vectors


@torch.inference_mode()
def measure_sequence(
    bot: Any,
    prompt: str,
    text: str,
    image_tensor: torch.Tensor,
    maximum: int,
    layers: tuple[int, ...],
    alphas: tuple[float, ...],
    temperatures: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    captured: dict[str, torch.Tensor] = {}
    projector = bot.model.get_model().mm_projector

    def capture(_module: Any, inputs: Any, output: Any) -> None:
        captured["pre"] = inputs[0].detach()
        captured["post"] = output.detach()

    handle = projector.register_forward_hook(capture)
    try:
        labels, _, _, hidden = model_inputs(
            bot, prompt, text, image_tensor, maximum
        )
    finally:
        handle.remove()
    measurements, vectors = token_measurements(
        bot, hidden, labels, layers, alphas, temperatures
    )
    if "pre" not in captured or "post" not in captured:
        raise RuntimeError("vision projector hook captured no features")
    vectors["visual_pre"] = (
        captured["pre"].float().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
    )
    vectors["visual_post"] = (
        captured["post"].float().mean(dim=(0, 1)).cpu().numpy().astype(np.float32)
    )
    return measurements, vectors


def feature_path(root: Path, domain: str, identifier: str) -> Path:
    digest = hashlib.sha256(f"{domain}:{identifier}".encode()).hexdigest()[:20]
    return root / "features" / f"{domain}_{digest}.npz"


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(row["record_key"])
        for row in load_jsonl(path)
        if row.get("status") == "ok"
    }


def source_probe(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, float] | None:
    counts = np.bincount(labels)
    if len(counts) < 2 or counts.min() < 2:
        return None
    folds = min(4, int(counts.min()))
    predicted = np.empty_like(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        components = min(8, len(train) - len(np.unique(labels)), features.shape[1])
        steps: list[Any] = [StandardScaler()]
        if features.shape[1] > 16 and components >= 2:
            steps.append(PCA(n_components=components, whiten=True, random_state=seed))
        steps.append(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            )
        )
        model = make_pipeline(*steps).fit(features[train], labels[train])
        predicted[test] = model.predict(features[test])
    return {
        "n": int(len(labels)),
        "folds": folds,
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "chance_accuracy": float(1.0 / len(np.unique(labels))),
    }


def quality_probe(
    features: np.ndarray,
    quality: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, Any] | None:
    counts = np.bincount(labels)
    if len(quality) < 12 or counts.min() < 3 or np.std(quality) == 0:
        return None
    folds = min(4, int(counts.min()))
    predicted = np.zeros_like(quality)
    source_controlled_quality = np.zeros_like(quality)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels):
        train_means = {
            label: float(np.mean(quality[train][labels[train] == label]))
            for label in np.unique(labels)
        }
        train_target = np.asarray(
            [quality[index] - train_means[int(labels[index])] for index in train]
        )
        source_controlled_quality[test] = np.asarray(
            [quality[index] - train_means[int(labels[index])] for index in test]
        )
        components = min(8, len(train) - 2, features.shape[1])
        steps: list[Any] = [StandardScaler()]
        if features.shape[1] > 16 and components >= 2:
            steps.append(PCA(n_components=components, whiten=True, random_state=seed))
        steps.append(Ridge(alpha=10.0))
        model = make_pipeline(*steps).fit(features[train], train_target)
        predicted[test] = model.predict(features[test])
    correlation = spearmanr(predicted, source_controlled_quality)
    return {
        "n": int(len(quality)),
        "folds": folds,
        "target": "quality minus training-fold source mean",
        "spearman": float(correlation.statistic),
        "pvalue": float(correlation.pvalue),
        "prediction_std": float(np.std(predicted)),
        "controlled_target_std": float(np.std(source_controlled_quality)),
    }


def lodo_setting_selection(
    records: list[dict[str, Any]], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    domains = sorted({str(row["domain"]) for row in records})
    settings = [
        setting
        for setting in sorted(records[0]["reference_evidence"]["mix_nll"])
        if setting == "final" or setting.startswith(prefixes)
    ]
    folds: dict[str, Any] = {}
    for heldout in domains:
        train = [row for row in records if row["domain"] != heldout]
        test = [row for row in records if row["domain"] == heldout]
        means = {
            setting: float(
                np.mean(
                    [row["reference_evidence"]["mix_nll"][setting] for row in train]
                )
            )
            for setting in settings
        }
        chosen = min(means, key=means.get)
        final = float(
            np.mean(
                [row["reference_evidence"]["mix_nll"]["final"] for row in test]
            )
        )
        selected = float(
            np.mean(
                [row["reference_evidence"]["mix_nll"][chosen] for row in test]
            )
        )
        folds[heldout] = {
            "training_domains": sorted({str(row["domain"]) for row in train}),
            "selected_setting": chosen,
            "heldout_n": len(test),
            "heldout_final_reference_nll": final,
            "heldout_selected_reference_nll": selected,
            "heldout_nll_delta": selected - final,
            "improves_heldout": selected < final,
        }
    return {
        "folds": folds,
        "all_heldout_improve": all(
            row["improves_heldout"] for row in folds.values()
        ),
    }


def analyze(output_dir: Path, seed: int, task: str) -> dict[str, Any]:
    records = [
        row
        for row in load_jsonl(output_dir / "raw.jsonl")
        if row.get("status") == "ok"
    ]
    domains = sorted({str(row["domain"]) for row in records})
    domain_index = {domain: index for index, domain in enumerate(domains)}
    labels = np.asarray([domain_index[str(row["domain"])] for row in records])
    quality = np.asarray(
        [
            (
                float(row["ce_evaluation"]["rule_normalized_correct"])
                if task == "ce"
                else float(row["lexical_metrics"]["rouge_l"])
            )
            for row in records
        ],
        dtype=np.float64,
    )
    arrays = [np.load(row["feature_file"], allow_pickle=False) for row in records]
    feature_names = (
        "pixel_intensity",
        "pixel_radial",
        "visual_pre",
        "visual_post",
        "decoder_final",
        "evidence_trajectory",
    )
    source_results: dict[str, Any] = {}
    quality_results: dict[str, Any] = {}
    for name in feature_names:
        matrix = np.stack([item[name] for item in arrays])
        source_results[name] = source_probe(matrix, labels, seed)
        quality_results[name] = quality_probe(matrix, quality, labels, seed)
    for item in arrays:
        item.close()

    by_domain = {}
    for domain in domains:
        chosen = [row for row in records if row["domain"] == domain]
        common = {
            "n": len(chosen),
            "mean_generated_words": float(
                np.mean([len(row["text"].split()) for row in chosen])
            ),
            "mean_visual_support_nll_delta": float(
                np.mean([row["visual_support_nll_delta"] for row in chosen])
            ),
        }
        if task == "ce":
            common.update(
                {
                    "rule_normalized_accuracy": float(
                        np.mean(
                            [
                                row["ce_evaluation"]["rule_normalized_correct"]
                                for row in chosen
                            ]
                        )
                    ),
                    "decision_first_accuracy": float(
                        np.mean(
                            [
                                row["ce_evaluation"]["decision_first_correct"]
                                for row in chosen
                            ]
                        )
                    ),
                    "decision_first_parse_rate": float(
                        np.mean(
                            [
                                row["ce_evaluation"][
                                    "decision_first_prediction"
                                ]
                                is not None
                                for row in chosen
                            ]
                        )
                    ),
                }
            )
        else:
            common.update(
                {
                    "mean_rouge_l": float(
                        np.mean(
                            [row["lexical_metrics"]["rouge_l"] for row in chosen]
                        )
                    ),
                    "mean_token_f1": float(
                        np.mean(
                            [row["lexical_metrics"]["token_f1"] for row in chosen]
                        )
                    ),
                }
            )
        by_domain[domain] = common
    return {
        "version": VERSION,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "status": "complete",
        "task": task,
        "n": len(records),
        "domains": domains,
        "by_domain": by_domain,
        "source_probe": source_results,
        (
            "quality_probe_ce_rule_normalized_correct"
            if task == "ce"
            else "quality_probe_rouge_l"
        ): quality_results,
        "lodo_reference_layer_selection": (
            lodo_setting_selection(records, ("layer_",))
            if len(domains) >= 2
            else None
        ),
        "lodo_reference_norm_matched_layer_selection": (
            lodo_setting_selection(records, ("norm_layer_",))
            if len(domains) >= 2
            else None
        ),
        "lodo_reference_temperature_control": (
            lodo_setting_selection(records, ("temperature_",))
            if len(domains) >= 2
            else None
        ),
        "interpretation_guard": (
            (
                "CE quality probing uses RULE's normalized generated-sentence "
                "metric; decision-first accuracy and parse rate remain diagnostics. "
                if task == "ce"
                else (
                    "ROUGE-L/token F1 and teacher-forced reference NLL are "
                    "diagnostics, not clinical factuality judgments. "
                )
            )
            + "No reference was used to generate or select an output."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=("report", "ce"), default="report")
    parser.add_argument(
        "--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision")
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-probe-tokens", type=int, default=192)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES
    )
    parser.add_argument("--generation-cache", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    raw_path = args.output_dir / "raw.jsonl"
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} exists; use --resume or choose another directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features").mkdir(exist_ok=True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    source_rows = (
        load_ce_rows(repo) if args.task == "ce" else load_report_rows(repo)
    )
    rows, availability = select_rows(
        source_rows,
        args.samples_per_domain,
        args.seed,
        balance_binary=args.task == "ce",
    )
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "repo_root": str(repo),
        "model": str(args.model_dir.resolve()),
        "huatuo_root": str(args.huatuo_root.resolve()),
        "samples_per_domain": args.samples_per_domain,
        "task": args.task,
        "n": len(rows),
        "domains": sorted(availability),
        "availability": availability,
        "selection": (
            "stable hash, one question per image, balanced 50/50 Yes/No"
            if args.task == "ce"
            else "stable hash, one report per image"
        ),
        "prompt": (
            "raw RULE question without the dataset image placeholder"
            if args.task == "ce"
            else REPORT_PROMPT
        ),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "repetition_penalty": 1.2,
        },
        "probe": {
            "max_tokens": args.max_probe_tokens,
            "layers": list(args.layers),
            "alphas": list(args.alphas),
            "temperatures": list(args.temperatures),
            "zero_visual_counterfactual": "zeros in model-visible normalized tensor",
            "reference_used_for_generation_or_selection": False,
        },
        "generation_cache": (
            str(args.generation_cache.resolve()) if args.generation_cache else None
        ),
        "code_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
    }
    atomic_json(args.output_dir / "config.json", config)
    done = completed_ids(raw_path) if args.resume else set()

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    bot.gen_kwargs.update(config["generation"])
    bot.gen_kwargs.pop("temperature", None)
    layers = tuple(args.layers)
    alphas = tuple(args.alphas)
    temperatures = tuple(args.temperatures)
    generation_cache = {}
    if args.generation_cache:
        generation_cache = {
            str(row["record_key"]): str(row["text"])
            for row in load_jsonl(args.generation_cache)
            if row.get("status") == "ok" and row.get("text")
        }

    for index, row in enumerate(rows):
        record_key = f"{row['domain']}:{row['id']}"
        if record_key in done:
            continue
        record: dict[str, Any] = {
            "version": VERSION,
            "record_key": record_key,
            "id": row["id"],
            "domain": row["domain"],
            "image": str(Path(row["image"]).resolve()),
            "prompt": str(row.get("prompt") or REPORT_PROMPT),
            "reference": row["reference"],
            "status": "error",
        }
        try:
            prompt = str(row.get("prompt") or REPORT_PROMPT)
            if generation_cache:
                text = generation_cache.get(record_key, "")
                if not text:
                    raise KeyError(f"generation cache lacks {record_key}")
            else:
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                response = bot.inference(prompt, [str(row["image"])])
                text = str(response[0] if response else "").strip()
            if not text:
                raise RuntimeError("generation returned empty text")
            image_tensors = torch.stack(
                bot.get_image_tensors([str(row["image"])])
            ).to(device=bot.model.device, dtype=torch.bfloat16)
            generated, vectors = measure_sequence(
                bot,
                prompt,
                text,
                image_tensors,
                args.max_probe_tokens,
                layers,
                alphas,
                temperatures,
            )
            reference, _ = measure_sequence(
                bot,
                prompt,
                row["reference"],
                image_tensors,
                args.max_probe_tokens,
                layers,
                alphas,
                temperatures,
            )
            zero_generated, _ = measure_sequence(
                bot,
                prompt,
                text,
                torch.zeros_like(image_tensors),
                args.max_probe_tokens,
                layers,
                alphas,
                temperatures,
            )
            pixel = image_descriptors(Path(row["image"]))
            final_layer = str(generated["final_layer"])
            trajectory_values = []
            for layer in sorted(generated["layers"], key=int):
                layer_row = generated["layers"][layer]
                trajectory_values.extend(
                    [
                        layer_row["mean_target_nll"],
                        layer_row["mean_entropy"],
                        layer_row["mean_top1_top2_margin"],
                        layer_row["mean_target_top1"],
                    ]
                )
            visual_support = (
                zero_generated["mix_nll"]["final"]
                - generated["mix_nll"]["final"]
            )
            trajectory_values.extend(
                [
                    visual_support,
                    reference["mix_nll"]["final"],
                    generated["mix_nll"]["final"],
                ]
            )
            path = feature_path(args.output_dir, row["domain"], row["id"])
            np.savez_compressed(
                path,
                pixel_intensity=pixel["intensity_stats"],
                pixel_radial=pixel["radial_all"],
                visual_pre=vectors["visual_pre"],
                visual_post=vectors["visual_post"],
                decoder_final=vectors[f"decoder_layer_{final_layer}"],
                evidence_trajectory=np.asarray(
                    trajectory_values, dtype=np.float32
                ),
                **{
                    key: value
                    for key, value in vectors.items()
                    if key.startswith("decoder_layer_")
                },
            )
            record.update(
                {
                    "status": "ok",
                    "text": text,
                    "lexical_metrics": lexical_metrics(text, row["reference"]),
                    "generated_evidence": generated,
                    "reference_evidence": reference,
                    "zero_visual_generated_evidence": zero_generated,
                    "visual_support_nll_delta": visual_support,
                    "feature_file": str(path),
                    "reference_used_for_generation_or_selection": False,
                    "completed_at": now_iso(),
                }
            )
            if args.task == "ce":
                _, ce_records = evaluate_rule_rows(
                    [
                        {
                            "question_id": row["id"],
                            "question": prompt,
                            "answer": row["reference"],
                            "image": str(row["image"]),
                        }
                    ],
                    [{"question_id": row["id"], "text": text}],
                )
                record["ce_evaluation"] = ce_records[0]
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            record.update(
                {
                    "error": f"CUDA OOM: {error}",
                    "traceback": traceback.format_exc(),
                    "completed_at": now_iso(),
                }
            )
        except Exception as error:
            record.update(
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "completed_at": now_iso(),
                }
            )
        append_jsonl(raw_path, record)
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{len(rows)}",
                    "record_key": record_key,
                    "status": record["status"],
                    "text": record.get("text", "")[:160],
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    del bot
    torch.cuda.empty_cache()
    summary = analyze(args.output_dir, args.seed, args.task)
    summary["config"] = config
    summary["errors"] = sum(
        row.get("status") != "ok" for row in load_jsonl(raw_path)
    )
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
