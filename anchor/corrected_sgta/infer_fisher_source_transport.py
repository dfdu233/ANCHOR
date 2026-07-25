"""Resumable n=32 LLaVA probe for question-conditioned Fisher source transport."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.fisher_source_transport import (
    equal_dose,
    fisher_matrix,
    pca_geometry,
    question_conditioned_direction,
    source_closure,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.infer_projector_parameter_metric import (
    LlavaParameterMetricAdapter,
    checked_feature_file,
)
from corrected_sgta.methods import gamma_transform
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import sha256_file


ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = "question-conditioned-fisher-source-transport-v1"
ARMS = ("original", "matched_xray", "wrong_ct", "wrong_mri", "gamma_feature")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xray-features", required=True, type=Path)
    parser.add_argument("--modality-features", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dose-fraction", type=float, default=0.1)
    parser.add_argument("--explained-variance", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.8)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_geometry(xray_path: Path, modality_path: Path, explained_variance: float):
    xray, xray_meta = checked_feature_file(xray_path, {"exact_raw": (64, 1024)})
    modality, modality_meta = checked_feature_file(
        modality_path,
        {"ct_source_raw": (64, 1024), "mri_source_raw": (64, 1024)},
    )
    arrays = {
        "xray": xray["exact_raw"],
        "ct": modality["ct_source_raw"],
        "mri": modality["mri_source_raw"],
    }
    geometry = {}
    for name, values in arrays.items():
        mean, basis = pca_geometry(values, explained_variance)
        geometry[name] = {"mean": mean, "basis": basis}
    provenance = {
        "xray_features": str(xray_path.resolve()),
        "xray_features_sha256": sha256_file(xray_path),
        "xray_upstream_fingerprint": xray_meta.get("fingerprint"),
        "modality_features": str(modality_path.resolve()),
        "modality_features_sha256": sha256_file(modality_path),
        "modality_upstream_fingerprint": modality_meta.get("fingerprint"),
        "ranks": {name: int(item["basis"].shape[1]) for name, item in geometry.items()},
    }
    return geometry, provenance


def eligible_rows(rows: list[dict], seed: int, maximum: int) -> list[dict]:
    selected = []
    seen_images = set()
    for row in sorted(
        rows, key=lambda item: hashlib.sha256(f"{seed}:{item.get('qid')}".encode()).hexdigest()
    ):
        try:
            if task_kind(row) != "binary" or labels_for_sample(row) != ("Yes", "No"):
                continue
            ground_truth_index(row)
            image_name = str(row.get("img_name", ""))
            if image_name in seen_images or resolve_image(image_name) is None:
                continue
            selected.append(row)
            seen_images.add(image_name)
        except ProtocolError:
            continue
        if maximum and len(selected) >= maximum:
            break
    return selected


@contextmanager
def shifted_projector(projector, delta: torch.Tensor | None):
    if delta is None:
        yield
        return
    original_forward = projector.forward

    def hooked(features: torch.Tensor, *args, **kwargs):
        shift = delta.to(device=features.device, dtype=features.dtype)
        return original_forward(features + shift.view(1, 1, -1), *args, **kwargs)

    projector.forward = hooked
    try:
        yield
    finally:
        projector.forward = original_forward


def raw_tokens(adapter: LlavaParameterMetricAdapter, image_tensor) -> torch.Tensor:
    tower = adapter.model.get_vision_tower()
    with torch.inference_mode():
        value = tower(image_tensor)
    if isinstance(value, list) or value.ndim != 3 or value.shape[0] != 1:
        raise RuntimeError("expected one raw visual-token tensor [1,T,D]")
    return value[0].float()


def label_logits(
    adapter: LlavaParameterMetricAdapter,
    image_tensor,
    image_size: tuple[int, int],
    prompt: str,
    label_ids: list[int],
    delta: torch.Tensor | None,
) -> torch.Tensor:
    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    projector = adapter.model.get_model().mm_projector
    with shifted_projector(projector, delta):
        _, position_ids, attention_mask, _, inputs_embeds, _ = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image_size],
            )
        )
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[0, -1]
        weights = adapter.model.get_output_embeddings().weight[label_ids]
        return hidden.to(weights.dtype) @ weights.T


def output_jacobian(
    adapter: LlavaParameterMetricAdapter,
    image_tensor,
    image_size: tuple[int, int],
    prompt: str,
    label_ids: list[int],
    dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapter.model.requires_grad_(False)
    delta = torch.zeros(
        dimension, device=adapter.model.device, dtype=torch.float32, requires_grad=True
    )
    with torch.enable_grad():
        logits = label_logits(
            adapter, image_tensor, image_size, prompt, label_ids, delta
        ).float()
        gradients = [
            torch.autograd.grad(
                logits[index], delta, retain_graph=index + 1 < len(label_ids)
            )[0]
            for index in range(len(label_ids))
        ]
    jacobian = torch.stack(gradients).detach()
    probabilities = torch.softmax(logits.detach(), dim=0)
    del logits, delta
    return jacobian, probabilities


def arm_evidence(
    adapter,
    image_tensor,
    image_size,
    prompt,
    label_ids,
    delta,
) -> tuple[list[float], list[float]]:
    with torch.inference_mode():
        logits = label_logits(
            adapter, image_tensor, image_size, prompt, label_ids, delta
        ).float()
        probabilities = torch.softmax(logits, dim=0)
    return logits.cpu().tolist(), probabilities.cpu().tolist()


def direction_energy(
    jacobian: torch.Tensor, probabilities: torch.Tensor, delta: torch.Tensor
) -> float:
    response = jacobian @ delta
    fisher = fisher_matrix(probabilities)
    return float((response @ fisher @ response).detach().cpu())


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    geometry, source_provenance = load_geometry(
        args.xray_features, args.modality_features, args.explained_variance
    )
    config = {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "source_geometry": source_provenance,
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid), first row per unique img_name",
        "dose_fraction": args.dose_fraction,
        "explained_variance": args.explained_variance,
        "gamma": args.gamma,
        "operator": "P_U G_q P_U (mu-z), normalized to 0.1*||P_U(mu-z)||",
        "arms": list(ARMS),
        "labels_used_for_direction": False,
        "ground_truth_used_for_direction": False,
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(f"metadata mismatch; choose a new output: {args.output}")
    else:
        atomic_json(metadata_path, metadata)
    repair_truncated_jsonl_tail(args.output)
    completed = load_successful_qids(args.output, fingerprint)
    selected = eligible_rows(rows, args.seed, args.max_samples)
    pending = [row for row in selected if str(row["qid"]) not in completed]
    print(
        f"fisher-transport fingerprint={fingerprint[:12]} eligible={len(pending)}/{len(selected)}",
        flush=True,
    )
    if not pending:
        return

    adapter = LlavaParameterMetricAdapter()
    device = adapter.model.device
    torch_geometry = {
        name: {
            "mean": torch.as_tensor(item["mean"], device=device),
            "basis": torch.as_tensor(item["basis"], device=device),
        }
        for name, item in geometry.items()
    }
    errors = 0
    try:
        with args.output.open("a") as output:
            for sample in tqdm(pending, desc="question-conditioned Fisher transport"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    with Image.open(image_path) as source:
                        image = resize_image(source.convert("RGB"), args.max_image_side)
                    prompt = build_prompt(sample)
                    labels = labels_for_sample(sample)
                    label_ids = []
                    for label in labels:
                        ids = adapter.tokenizer.encode(label, add_special_tokens=False)
                        if len(ids) != 1:
                            raise RuntimeError(f"label is not one token: {label!r} -> {ids}")
                        label_ids.append(int(ids[0]))
                    image_tensor = adapter.prepare_image_tensor(image)
                    pooled = raw_tokens(adapter, image_tensor).mean(dim=0)
                    jacobian, original_probabilities = output_jacobian(
                        adapter,
                        image_tensor,
                        image.size,
                        prompt,
                        label_ids,
                        pooled.numel(),
                    )

                    directions = {}
                    diagnostics = {}
                    for modality in ("xray", "ct", "mri"):
                        item = torch_geometry[modality]
                        residual = item["mean"] - pooled
                        delta, detail = question_conditioned_direction(
                            jacobian,
                            original_probabilities,
                            item["basis"],
                            residual,
                            args.dose_fraction,
                        )
                        directions[modality] = delta
                        diagnostics[modality] = {
                            **detail,
                            **source_closure(pooled, item["mean"], delta),
                        }
                    matched_dose = diagnostics["xray"]["dose"]
                    directions["ct"] = equal_dose(directions["ct"], matched_dose)
                    directions["mri"] = equal_dose(directions["mri"], matched_dose)
                    for modality in ("ct", "mri"):
                        diagnostics[modality].update(
                            source_closure(
                                pooled, torch_geometry[modality]["mean"], directions[modality]
                            )
                        )
                        diagnostics[modality]["dose"] = float(
                            directions[modality].norm().detach().cpu()
                        )
                        diagnostics[modality]["fisher_energy"] = direction_energy(
                            jacobian, original_probabilities, directions[modality]
                        )

                    gamma_image = gamma_transform(image, args.gamma)
                    gamma_tensor = adapter.prepare_image_tensor(gamma_image)
                    gamma_direction = raw_tokens(adapter, gamma_tensor).mean(dim=0) - pooled
                    gamma_direction = equal_dose(gamma_direction, matched_dose)
                    directions["gamma"] = gamma_direction
                    diagnostics["gamma"] = {
                        "dose": float(gamma_direction.norm().detach().cpu()),
                        "fisher_energy": direction_energy(
                            jacobian, original_probabilities, gamma_direction
                        ),
                    }

                    arm_deltas = [
                        None,
                        directions["xray"],
                        directions["ct"],
                        directions["mri"],
                        directions["gamma"],
                    ]
                    arm_logits, arm_probabilities = [], []
                    for delta in arm_deltas:
                        logits, probabilities = arm_evidence(
                            adapter,
                            image_tensor,
                            image.size,
                            prompt,
                            label_ids,
                            delta,
                        )
                        arm_logits.append(logits)
                        arm_probabilities.append(probabilities)
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "arms": list(ARMS),
                        "label_logits": arm_logits,
                        "label_probabilities": arm_probabilities,
                        "direction_diagnostics": {
                            "matched_xray": diagnostics["xray"],
                            "wrong_ct": diagnostics["ct"],
                            "wrong_mri": diagnostics["mri"],
                            "gamma_feature": diagnostics["gamma"],
                        },
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "error",
                        "qid": sample.get("qid"),
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        gc.collect()
                        torch.cuda.empty_cache()
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        adapter.close()
    print(f"finished rows={len(pending)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
