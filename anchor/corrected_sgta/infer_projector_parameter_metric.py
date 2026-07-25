"""Resumable LLaVA CXR probe for pre-projector parameter-metric alignment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import (
    encode_array,
    load_successful_qids,
    repair_truncated_jsonl_tail,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_surface import LlavaMedSurfaceAdapter
from corrected_sgta.projector_parameter_metric import ParameterMetricTransport
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
VERSION = "sgta-projector-parameter-metric-probe-v4"
ARMS = (
    "original",
    "metric_matched",
    "euclidean_matched",
    "metric_wrong_ct",
    "metric_wrong_mri",
    "away",
)


@dataclass
class ExactForward:
    surface_logits: np.ndarray
    exact_next_token_nll: np.ndarray
    language_feature: np.ndarray
    exact_label_token_ids: list[int]


def snapshot_preprocessed_image(
    image_tensor: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor | list[torch.Tensor]:
    """Clone a prepared tensor solely to audit that model code does not mutate it."""

    if isinstance(image_tensor, list):
        return [item.detach().clone() for item in image_tensor]
    return image_tensor.detach().clone()


def assert_preprocessed_image_unchanged(
    image_tensor: torch.Tensor | list[torch.Tensor],
    snapshot: torch.Tensor | list[torch.Tensor],
) -> None:
    """Fail closed if multimodal preparation mutates the reusable image tensor."""

    if isinstance(image_tensor, list) != isinstance(snapshot, list):
        raise RuntimeError("preprocessed image container type changed in-place")
    current = image_tensor if isinstance(image_tensor, list) else [image_tensor]
    frozen = snapshot if isinstance(snapshot, list) else [snapshot]
    if len(current) != len(frozen):
        raise RuntimeError("preprocessed image list length changed in-place")
    for index, (value, reference) in enumerate(zip(current, frozen)):
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not bool(torch.equal(value, reference))
        ):
            raise RuntimeError(
                f"preprocessed image tensor {index} was mutated in-place"
            )


class LlavaParameterMetricAdapter(LlavaMedSurfaceAdapter):
    """LLaVA adapter with exact next-token NLL for literal Yes/No labels."""

    @torch.inference_mode()
    def prepare_image_tensor(
        self, image: Image.Image
    ) -> torch.Tensor | list[torch.Tensor]:
        from llava.mm_utils import process_images

        image_tensor = process_images(
            [image], self.image_processor, self.model.config
        )
        if isinstance(image_tensor, list):
            return [
                item.to(self.model.device, dtype=self.model.dtype)
                for item in image_tensor
            ]
        return image_tensor.to(self.model.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def forward_binary_exact(
        self,
        image_tensor: torch.Tensor | list[torch.Tensor],
        image_size: tuple[int, int],
        prompt: str,
        labels: Sequence[str],
    ) -> ExactForward:
        snapshot = snapshot_preprocessed_image(image_tensor)
        if tuple(labels) != ("Yes", "No"):
            raise ValueError("this preregistered probe is restricted to Yes/No")
        exact_ids = []
        for label in labels:
            tokens = self.tokenizer.encode(str(label), add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(
                    f"exact next-token probe requires one-token labels: {label!r} -> {tokens}"
                )
            exact_ids.append(int(tokens[0]))
        input_ids = self._prompt_ids(prompt).to(self.model.device)
        _, position_ids, attention_mask, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image_size],
            )
        )
        base_output = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = base_output.last_hidden_state[0, -1]
        vocabulary_weight = self.model.get_output_embeddings().weight
        vocabulary_logits = hidden.to(vocabulary_weight.dtype) @ vocabulary_weight.T
        log_probability = torch.log_softmax(vocabulary_logits.float(), dim=-1)
        exact_nll = -log_probability[exact_ids]
        surface_logits = torch.stack(
            [
                vocabulary_logits[group].max()
                for group in self.label_id_groups(labels)
            ]
        )
        assert_preprocessed_image_unchanged(image_tensor, snapshot)
        return ExactForward(
            surface_logits=surface_logits.float().cpu().numpy(),
            exact_next_token_nll=exact_nll.cpu().numpy(),
            language_feature=hidden.float().cpu().numpy(),
            exact_label_token_ids=exact_ids,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xray-features", required=True, type=Path)
    parser.add_argument("--modality-features", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def checked_feature_file(
    path: Path, required: dict[str, tuple[int, int]]
) -> tuple[dict[str, np.ndarray], dict]:
    analysis_path = path.with_name("analysis.json")
    if not analysis_path.is_file():
        raise RuntimeError(f"missing feature provenance: {analysis_path}")
    analysis = json.loads(analysis_path.read_text())
    if analysis.get("features_sha256") != sha256_file(path):
        raise RuntimeError(f"feature/provenance hash mismatch: {path}")
    payload = np.load(path, allow_pickle=False)
    arrays = {}
    for key, shape in required.items():
        if key not in payload:
            raise RuntimeError(f"missing raw source array {key!r} in {path}")
        value = payload[key].astype(np.float32)
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"raw source array {key!r} has shape {value.shape}, expected {shape}"
            )
        arrays[key] = value
    return arrays, analysis


def load_raw_centers(xray_path: Path, modality_path: Path) -> tuple[dict, dict]:
    xray, xray_meta = checked_feature_file(
        xray_path, {"exact_raw": (64, 1024)}
    )
    modality, modality_meta = checked_feature_file(
        modality_path,
        {"ct_source_raw": (64, 1024), "mri_source_raw": (64, 1024)},
    )
    centers = {
        "xray": xray["exact_raw"].mean(axis=0),
        "ct": modality["ct_source_raw"].mean(axis=0),
        "mri": modality["mri_source_raw"].mean(axis=0),
    }
    provenance = {
        "xray": {
            "role": (
                "fixed 64-image exact LLaVA-alignment X-ray support proxy; "
                "not claimed to be the checkpoint training mean"
            ),
            "features": str(xray_path.resolve()),
            "features_sha256": sha256_file(xray_path),
            "analysis_sha256": sha256_file(xray_path.with_name("analysis.json")),
            "source_key": "exact_raw",
            "n": 64,
            "upstream_fingerprint": xray_meta.get("fingerprint"),
        },
        "ct_mri_controls": {
            "role": "fixed 64-image wrong-modality source controls",
            "features": str(modality_path.resolve()),
            "features_sha256": sha256_file(modality_path),
            "analysis_sha256": sha256_file(
                modality_path.with_name("analysis.json")
            ),
            "source_keys": ["ct_source_raw", "mri_source_raw"],
            "n_each": 64,
            "upstream_fingerprint": modality_meta.get("fingerprint"),
        },
    }
    return centers, provenance


def eligible_rows(rows: list[dict], seed: int, maximum: int) -> list[dict]:
    selected = []
    for row in rows:
        try:
            if task_kind(row) != "binary":
                continue
            if labels_for_sample(row) != ("Yes", "No"):
                continue
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is not None:
                selected.append(row)
        except ProtocolError:
            continue
    selected.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest()
    )
    return selected[:maximum] if maximum else selected


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    centers, center_provenance = load_raw_centers(
        args.xray_features, args.modality_features
    )
    project_root = Path(__file__).resolve().parents[1]
    code_files = (
        "corrected_sgta/projector_parameter_metric.py",
        "corrected_sgta/infer_projector_parameter_metric.py",
        "corrected_sgta/models.py",
        "corrected_sgta/models_surface.py",
        "corrected_sgta/cache.py",
        "corrected_sgta/protocol_v2.py",
    )
    config = {
        "cache_version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": "llava",
        "model_identity": model_identity("llava"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "source_support": center_provenance,
        "max_samples": args.max_samples,
        "max_image_side": args.max_image_side,
        "seed": args.seed,
        "subset_order": "sha256(seed:qid)",
        "question_type": "binary Yes/No only",
        "primary_score": (
            "exact next-token conditional NLL for literal Yes and No; both "
            "labels are verified to be one token, so this equals their complete "
            "label-string NLL only for this preregistered probe"
        ),
        "free_generation_evaluated": False,
        "operator": (
            "raw pre-projector delta follows M*R with "
            "M=Jbar^T*Jbar and Jbar=mean_t J_pi(z_t); "
            "the optimal direction step uses alpha=(R^T d)/(d^T d)"
        ),
        "dose_rule": (
            "compute optimal X-ray, CT, and MRI metric steps first; use their "
            "minimum norm as one per-image common raw L2 dose for all metric, "
            "Euclidean matched, and away arms"
        ),
        "metric_scope": (
            "exact local pullback metric for the projected visual-token mean "
            "under a uniform raw-token shift"
        ),
        "sample_geometry_freeze": (
            "freeze geometry, directions, and common dose at the first "
            "transformed arm; later repeated vision forwards reuse the fixed "
            "intervention and fail closed if max raw-token drift exceeds 1e-3"
        ),
        "deterministic_image_preprocessing": (
            "call LLaVA process_images exactly once per sample and reuse the "
            "same immutable preprocessed tensor plus original image size for "
            "the original and all transformed arms"
        ),
        "arms": list(ARMS),
        "outcome_labels_used_for_transform": False,
        "code_identity": {
            name: sha256_file(project_root / name) for name in code_files
        },
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "cache_version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "dataset_validation": validation,
        "non_claims": [
            "The X-ray support mean is not identified as the VLM training mean.",
            "The pullback metric measures projector visibility, not correctness.",
            "Pilot outcomes cannot establish statistical significance.",
        ],
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"metadata mismatch; choose a new output: {args.output}"
            )
    else:
        atomic_json(metadata_path, metadata)
    repair_truncated_jsonl_tail(args.output)
    completed = load_successful_qids(args.output, fingerprint)
    selected = eligible_rows(rows, args.seed, args.max_samples)
    pending = [row for row in selected if str(row["qid"]) not in completed]
    print(
        f"parameter-metric fingerprint={fingerprint[:12]} "
        f"eligible={len(pending)}/{len(selected)}",
        flush=True,
    )
    if not pending:
        return

    adapter = LlavaParameterMetricAdapter()
    projector = adapter.model.get_model().mm_projector
    transport = ParameterMetricTransport(projector, centers)
    errors = 0
    try:
        with args.output.open("a") as output:
            for sample in tqdm(pending, desc="projector parameter metric"):
                try:
                    image_path = resolve_image(sample.get("img_name", ""))
                    with Image.open(image_path) as source:
                        image = resize_image(
                            source.convert("RGB"), args.max_image_side
                        )
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    image_tensor = adapter.prepare_image_tensor(image)
                    image_size = image.size
                    transport.reset_sample()
                    results = [
                        adapter.forward_binary_exact(
                            image_tensor, image_size, prompt, labels
                        )
                    ]
                    diagnostics: list[dict | None] = [None]
                    for arm in ARMS[1:]:
                        with transport.apply(arm):
                            result = adapter.forward_binary_exact(
                                image_tensor, image_size, prompt, labels
                            )
                        if (
                            transport.last_record is None
                            or transport.last_record.arm != arm
                        ):
                            raise RuntimeError(f"missing hook diagnostics for {arm}")
                        results.append(result)
                        diagnostics.append(transport.last_record.diagnostics)
                    row = {
                        "cache_version": VERSION,
                        "protocol_version": PROTOCOL_VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample.get("img_name", ""),
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "arms": list(ARMS),
                        "surface_logits": [
                            item.surface_logits.tolist() for item in results
                        ],
                        "exact_next_token_nll": [
                            item.exact_next_token_nll.tolist() for item in results
                        ],
                        "exact_label_token_ids": results[0].exact_label_token_ids,
                        "language_features": encode_array(
                            np.stack(
                                [item.language_feature for item in results]
                            )
                        ),
                        "transform_diagnostics": diagnostics,
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        "cache_version": VERSION,
                        "protocol_version": PROTOCOL_VERSION,
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
    finally:
        adapter.close()
    print(f"finished rows={len(pending)} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
