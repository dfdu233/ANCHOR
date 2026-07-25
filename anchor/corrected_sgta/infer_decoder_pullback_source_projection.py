"""Single-arm RULE pilot for decoder-pullback source projection (DPSP)."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.evaluate_medheval_answers import rule_pope_prediction
from corrected_sgta.fisher_source_transport import (
    decoder_pullback_source_projection,
    pca_geometry,
)
from corrected_sgta.infer_fisher_source_transport import (
    arm_evidence,
    output_jacobian,
    raw_tokens,
    shifted_projector,
)
from corrected_sgta.infer_projector_parameter_metric import (
    LlavaParameterMetricAdapter,
    checked_feature_file,
)
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    ProtocolError,
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

VERSION = "decoder-pullback-source-projection-rule-v2"
ARMS = ("original", "dpsp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--xray-features", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--explained-variance", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_source_geometry(path: Path, explained_variance: float):
    arrays, metadata = checked_feature_file(path, {"exact_raw": (64, 1024)})
    mean, basis = pca_geometry(arrays["exact_raw"], explained_variance)
    return mean, basis, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "upstream_fingerprint": metadata.get("fingerprint"),
        "n": int(arrays["exact_raw"].shape[0]),
        "dimension": int(arrays["exact_raw"].shape[1]),
        "rank": int(basis.shape[1]),
    }


def rule_prompt(sample: dict) -> str:
    return (
        str(sample["question"]).strip()
        + " Please answer the question based on the image and choose from the "
        "following two options: [yes, no]."
    )


@torch.inference_mode()
def decode(
    adapter: LlavaParameterMetricAdapter,
    image_tensor,
    image_size: tuple[int, int],
    prompt: str,
    delta: torch.Tensor | None,
    max_new_tokens: int,
) -> str:
    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    projector = adapter.model.get_model().mm_projector
    with shifted_projector(projector, delta):
        output_ids = adapter.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return adapter.tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()


def eligible_rows(rows: list[dict], maximum: int) -> list[dict]:
    selected = []
    for row in rows:
        try:
            if task_kind(row) != "binary" or labels_for_sample(row) != ("Yes", "No"):
                continue
            ground_truth_index(row)
            if resolve_image(row.get("img_name", "")) is None:
                continue
        except ProtocolError:
            continue
        selected.append(row)
        if maximum and len(selected) >= maximum:
            break
    return selected


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    mean, basis, source = load_source_geometry(
        args.xray_features, args.explained_variance
    )
    config = {
        "version": VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_identity": model_identity("llava"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "source": source,
        "max_samples": args.max_samples,
        "explained_variance": args.explained_variance,
        "max_new_tokens": args.max_new_tokens,
        "conversation_mode": "vicuna_v1",
        "prompt_style": "RULE model_vqa_iuxray no-reference branch",
        "operator": "r=P_perp(mu-z); d=P_perp J^T F J r; delta=<r,d>/(||d||^2+eps)d",
        "arms": list(ARMS),
        "labels_used_for_direction": False,
        "ground_truth_used_for_direction": False,
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
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
    selected = eligible_rows(rows, args.max_samples)
    pending = [row for row in selected if str(row["qid"]) not in completed]
    print(
        f"DPSP fingerprint={fingerprint[:12]} pending={len(pending)}/{len(selected)}",
        flush=True,
    )
    if not pending:
        return

    adapter = LlavaParameterMetricAdapter(conv_mode="vicuna_v1")
    device = adapter.model.device
    source_mean = torch.as_tensor(mean, device=device)
    tangent_basis = torch.as_tensor(basis, device=device)
    errors = 0
    try:
        with args.output.open("a") as output:
            for sample in tqdm(pending, desc="DPSP RULE pilot"):
                try:
                    image_path = resolve_image(sample["img_name"])
                    with Image.open(image_path) as source_image:
                        image = source_image.convert("RGB")
                    prompt = rule_prompt(sample)
                    labels = labels_for_sample(sample)
                    label_ids = []
                    for label in labels:
                        ids = adapter.tokenizer.encode(label, add_special_tokens=False)
                        if len(ids) != 1:
                            raise RuntimeError(
                                f"label is not one token: {label!r} -> {ids}"
                            )
                        label_ids.append(int(ids[0]))
                    image_tensor = adapter.prepare_image_tensor(image)
                    pooled = raw_tokens(adapter, image_tensor).mean(dim=0)
                    jacobian, probabilities = output_jacobian(
                        adapter,
                        image_tensor,
                        image.size,
                        prompt,
                        label_ids,
                        pooled.numel(),
                    )
                    delta, diagnostics = decoder_pullback_source_projection(
                        jacobian,
                        probabilities,
                        tangent_basis,
                        source_mean - pooled,
                    )
                    logits, arm_probabilities, decoded = [], [], []
                    for arm_delta in (None, delta):
                        arm_logits, arm_probs = arm_evidence(
                            adapter,
                            image_tensor,
                            image.size,
                            prompt,
                            label_ids,
                            arm_delta,
                        )
                        logits.append(arm_logits)
                        arm_probabilities.append(arm_probs)
                        decoded.append(
                            decode(
                                adapter,
                                image_tensor,
                                image.size,
                                prompt,
                                arm_delta,
                                args.max_new_tokens,
                            )
                        )
                    row = {
                        "cache_version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "qid": sample["qid"],
                        "img_name": sample["img_name"],
                        "labels": list(labels),
                        "gt_index": ground_truth_index(sample),
                        "arms": list(ARMS),
                        "label_logits": logits,
                        "label_probabilities": arm_probabilities,
                        "decoded": decoded,
                        "rule_predictions": [
                            rule_pope_prediction(value) for value in decoded
                        ],
                        "direction_diagnostics": diagnostics,
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
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
