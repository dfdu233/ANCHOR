#!/usr/bin/env python3
"""Collect source-only paired activations for an ANCHOR-Null pilot.

This collector deliberately has no target-dataset argument.  It samples one
question per (source domain, image hash), applies one frozen, formal X-ray
Fourier center, and stores the quantities needed by ``domain_halluspace``.
Fitting and target evaluation are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.cache import encode_array
from corrected_sgta.frequency_alignment_release2 import (
    feddg_frequency_interpolation_release2,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.provenance_v3 import model_identity
from corrected_sgta.source_bank_v2 import SOURCE_BANK_VERSION, sha256_file
from corrected_sgta.train_rule_dg_adapter import (
    IGNORE_INDEX,
    build_teacher_forcing,
    canonical_answer,
    extract_question_only,
    process_image,
    rule_no_reference_prompt,
)


VERSION = "rule-source-halluspace-collection-v1"
CONV_MODE = "vicuna_v1"
ALLOWED_CENTER_IDS = ("mimic_cxr_leaksafe", "pubmedvision_xray_formal")
DEFAULT_CENTER_ID = "pubmedvision_xray_formal"
LOW_FREQUENCY_RATIO = 0.03
SOURCE_RATIO = 0.0
PROJECTED_WIDTH = 4096


class CollectionError(RuntimeError):
    """Raised when a source-only collection contract is violated."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise CollectionError("source-dev must be a JSON array of objects")
    return value


def row_question_answer(row: Mapping[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise CollectionError(f"invalid source conversation: {row.get('id')}")
    question = extract_question_only(conversations[0].get("value"))
    answer = canonical_answer(conversations[1].get("value"))
    return rule_no_reference_prompt(question), answer


def select_source_rows(
    rows: list[dict[str, Any]],
    limit_per_domain: int,
    seed: int,
    domains: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministically without consulting labels or target data."""

    if limit_per_domain <= 0:
        raise CollectionError("limit-per-domain must be positive")
    requested_domains = None if domains is None else set(domains)
    if requested_domains is not None and not requested_domains:
        raise CollectionError("domains must be omitted or contain at least one domain")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        domain = str(row.get("source_domain", "")).strip()
        if requested_domains is not None and domain not in requested_domains:
            continue
        image_hash = str(row.get("image_sha256", "")).strip().lower()
        image = Path(str(row.get("image", "")))
        if not domain or len(image_hash) != 64:
            raise CollectionError(
                f"row lacks source_domain/image_sha256: {row.get('id')}"
            )
        if not image.is_absolute():
            raise CollectionError(f"source image must be absolute: {image}")
        grouped.setdefault(domain, {}).setdefault(image_hash, []).append(row)
    if requested_domains is not None:
        missing = requested_domains - set(grouped)
        if missing:
            raise CollectionError(f"requested source domains are absent: {sorted(missing)}")

    selected: list[dict[str, Any]] = []
    for domain, by_image in sorted(grouped.items()):
        image_hashes = sorted(
            by_image,
            key=lambda value: stable_sha256([seed, domain, value]),
        )[:limit_per_domain]
        for image_hash in image_hashes:
            candidates = sorted(
                by_image[image_hash],
                key=lambda row: stable_sha256(
                    [
                        seed,
                        domain,
                        image_hash,
                        row.get("id"),
                        row.get("source_id"),
                    ]
                ),
            )
            selected.append(candidates[0])
    return sorted(
        selected,
        key=lambda row: (
            str(row["source_domain"]),
            stable_sha256([seed, row["source_domain"], row["image_sha256"]]),
        ),
    )


def selected_image_identity(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    identities = []
    for row in rows:
        image = Path(str(row["image"]))
        if not image.is_file():
            raise FileNotFoundError(image)
        blob_hash = sha256_file(image)
        expected = str(row.get("image_blob_sha256", ""))
        if expected and blob_hash != expected:
            raise CollectionError(f"source image blob hash mismatch: {image}")
        identities.append(
            {
                "record_id": record_id(row),
                "source_domain": str(row["source_domain"]),
                "image_sha256": str(row["image_sha256"]),
                "image_blob_sha256": blob_hash,
                "image": str(image),
            }
        )
    return identities


def record_id(row: Mapping[str, Any]) -> str:
    return stable_sha256(
        [
            str(row.get("source_domain")),
            str(row.get("image_sha256")),
            str(row.get("id")),
        ]
    )


def load_fixed_centers(
    source_bank_path: Path, source_id: str = DEFAULT_CENTER_ID
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(source_bank_path.read_text(encoding="utf-8"))
    if manifest.get("source_bank_version") != SOURCE_BANK_VERSION:
        raise CollectionError(f"unsupported source bank: {source_bank_path}")
    entries = {str(item.get("source_id")): item for item in manifest.get("entries", [])}
    if source_id not in ALLOWED_CENTER_IDS:
        raise CollectionError(f"source-id is not an allowed fixed center: {source_id}")
    entry = entries.get(source_id)
    if entry is None:
        raise CollectionError(f"source bank lacks fixed center {source_id}")
    if not entry.get("formal") or str(entry.get("modality")) != "xray":
        raise CollectionError(f"center is not a formal X-ray source: {source_id}")
    amplitude_path = Path(str(entry.get("amplitude_file", "")))
    if not amplitude_path.is_file():
        raise FileNotFoundError(amplitude_path)
    observed = sha256_file(amplitude_path)
    if observed != entry.get("amplitude_sha256"):
        raise CollectionError(f"amplitude hash mismatch: {source_id}")
    amplitude = np.load(amplitude_path, allow_pickle=False)
    if amplitude.ndim not in (2, 3) or not np.isfinite(amplitude).all():
        raise CollectionError(f"invalid amplitude center: {source_id}")
    identity = {
        "source_id": source_id,
        "formal": True,
        "modality": "xray",
        "amplitude_file": str(amplitude_path.resolve()),
        "amplitude_sha256": observed,
        "shape": list(amplitude.shape),
    }
    return np.asarray(amplitude), identity


class PostProjectorCapture(AbstractContextManager):
    """Replace post-projector tokens by a leaf while retaining their values."""

    def __init__(self, model: Any, *, require_gradient: bool):
        self.projector = model.get_model().mm_projector
        self.require_gradient = require_gradient
        self.output: torch.Tensor | None = None
        self.handle: Any = None

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> torch.Tensor:
        if not isinstance(output, torch.Tensor) or output.ndim not in (2, 3):
            raise CollectionError("mm_projector must return [tokens,width] or [batch,tokens,width]")
        if output.shape[-1] != PROJECTED_WIDTH:
            raise CollectionError(
                f"unexpected projected width {output.shape[-1]} != {PROJECTED_WIDTH}"
            )
        leaf = output.detach()
        if self.require_gradient:
            leaf = leaf.requires_grad_(True)
        self.output = leaf
        return leaf

    def __enter__(self):
        self.output = None
        self.handle = self.projector.register_forward_hook(self._capture)
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return False


def pooled_projected_activation(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3:
        value = tokens.mean(dim=(0, 1))
    elif tokens.ndim == 2:
        value = tokens.mean(dim=0)
    else:
        raise CollectionError("projected tokens must be rank two or three")
    if value.shape != (PROJECTED_WIDTH,) or not torch.isfinite(value).all():
        raise CollectionError("invalid pooled post-projector activation")
    return value.detach().float().cpu()


def uniform_shift_gradient(loss: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Gradient w.r.t. one vector broadcast uniformly to all visual tokens."""

    gradient = torch.autograd.grad(loss, tokens, retain_graph=False)[0]
    axes = tuple(range(gradient.ndim - 1))
    value = gradient.float().sum(dim=axes)
    if value.shape != (PROJECTED_WIDTH,) or not torch.isfinite(value).all():
        raise CollectionError("invalid uniform-token-shift gradient")
    return value.detach().cpu()


def sequence_measurement(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    with_gradient: bool,
) -> tuple[float, torch.Tensor, torch.Tensor | None]:
    """Return mean correct-sequence NLL, pooled activation, and optional gradient."""

    model = adapter.model
    model.zero_grad(set_to_none=True)
    ids = input_ids.to(model.device)
    targets = labels.to(model.device)
    pixels = process_image(adapter, image)
    with PostProjectorCapture(model, require_gradient=with_gradient) as capture:
        _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = (
            model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                None,
                None,
                targets,
                pixels,
                image_sizes=[image.size],
            )
        )
        if capture.output is None:
            raise CollectionError("post-projector hook was not called")
        output = model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        vocabulary_weight = model.get_output_embeddings().weight
        logits = output.last_hidden_state.to(vocabulary_weight.dtype) @ vocabulary_weight.T
        if expanded_labels is None:
            raise CollectionError("multimodal preparation returned no labels")
        shifted_logits = logits[:, :-1].float().contiguous()
        shifted_labels = expanded_labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        if not torch.isfinite(loss):
            raise CollectionError("non-finite correct-sequence NLL")
        activation = pooled_projected_activation(capture.output)
        gradient = (
            uniform_shift_gradient(loss, capture.output) if with_gradient else None
        )
    return float(loss.detach().cpu()), activation, gradient


def repair_jsonl_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    stripped = data.rstrip(b"\r\n")
    if not stripped:
        return
    start = stripped.rfind(b"\n") + 1
    try:
        json.loads(stripped[start:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(start)
        return
    if not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")


def load_tensor_cache(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        return {"version": VERSION, "fingerprint": fingerprint, "records": {}}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != VERSION
        or payload.get("fingerprint") != fingerprint
        or not isinstance(payload.get("records"), dict)
    ):
        raise CollectionError("tensor cache fingerprint/schema mismatch")
    return payload


def successful_record_ids(
    path: Path, fingerprint: str, tensor_records: Mapping[str, Any]
) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("fingerprint") != fingerprint:
            raise CollectionError("JSONL cache fingerprint mismatch")
        if row.get("status") == "ok":
            identity = str(row.get("record_id"))
            if identity not in tensor_records:
                raise CollectionError(f"JSONL success lacks tensor record: {identity}")
            completed.add(identity)
    return completed


def code_identity(project_root: Path) -> dict[str, str]:
    names = (
        "corrected_sgta/collect_rule_source_halluspace.py",
        "corrected_sgta/domain_halluspace.py",
        "corrected_sgta/frequency_alignment_release2.py",
        "corrected_sgta/models_alignment.py",
        "corrected_sgta/train_rule_dg_adapter.py",
        "corrected_sgta/source_bank_v2.py",
    )
    return {name: sha256_file(project_root / name) for name in names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dev", required=True, type=Path)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor-output", type=Path)
    parser.add_argument(
        "--source-id",
        choices=ALLOWED_CENTER_IDS,
        default=DEFAULT_CENTER_ID,
        help="Use exactly one formal X-ray center per source image.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional source_domain allow-list; omitted means all source domains.",
    )
    parser.add_argument("--limit-per-domain", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-image-side", type=int, default=384)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_image_side <= 0:
        raise CollectionError("max-image-side must be positive")
    tensor_output = args.tensor_output or args.output.with_suffix(".pt")
    rows = load_json_rows(args.source_dev)
    selected = select_source_rows(
        rows, args.limit_per_domain, args.seed, domains=args.domains
    )
    images = selected_image_identity(selected)
    center, center_identity = load_fixed_centers(args.source_bank, args.source_id)
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "version": VERSION,
        "scope": "source-only; no target dataset, target labels, or target selection",
        "model": "llava",
        "model_identity": model_identity("llava"),
        "conv_mode": CONV_MODE,
        "code_identity": code_identity(project_root),
        "source_dev": str(args.source_dev.resolve()),
        "source_dev_sha256": sha256_file(args.source_dev),
        "source_bank": str(args.source_bank.resolve()),
        "source_bank_sha256": sha256_file(args.source_bank),
        "fixed_center": center_identity,
        "selection": {
            "unit": "one QA per (source_domain,image_sha256)",
            "label_free": True,
            "limit_per_domain": args.limit_per_domain,
            "seed": args.seed,
            "domains": "all" if args.domains is None else sorted(args.domains),
            "selected": images,
        },
        "fourier": {
            "implementation": "frequency_alignment_release2",
            "low_frequency_ratio": LOW_FREQUENCY_RATIO,
            "source_ratio": SOURCE_RATIO,
        },
        "activation": "mean post-projector visual tokens; width 4096",
        "clinical_gradient": (
            "gradient of mean full-correct-sequence NLL with respect to a "
            "uniform post-projector token shift; sum of raw token gradients"
        ),
        "nll": "teacher-forced complete canonical Yes./No. sequence; mean token NLL",
        "max_image_side": args.max_image_side,
    }
    fingerprint = stable_sha256(config)
    metadata = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "output": str(args.output.resolve()),
        "tensor_output": str(tensor_output.resolve()),
        "requested_records": len(selected),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_output.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise CollectionError("metadata fingerprint mismatch; choose a new output")
    elif args.output.exists() or tensor_output.exists():
        raise CollectionError("cache exists without matching metadata")
    else:
        atomic_json(meta_path, metadata)
    if not args.resume and (args.output.exists() or tensor_output.exists()):
        raise CollectionError("output exists; use --resume for the identical fingerprint")

    repair_jsonl_tail(args.output)
    tensor_cache = load_tensor_cache(tensor_output, fingerprint)
    completed = successful_record_ids(
        args.output, fingerprint, tensor_cache["records"]
    )
    pending = [row for row in selected if record_id(row) not in completed]
    if not pending:
        print(f"{VERSION} fingerprint={fingerprint[:12]} complete", flush=True)
        return

    adapter = LlavaMedAlignmentAdapter(conv_mode=CONV_MODE)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    try:
        with args.output.open("a", encoding="utf-8") as stream:
            for row in tqdm(pending, desc="source halluspace collection"):
                identity = record_id(row)
                try:
                    prompt, answer = row_question_answer(row)
                    input_ids, labels = build_teacher_forcing(adapter, prompt, answer)
                    token_count = int(labels.ne(IGNORE_INDEX).sum())
                    with Image.open(row["image"]) as source:
                        clean_image = resize_image(source.convert("RGB"), args.max_image_side)
                    clean_nll, clean_activation, clinical_gradient = sequence_measurement(
                        adapter,
                        clean_image,
                        input_ids,
                        labels,
                        with_gradient=True,
                    )
                    assert clinical_gradient is not None
                    shifted_image = feddg_frequency_interpolation_release2(
                        clean_image,
                        center,
                        low_frequency_ratio=LOW_FREQUENCY_RATIO,
                        source_ratio=SOURCE_RATIO,
                    )
                    with torch.no_grad():
                        shifted_nll, shifted_activation, _ = sequence_measurement(
                            adapter,
                            shifted_image,
                            input_ids,
                            labels,
                            with_gradient=False,
                        )
                    tensor_cache["records"][identity] = {
                        "clean_activation": clean_activation,
                        "clinical_gradient": clinical_gradient,
                        "shifted_activation": shifted_activation,
                    }
                    atomic_torch_save(tensor_output, tensor_cache)
                    output_row = {
                        "version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "ok",
                        "record_id": identity,
                        "id": str(row.get("id")),
                        "source_id": str(row.get("source_id", row.get("id"))),
                        "source_domain": str(row["source_domain"]),
                        "image": str(row["image"]),
                        "image_sha256": str(row["image_sha256"]),
                        "gt_label": answer[:-1].lower(),
                        "answer": answer,
                        "answer_token_count": token_count,
                        "clean_activation": encode_array(
                            clean_activation.numpy(), dtype="float32"
                        ),
                        "shifted_activation": encode_array(
                            shifted_activation.numpy(), dtype="float32"
                        ),
                        "clinical_gradient": encode_array(
                            clinical_gradient.numpy(), dtype="float32"
                        ),
                        "clean_correct_nll": clean_nll,
                        "shifted_correct_nll": shifted_nll,
                        "clean_correct_sequence_mean_nll": clean_nll,
                        "clean_correct_sequence_total_nll": clean_nll * token_count,
                        "shifted_correct_sequence_total_nll": shifted_nll * token_count,
                        "view_source_id": args.source_id,
                        "transform": {
                            "implementation": "frequency_alignment_release2",
                            "low_frequency_ratio": LOW_FREQUENCY_RATIO,
                            "source_ratio": SOURCE_RATIO,
                        },
                        "tensor_key": identity,
                    }
                except Exception as error:
                    output_row = {
                        "version": VERSION,
                        "fingerprint": fingerprint,
                        "status": "error",
                        "record_id": identity,
                        "source_id": str(row.get("source_id", row.get("id"))),
                        "source_domain": str(row.get("source_domain")),
                        "image_sha256": str(row.get("image_sha256")),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
