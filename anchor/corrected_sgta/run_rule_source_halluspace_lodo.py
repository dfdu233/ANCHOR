#!/usr/bin/env python3
"""Source-only LODO evaluation of the compact ANCHOR-Null halluspace.

The runner consumes the paired source collector, fits every held-out-domain
fold using only the other source domains, and evaluates complete ``Yes.`` and
``No.`` sequence NLL on clean and exactly reconstructed shifted images.  All
mathematical hyperparameters are frozen; no target dataset is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from corrected_sgta.collect_rule_source_halluspace import (
    CONV_MODE,
    LOW_FREQUENCY_RATIO,
    SOURCE_RATIO,
    atomic_json,
    atomic_torch_save,
    load_fixed_centers,
    record_id,
    repair_jsonl_tail,
    row_question_answer,
)
from corrected_sgta.cache import decode_array
from corrected_sgta.domain_halluspace import (
    DomainHalluspace,
    RankDiagnostics,
    fit_domain_halluspace,
)
from corrected_sgta.frequency_alignment_release2 import (
    feddg_frequency_interpolation_release2,
)
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models import LLAVA_PATH
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_dg_adapter_fingerprint_v3 import tree_identity
from corrected_sgta.source_bank_v2 import sha256_file
from corrected_sgta.train_rule_dg_adapter import (
    IGNORE_INDEX,
    build_teacher_forcing,
    canonical_answer,
    process_image,
)


VERSION = "rule-source-halluspace-lodo-v1"
PARALLEL_REPETITIONS = 32
PARALLEL_QUANTILE = 0.95
SHRINKAGE_STRENGTH = 1.0
FISHER_WEIGHT = 1.0
RIDGE_SCALE = 0.01


class HalluspaceRunnerError(RuntimeError):
    """Fail-closed data, protocol, or runtime error."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _vector(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, Mapping) and "values" in value:
        value = value["values"]
    elif isinstance(value, Mapping) and {"dtype", "shape", "data"} <= set(value):
        value = decode_array(dict(value))
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() < 1 or not torch.isfinite(tensor).all():
        raise HalluspaceRunnerError(f"{name} must encode one finite vector")
    return tensor.cpu()


def _canonical_label(value: Any) -> str:
    answer = canonical_answer(value)
    return answer[:-1].lower()


def _source_rows_by_record(meta: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_dev = Path(str(meta.get("config", {}).get("source_dev", "")))
    if not source_dev.is_file():
        return {}
    rows = json.loads(source_dev.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise HalluspaceRunnerError("collector source_dev must be a JSON array")
    return {record_id(row): row for row in rows}


def load_collector_pairs(
    jsonl_path: Path,
    tensor_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load actual collector output or the agreed direct-vector JSONL schema."""

    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    meta_path = meta_path or jsonl_path.with_suffix(jsonl_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    tensor_path = tensor_path or jsonl_path.with_suffix(".pt")
    tensor_records: Mapping[str, Any] = {}
    if tensor_path.is_file():
        tensor_payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor_payload, dict) or not isinstance(
            tensor_payload.get("records"), dict
        ):
            raise HalluspaceRunnerError("collector tensor cache has invalid schema")
        if meta and tensor_payload.get("fingerprint") != meta.get("fingerprint"):
            raise HalluspaceRunnerError("collector JSON/tensor fingerprint mismatch")
        tensor_records = tensor_payload["records"]
    source_rows = _source_rows_by_record(meta)
    pairs: list[dict[str, Any]] = []
    collector_fingerprints = set()
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status", "ok") != "ok":
            continue
        if row.get("fingerprint"):
            collector_fingerprints.add(str(row["fingerprint"]))
        identity = str(row.get("record_id", row.get("id", "")))
        domain = str(row.get("source_domain", "")).strip()
        image = Path(str(row.get("image", "")))
        if not identity or not domain or not image.is_absolute():
            raise HalluspaceRunnerError(f"invalid collector identity at line {line_number}")
        source_row = source_rows.get(identity)
        if "question" in row:
            question = str(row["question"])
            label = _canonical_label(row.get("gt_label", row.get("answer")))
        elif source_row is not None:
            question, answer = row_question_answer(source_row)
            label = _canonical_label(answer)
        else:
            raise HalluspaceRunnerError(
                f"collector row {identity} lacks question and recoverable source_dev"
            )
        tensor_record = tensor_records.get(str(row.get("tensor_key", identity)), {})
        clean_activation = _vector(
            row.get("clean_activation", tensor_record.get("clean_activation")),
            "clean_activation",
        )
        clinical_gradient = _vector(
            row.get("clinical_gradient", tensor_record.get("clinical_gradient")),
            "clinical_gradient",
        )
        direct_shift_key = (
            "shift_activation"
            if "shift_activation" in row
            else "shifted_activation" if "shifted_activation" in row else None
        )
        if direct_shift_key is not None:
            transform = dict(row.get("view_config", row.get("transform", {})))
            transform.setdefault(
                "source_id", str(row.get("view_source_id", "direct"))
            )
            views = [
                {
                    "source_id": transform["source_id"],
                    "activation": _vector(row[direct_shift_key], "shifted_activation"),
                    "shift_nll": float(
                        row.get("shift_correct_nll", row.get("shifted_correct_nll"))
                    ),
                    "clean_nll": float(row["clean_correct_nll"]),
                    "view_config": transform,
                }
            ]
        else:
            shifted_tensors = tensor_record.get("shifted_activations", {})
            views = []
            for view in row.get("shifted_views", []):
                source_id = str(view.get("source_id", ""))
                if source_id not in shifted_tensors:
                    raise HalluspaceRunnerError(
                        f"collector row {identity} lacks tensor for {source_id}"
                    )
                views.append(
                    {
                        "source_id": source_id,
                        "activation": _vector(
                            shifted_tensors[source_id], "shifted_activation"
                        ),
                        "shift_nll": float(view["correct_sequence_mean_nll"]),
                        "clean_nll": float(row["clean_correct_sequence_mean_nll"]),
                        "view_config": {
                            "source_id": source_id,
                            "low_frequency_ratio": LOW_FREQUENCY_RATIO,
                            "source_ratio": SOURCE_RATIO,
                        },
                    }
                )
        if not views:
            raise HalluspaceRunnerError(f"collector row {identity} has no shifted views")
        width = clean_activation.numel()
        if clinical_gradient.numel() != width:
            raise HalluspaceRunnerError(f"activation/gradient width mismatch for {identity}")
        for view in views:
            if view["activation"].numel() != width:
                raise HalluspaceRunnerError(f"shift width mismatch for {identity}")
            if not math.isfinite(view["shift_nll"]) or not math.isfinite(view["clean_nll"]):
                raise HalluspaceRunnerError(f"non-finite NLL for {identity}")
            pairs.append(
                {
                    "eval_id": stable_sha256([identity, view["source_id"]]),
                    "record_id": identity,
                    "source_domain": domain,
                    "image": str(image),
                    "image_sha256": str(row.get("image_sha256", "")),
                    "question": question,
                    "gt_label": label,
                    "clean_activation": clean_activation,
                    "shift_activation": view["activation"],
                    "clean_correct_nll": view["clean_nll"],
                    "shift_correct_nll": view["shift_nll"],
                    "clinical_gradient": clinical_gradient,
                    "view_config": view["view_config"],
                }
            )
    if len(collector_fingerprints) > 1:
        raise HalluspaceRunnerError("collector JSONL mixes fingerprints")
    if meta and collector_fingerprints and meta.get("fingerprint") not in collector_fingerprints:
        raise HalluspaceRunnerError("collector metadata/JSONL fingerprint mismatch")
    if not pairs:
        raise HalluspaceRunnerError("collector contains no successful pairs")
    return pairs, meta


def fit_lodo_folds(
    pairs: list[dict[str, Any]], *, seed: int = 0
) -> dict[str, dict[str, Any]]:
    """Fit each fold exclusively from the other source domains."""

    domains = sorted({row["source_domain"] for row in pairs})
    if len(domains) < 2:
        raise HalluspaceRunnerError("LODO requires at least two source domains")
    folds = {}
    for heldout in domains:
        training = [row for row in pairs if row["source_domain"] != heldout]
        clean = torch.stack([row["clean_activation"] for row in training])
        shifted = torch.stack([row["shift_activation"] for row in training])
        gradients = torch.stack([row["clinical_gradient"] for row in training])
        delta = torch.tensor(
            [row["shift_correct_nll"] - row["clean_correct_nll"] for row in training],
            dtype=torch.float32,
        )
        harmful = delta.clamp_min(0.0)
        stable = delta.le(0.0).float()
        ridge = RIDGE_SCALE * float(gradients.double().square().mean())
        if not math.isfinite(ridge) or ridge <= 0:
            raise HalluspaceRunnerError(f"fold {heldout} has non-positive gradient ridge")
        fitted = fit_domain_halluspace(
            clean,
            shifted,
            harmful,
            gradients,
            stable_weights=stable,
            fisher_weight=FISHER_WEIGHT,
            fisher_ridge=ridge,
            parallel_repetitions=PARALLEL_REPETITIONS,
            parallel_quantile=PARALLEL_QUANTILE,
            shrinkage_strength=SHRINKAGE_STRENGTH,
            seed=seed,
        )
        unique_clean = {}
        for row in training:
            unique_clean.setdefault(row["record_id"], row["clean_activation"])
        center = torch.stack(list(unique_clean.values())).double().mean(dim=0)
        folds[heldout] = {
            "heldout_domain": heldout,
            "training_domains": sorted({row["source_domain"] for row in training}),
            "training_pair_count": len(training),
            "training_image_count": len(unique_clean),
            "ridge": ridge,
            "center": center,
            "fitted": fitted,
        }
    return folds


def fold_to_payload(fold: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in fold.items() if key != "fitted"}
    result["fitted"] = asdict(fold["fitted"])
    return result


def fold_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    fitted = dict(result["fitted"])
    fitted["rank"] = RankDiagnostics(**fitted["rank"])
    result["fitted"] = DomainHalluspace(**fitted)
    return result


def source_rank_gate(folds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected = {
        domain: int(fold["fitted"].rank.selected_rank)
        for domain, fold in sorted(folds.items())
    }
    passed = bool(selected) and all(rank > 0 for rank in selected.values())
    return {
        "status": "passed" if passed else "failed",
        "require_all_folds_positive_rank": True,
        "fold_selected_rank": selected,
    }


def uniform_projection_delta(
    pooled: torch.Tensor,
    fitted: DomainHalluspace,
    center: torch.Tensor,
) -> torch.Tensor:
    """Return the low-rank correction without materializing a full projector."""

    if pooled.ndim != 1 or pooled.numel() != fitted.basis.shape[0]:
        raise HalluspaceRunnerError("pooled activation width mismatch")
    dtype, device = pooled.dtype, pooled.device
    centered = pooled.float() - center.to(device=device, dtype=torch.float32)
    basis = fitted.basis.to(device=device, dtype=torch.float32)
    dual = fitted.dual_basis.to(device=device, dtype=torch.float32)
    shrinkage = fitted.shrinkage.to(device=device, dtype=torch.float32)
    correction = -((centered @ dual) * shrinkage) @ basis.T
    if not torch.isfinite(correction).all():
        raise HalluspaceRunnerError("non-finite uniform projection correction")
    return correction.to(dtype=dtype)


class PostProjectorUniformCorrection(AbstractContextManager):
    def __init__(self, model: Any, fitted: DomainHalluspace, center: torch.Tensor):
        self.projector = model.get_model().mm_projector
        self.fitted = fitted
        self.center = center
        self.handle: Any = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> torch.Tensor:
        if not isinstance(output, torch.Tensor) or output.ndim not in (2, 3):
            raise HalluspaceRunnerError("mm_projector output must be token-by-width")
        axes = tuple(range(output.ndim - 1))
        pooled = output.mean(dim=axes)
        correction = uniform_projection_delta(pooled, self.fitted, self.center)
        return output + correction.reshape((1,) * (output.ndim - 1) + (-1,))

    def __enter__(self):
        self.handle = self.projector.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return False


@torch.inference_mode()
def complete_sequence_nll(
    adapter: LlavaMedAlignmentAdapter,
    image: Image.Image,
    prompt: str,
    answer: str,
    fitted: DomainHalluspace | None = None,
    center: torch.Tensor | None = None,
) -> float:
    input_ids, labels = build_teacher_forcing(adapter, prompt, answer)
    context = (
        nullcontext()
        if fitted is None
        else PostProjectorUniformCorrection(adapter.model, fitted, center)
    )
    ids = input_ids.to(adapter.model.device)
    targets = labels.to(adapter.model.device)
    pixels = process_image(adapter, image)
    with context:
        _, position_ids, attention_mask, _, embeds, expanded_labels = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                ids, None, None, None, targets, pixels, image_sizes=[image.size]
            )
        )
        output = adapter.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=embeds,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        weight = adapter.model.get_output_embeddings().weight
        logits = output.last_hidden_state.to(weight.dtype) @ weight.T
    if expanded_labels is None:
        raise HalluspaceRunnerError("teacher forcing returned no expanded labels")
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        expanded_labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    if not torch.isfinite(loss):
        raise HalluspaceRunnerError("non-finite complete-sequence NLL")
    return float(loss)


def prediction(nll: Mapping[str, float]) -> str:
    return "yes" if float(nll["yes"]) <= float(nll["no"]) else "no"


def summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise HalluspaceRunnerError("cannot summarize empty evaluations")
    groups = {}
    for scope in ("shifted", "clean"):
        base_correct = projected_correct = rescues = harms = 0
        base_nll = projected_nll = 0.0
        for row in rows:
            label = row["gt_label"]
            baseline = row[scope]["baseline_nll"]
            projected = row[scope]["projected_nll"]
            base_ok = prediction(baseline) == label
            projected_ok = prediction(projected) == label
            base_correct += int(base_ok)
            projected_correct += int(projected_ok)
            rescues += int(not base_ok and projected_ok)
            harms += int(base_ok and not projected_ok)
            base_nll += float(baseline[label])
            projected_nll += float(projected[label])
        count = len(rows)
        groups[scope] = {
            "n": count,
            "baseline_accuracy": base_correct / count,
            "projected_accuracy": projected_correct / count,
            "delta_pp": 100.0 * (projected_correct - base_correct) / count,
            "rescues": rescues,
            "harms": harms,
            "baseline_correct_sequence_nll": base_nll / count,
            "projected_correct_sequence_nll": projected_nll / count,
            "correct_sequence_nll_delta": (projected_nll - base_nll) / count,
        }
    clean = groups["clean"]
    checks = {
        "zero_clean_harms": clean["harms"] == 0,
        "clean_accuracy_nondecreasing": clean["delta_pp"] >= 0.0,
        "clean_correct_nll_nonincreasing": clean["correct_sequence_nll_delta"] <= 0.0,
    }
    return {
        **groups,
        "safety_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
        },
    }


def completed_ids(path: Path, fingerprint: str) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("fingerprint") != fingerprint:
            raise HalluspaceRunnerError("evaluation JSONL fingerprint mismatch")
        if row.get("status") == "ok":
            completed.add(str(row["eval_id"]))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector-jsonl", required=True, type=Path)
    parser.add_argument("--collector-tensors", type=Path)
    parser.add_argument("--collector-meta", type=Path)
    parser.add_argument("--model-path", type=Path, default=LLAVA_PATH)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, collector_meta = load_collector_pairs(
        args.collector_jsonl, args.collector_tensors, args.collector_meta
    )
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "version": VERSION,
        "scope": "source-only LODO; no target dataset or target selection",
        "collector_jsonl": str(args.collector_jsonl.resolve()),
        "collector_jsonl_sha256": sha256_file(args.collector_jsonl),
        "collector_fingerprint": collector_meta.get("fingerprint"),
        "model": tree_identity(args.model_path),
        "conv_mode": CONV_MODE,
        "mathematics": {
            "harmful_weights": "max(shift_correct_mean_nll-clean_correct_mean_nll,0)",
            "stable_weights": "1[delta_mean_nll<=0]",
            "ridge": "0.01*mean(clinical_gradient^2), per LODO fold",
            "parallel_repetitions": PARALLEL_REPETITIONS,
            "parallel_quantile": PARALLEL_QUANTILE,
            "shrinkage_strength": SHRINKAGE_STRENGTH,
            "fisher_weight": FISHER_WEIGHT,
            "center": "mean unique training-record clean activation",
        },
        "evaluation": "full Yes./No. teacher-forced total sequence NLL",
        "seed": args.seed,
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "halluspace": sha256_file(project_root / "corrected_sgta/domain_halluspace.py"),
            "collector": sha256_file(project_root / "corrected_sgta/collect_rule_source_halluspace.py"),
        },
    }
    fingerprint = stable_sha256(config)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    fold_path = args.output.with_suffix(args.output.suffix + ".folds.pt")
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    metadata = {"fingerprint": fingerprint, "config": config, "pair_count": len(pairs)}
    folds = None
    if args.resume and meta_path.is_file() and fold_path.is_file() and not args.dry_run:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        payload = torch.load(fold_path, map_location="cpu", weights_only=True)
        if (
            existing_meta.get("fingerprint") != fingerprint
            or payload.get("fingerprint") != fingerprint
        ):
            raise HalluspaceRunnerError("resume metadata/fold fingerprint mismatch")
        folds = {
            domain: fold_from_payload(fold)
            for domain, fold in payload["folds"].items()
        }
    if folds is None:
        folds = fit_lodo_folds(pairs, seed=args.seed)
    rank_gate = source_rank_gate(folds)
    if args.dry_run:
        print(
            json.dumps(
                {
                    **metadata,
                    "folds": {
                        domain: {
                            "training_domains": fold["training_domains"],
                            "rank": fold["fitted"].rank.selected_rank,
                            "ridge": fold["ridge"],
                        }
                        for domain, fold in folds.items()
                    },
                    "rank_gate": rank_gate,
                    "would_load_vlm": rank_gate["status"] == "passed",
                },
                indent=2,
            )
        )
        return
    if meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise HalluspaceRunnerError("output metadata fingerprint mismatch")
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        if not fold_path.is_file():
            raise HalluspaceRunnerError("metadata exists without fold artifact")
    else:
        if args.output.exists() or fold_path.exists():
            raise HalluspaceRunnerError("output exists without matching metadata")
        atomic_json(meta_path, metadata)
        atomic_torch_save(
            fold_path,
            {
                "version": VERSION,
                "fingerprint": fingerprint,
                "folds": {domain: fold_to_payload(fold) for domain, fold in folds.items()},
            },
        )
    if fold_path.exists():
        payload = torch.load(fold_path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != fingerprint:
            raise HalluspaceRunnerError("fold artifact fingerprint mismatch")
        folds = {domain: fold_from_payload(fold) for domain, fold in payload["folds"].items()}
    rank_gate = source_rank_gate(folds)
    if rank_gate["status"] == "failed":
        args.output.touch(exist_ok=True)
        atomic_json(
            summary_path,
            {
                "version": VERSION,
                "fingerprint": fingerprint,
                "scope": config["scope"],
                "status": "terminated_before_vlm_evaluation",
                "reason": "at least one source-only LODO fold selected rank zero",
                "rank_gate": rank_gate,
                "vlm_loaded": False,
                "projection_evaluated": False,
                "folds": {
                    domain: {
                        "training_domains": fold["training_domains"],
                        "training_pair_count": fold["training_pair_count"],
                        "training_image_count": fold["training_image_count"],
                        "ridge": fold["ridge"],
                        "rank_diagnostics": jsonable(asdict(fold["fitted"].rank)),
                    }
                    for domain, fold in sorted(folds.items())
                },
            },
        )
        print(json.dumps({"summary": str(summary_path), "rank_gate": rank_gate}, indent=2))
        return
    repair_jsonl_tail(args.output)
    done = completed_ids(args.output, fingerprint)

    source_bank = Path(str(collector_meta.get("config", {}).get("source_bank", "")))
    source_ids = {str(row["view_config"]["source_id"]) for row in pairs}
    if len(source_ids) != 1:
        raise HalluspaceRunnerError("finalized collector protocol requires one frozen center")
    frozen_source_id = next(iter(source_ids))
    center_amplitude, _ = load_fixed_centers(source_bank, frozen_source_id)
    max_side = int(collector_meta.get("config", {}).get("max_image_side", 384))
    adapter = LlavaMedAlignmentAdapter(model_path=args.model_path, conv_mode=CONV_MODE)
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    try:
        for row in tqdm(pairs, desc="source halluspace LODO"):
            if row["eval_id"] in done:
                continue
            fold = folds[row["source_domain"]]
            fitted, center = fold["fitted"], fold["center"]
            source_id = row["view_config"]["source_id"]
            if source_id != frozen_source_id:
                raise HalluspaceRunnerError(f"unexpected frozen source center {source_id}")
            with Image.open(row["image"]) as handle:
                clean_image = resize_image(handle.convert("RGB"), max_side)
            shifted_image = feddg_frequency_interpolation_release2(
                clean_image,
                center_amplitude,
                low_frequency_ratio=float(row["view_config"].get("low_frequency_ratio", LOW_FREQUENCY_RATIO)),
                source_ratio=float(row["view_config"].get("source_ratio", SOURCE_RATIO)),
            )
            output = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "status": "ok",
                "eval_id": row["eval_id"],
                "record_id": row["record_id"],
                "source_domain": row["source_domain"],
                "source_id": source_id,
                "gt_label": row["gt_label"],
            }
            for scope, image in (("shifted", shifted_image), ("clean", clean_image)):
                output[scope] = {
                    "baseline_nll": {
                        label: complete_sequence_nll(
                            adapter, image, row["question"], label.capitalize() + "."
                        )
                        for label in ("yes", "no")
                    },
                    "projected_nll": {
                        label: complete_sequence_nll(
                            adapter,
                            image,
                            row["question"],
                            label.capitalize() + ".",
                            fitted,
                            center,
                        )
                        for label in ("yes", "no")
                    },
                }
            append_jsonl(args.output, output)
    finally:
        del adapter

    rows = [
        json.loads(line)
        for line in args.output.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") == "ok"
    ]
    by_fold = {
        domain: summarize_evaluations(
            [row for row in rows if row["source_domain"] == domain]
        )
        for domain in sorted(folds)
    }
    micro = summarize_evaluations(rows)
    atomic_json(
        summary_path,
        {
            "version": VERSION,
            "fingerprint": fingerprint,
            "scope": config["scope"],
            "folds": {
                domain: {
                    **metrics,
                    "selected_rank": folds[domain]["fitted"].rank.selected_rank,
                    "rank_diagnostics": jsonable(asdict(folds[domain]["fitted"].rank)),
                    "ridge": folds[domain]["ridge"],
                }
                for domain, metrics in by_fold.items()
            },
            "micro": micro,
        },
    )
    print(json.dumps({"summary": str(summary_path), "safety_gate": micro["safety_gate"]}, indent=2))


if __name__ == "__main__":
    main()
