#!/usr/bin/env python3
"""Source-only LODO runner for the terminal ANCHOR attention experiment.

The runner has no target-test argument.  It first caches label-free visual
attention log-odds on source training domains, fits each fold's center using
only the other source domains, and compares identity with frozen ANCHOR on the
held-out source development domain using complete ``Yes.``/``No.`` sequence
NLL.  Each fold's trust radius is the frozen q90 absolute source-to-center gap
over enabled heads, computed only from that fold's training source domains.

The terminal protocol is intentionally narrow: LLaVA-Med/Mistral,
Transformers 5.6 eager attention, batch size one, fixed layers 2--31, and the
first answer-decision query only.  It exposes no layer or radius search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from tqdm import tqdm

from corrected_sgta.models import LLAVA_PATH
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_attention_mass_anchor import (
    PROTOCOL_VERSION,
    SOURCE_SELECTION_SCOPE,
    SourceOnlyGateConfig,
    anchor_attention_logits,
    evaluate_source_only_gate,
    robust_source_attention_center,
    visual_attention_log_odds,
)
from corrected_sgta.rule_dg_adapter_fingerprint_v3 import tree_identity
from corrected_sgta.train_rule_dg_adapter import (
    IGNORE_INDEX,
    build_teacher_forcing,
    canonical_answer,
    file_sha256,
    process_image,
    rule_no_reference_prompt,
)


VERSION = "rule-attention-mass-anchor-source-lodo-v2"
TERMINAL_LAYERS = tuple(range(2, 32))
SOURCE_Q90 = 0.90
MAD_MULTIPLIER = 2.5
ANCHOR_KEY = "source_q90"


class RunnerError(RuntimeError):
    """Fail-closed protocol or runtime error."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RunnerError(f"non-object JSONL row at {path}:{line_number}")
                rows.append(value)
    return rows


def row_label(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise RunnerError(f"invalid two-turn source row: {row.get('id')}")
    return canonical_answer(conversations[1].get("value"))[:-1].lower()


def row_question(row: Mapping[str, Any]) -> str:
    conversations = row["conversations"]
    value = str(conversations[0].get("value", "")).strip()
    if not value:
        raise RunnerError(f"empty source question: {row.get('id')}")
    return value


def select_source_rows(
    rows: list[dict[str, Any]], limit: int, seed: int, *, balanced: bool
) -> list[dict[str, Any]]:
    """Deterministic source selection; balancing is used only for LODO labels."""

    if limit <= 0:
        raise RunnerError("limit-per-domain must be positive in the pilot")
    ordered = sorted(
        rows,
        key=lambda row: stable_sha256(
            [seed, row.get("id"), row.get("image_sha256"), row.get("source_domain")]
        ),
    )
    if not balanced:
        return ordered[:limit]
    buckets = {"yes": [], "no": []}
    for row in ordered:
        buckets[row_label(row)].append(row)
    selected = buckets["yes"][: limit // 2] + buckets["no"][: limit // 2]
    selected_ids = {str(row["id"]) for row in selected}
    selected.extend(row for row in ordered if str(row["id"]) not in selected_ids)
    return sorted(selected[:limit], key=lambda row: str(row["id"]))


def validate_selected_images(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = []
    for row in rows:
        path = Path(str(row.get("image", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_sha256(path)
        recorded = str(row.get("image_blob_sha256", ""))
        if recorded and observed != recorded:
            raise RunnerError(f"image hash mismatch: {path}")
        identity.append({"id": str(row["id"]), "image": str(path), "sha256": observed})
    return identity


def source_paths(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, dict[str, Path]]:
    outputs = manifest.get("outputs", {}).get("by_domain", {})
    if len(outputs) < 3:
        raise RunnerError("expanded source manifest must contain at least three domains")
    result: dict[str, dict[str, Path]] = {}
    for domain, splits in sorted(outputs.items()):
        result[domain] = {}
        for split in ("train", "dev"):
            raw = splits.get(split, {}).get("jsonl")
            if not raw:
                raise RunnerError(f"manifest missing {domain}/{split} JSONL")
            path = Path(raw)
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.is_file():
                raise FileNotFoundError(path)
            result[domain][split] = path.resolve()
    return result


def visual_interval(
    input_ids: torch.Tensor,
    expanded_length: int,
    image_token_index: int | None = None,
) -> tuple[int, int]:
    if image_token_index is None:
        from llava.constants import IMAGE_TOKEN_INDEX

        image_token_index = IMAGE_TOKEN_INDEX

    positions = input_ids[0].eq(image_token_index).nonzero(as_tuple=False).flatten()
    if positions.numel() != 1:
        raise RunnerError(f"expected one image placeholder, observed {positions.numel()}")
    start = int(positions.item())
    count = int(expanded_length - input_ids.shape[1] + 1)
    if count <= 0:
        raise RunnerError("multimodal preparation did not expand the image placeholder")
    return start, start + count


class MistralAttentionMassHook(AbstractContextManager):
    """Independent Transformers-5 eager hook for selected Mistral layers."""

    def __init__(self, model: Any, layers: tuple[int, ...]):
        self.model = model
        self.layers = layers
        self.original: dict[int, Any] = {}
        self.image_start = 0
        self.image_end = 0
        self.query_index = -1
        self.center: torch.Tensor | None = None
        self.head_mask: torch.Tensor | None = None
        self.tau = 0.0
        self.apply_anchor = False
        self.capture: dict[int, torch.Tensor] = {}
        self.delta: dict[int, torch.Tensor] = {}

    def configure(
        self,
        *,
        image_start: int,
        image_end: int,
        query_index: int,
        center: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        tau: float = 0.0,
    ) -> None:
        self.image_start = image_start
        self.image_end = image_end
        self.query_index = query_index
        self.center = center
        self.head_mask = head_mask
        self.tau = float(tau)
        self.apply_anchor = center is not None
        self.capture = {}
        self.delta = {}

    def __enter__(self):
        from transformers.models.mistral import modeling_mistral

        layers = self.model.model.layers
        if not self.layers or min(self.layers) < 0 or max(self.layers) >= len(layers):
            raise RunnerError(f"invalid pilot layers {self.layers} for {len(layers)} layers")
        for layer_index in self.layers:
            attention = layers[layer_index].self_attn
            self.original[layer_index] = attention.forward

            def anchored_forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                _layer_index=layer_index,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                query, key = modeling_mistral.apply_rotary_pos_emb(query, key, cos, sin)
                if past_key_values is not None:
                    key, value = past_key_values.update(key, value, module.layer_idx)
                key = modeling_mistral.repeat_kv(key, module.num_key_value_groups)
                value = modeling_mistral.repeat_kv(value, module.num_key_value_groups)
                weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
                if attention_mask is not None:
                    weights = weights + attention_mask
                query_index = self.query_index if self.query_index >= 0 else weights.shape[-2] - 1
                if not 0 <= query_index < weights.shape[-2]:
                    raise RunnerError("decision query is outside attention query range")
                decision = weights[:, :, query_index : query_index + 1, :]
                before = visual_attention_log_odds(decision, self.image_start, self.image_end)
                self.capture[_layer_index] = before[0, :, 0].detach().float().cpu()
                if self.apply_anchor:
                    position = self.layers.index(_layer_index)
                    decision, audit = anchor_attention_logits(
                        decision,
                        self.image_start,
                        self.image_end,
                        self.center[position],
                        self.tau,
                        self.head_mask[position] if self.head_mask is not None else None,
                    )
                    weights = weights.clone()
                    weights[:, :, query_index : query_index + 1, :] = decision
                    self.delta[_layer_index] = audit["delta"][0, :, 0].detach().float().cpu()
                weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
                weights = F.dropout(
                    weights,
                    p=0.0 if not module.training else module.attention_dropout,
                    training=module.training,
                )
                output = torch.matmul(weights, value).transpose(1, 2).contiguous()
                output = output.reshape(*input_shape, -1).contiguous()
                return module.o_proj(output), weights

            attention.forward = types.MethodType(anchored_forward, attention)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for layer_index, forward in self.original.items():
            self.model.model.layers[layer_index].self_attn.forward = forward
        self.original = {}
        return False

    def stacked_capture(self) -> torch.Tensor:
        if set(self.capture) != set(self.layers):
            raise RunnerError(f"incomplete attention capture: {sorted(self.capture)}")
        return torch.stack([self.capture[layer] for layer in self.layers])


def prepare_multimodal(adapter, image: Image.Image, input_ids, labels):
    ids = input_ids.to(adapter.model.device)
    targets = None if labels is None else labels.to(adapter.model.device)
    pixels = process_image(adapter, image)
    return adapter.model.prepare_inputs_labels_for_multimodal(
        ids, None, None, None, targets, pixels, image_sizes=[image.size]
    )


@torch.inference_mode()
def collect_source_log_odds(adapter, hook, image: Image.Image, prompt: str) -> torch.Tensor:
    input_ids = adapter._prompt_ids(prompt)
    _, position_ids, attention_mask, _, embeds, _ = prepare_multimodal(
        adapter, image, input_ids, None
    )
    image_start, image_end = visual_interval(input_ids, embeds.shape[1])
    hook.configure(
        image_start=image_start,
        image_end=image_end,
        query_index=embeds.shape[1] - 1,
    )
    adapter.model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=embeds,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return hook.stacked_capture()


@torch.inference_mode()
def complete_sequence_nll(
    adapter,
    hook,
    image: Image.Image,
    prompt: str,
    answer: str,
    center: torch.Tensor | None,
    head_mask: torch.Tensor | None,
    tau: float,
) -> float:
    input_ids, labels = build_teacher_forcing(adapter, prompt, answer)
    _, position_ids, attention_mask, _, embeds, expanded_labels = prepare_multimodal(
        adapter, image, input_ids, labels
    )
    if expanded_labels is None:
        raise RunnerError("teacher forcing returned no expanded labels")
    target_positions = expanded_labels[0].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if target_positions.numel() < 2:
        raise RunnerError("complete Yes/No sequence must have at least two target tokens")
    decision_query = int(target_positions[0]) - 1
    image_start, image_end = visual_interval(input_ids, embeds.shape[1])
    hook.configure(
        image_start=image_start,
        image_end=image_end,
        query_index=decision_query,
        center=center,
        head_mask=head_mask,
        tau=tau,
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
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        expanded_labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    if not math.isfinite(float(loss)):
        raise RunnerError("non-finite complete-sequence NLL")
    return float(loss)


def prediction(nll: Mapping[str, float]) -> str:
    return "yes" if float(nll["yes"]) <= float(nll["no"]) else "no"


def summarize_fold_records(records: list[dict[str, Any]], tau_key: str) -> dict[str, Any]:
    if not records:
        raise RunnerError("cannot summarize an empty source LODO fold")
    rescues = harms = identity_correct = anchored_correct = 0
    rescues_by_label = {"yes": 0, "no": 0}
    margin_deltas = {"yes": [], "no": []}
    identity_truth_nll = []
    anchored_truth_nll = []
    for row in records:
        label = row["label"]
        base = row["identity"]
        anchored = row["anchor"][tau_key]
        base_ok = prediction(base) == label
        anchor_ok = prediction(anchored) == label
        identity_correct += int(base_ok)
        anchored_correct += int(anchor_ok)
        if not base_ok and anchor_ok:
            rescues += 1
            rescues_by_label[label] += 1
        if base_ok and not anchor_ok:
            harms += 1
        base_margin = float(base["no"]) - float(base["yes"])
        anchor_margin = float(anchored["no"]) - float(anchored["yes"])
        margin_deltas[label].append(anchor_margin - base_margin)
        identity_truth_nll.append(float(base[label]))
        anchored_truth_nll.append(float(anchored[label]))
    if not margin_deltas["yes"] or not margin_deltas["no"]:
        raise RunnerError("held-out source fold must contain both labels")
    balanced_shift = 0.5 * sum(
        sum(margin_deltas[label]) / len(margin_deltas[label]) for label in ("yes", "no")
    )
    n = len(records)
    return {
        "n": n,
        "identity_accuracy": identity_correct / n,
        "anchored_accuracy": anchored_correct / n,
        "delta_pp": 100.0 * (anchored_correct - identity_correct) / n,
        "rescues": rescues,
        "harms": harms,
        "rescues_by_label": rescues_by_label,
        "balanced_margin_shift": balanced_shift,
        "identity_truth_sequence_nll": sum(identity_truth_nll) / n,
        "anchored_truth_sequence_nll": sum(anchored_truth_nll) / n,
    }


def source_q90_tau(
    center_log_odds: torch.Tensor,
    head_mask: torch.Tensor,
    training_domain_log_odds: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Fit a fold-local q90 radius from enabled training-source heads only."""

    if not training_domain_log_odds:
        raise RunnerError("source-q90 requires training source domains")
    if center_log_odds.shape != head_mask.shape or not bool(head_mask.any()):
        raise RunnerError("source-q90 requires a non-empty aligned head mask")
    values = []
    sample_count = 0
    for domain in sorted(training_domain_log_odds):
        tensor = training_domain_log_odds[domain].float()
        if tensor.ndim != center_log_odds.ndim + 1 or tensor.shape[1:] != center_log_odds.shape:
            raise RunnerError(f"source-q90 shape mismatch for {domain}")
        if tensor.shape[0] < 1 or not torch.isfinite(tensor).all():
            raise RunnerError(f"source-q90 requires finite samples for {domain}")
        sample_count += tensor.shape[0]
        values.append((tensor - center_log_odds.unsqueeze(0)).abs()[:, head_mask])
    enabled_gaps = torch.cat(values, dim=0).flatten()
    tau = torch.quantile(enabled_gaps, SOURCE_Q90, interpolation="linear")
    if not torch.isfinite(tau) or float(tau) <= 0:
        raise RunnerError("source-q90 produced a non-positive or non-finite radius")
    return {
        "tau": float(tau),
        "quantile": SOURCE_Q90,
        "sample_count": sample_count,
        "enabled_head_count": int(head_mask.sum()),
        "gap_count": int(enabled_gaps.numel()),
    }


def signed_correct_margin_deltas(
    records: list[dict[str, Any]], tau_key: str = ANCHOR_KEY
) -> list[float]:
    values = []
    for row in records:
        base = row["identity"]
        anchored = row["anchor"][tau_key]
        margin_delta = (
            float(anchored["no"])
            - float(anchored["yes"])
            - float(base["no"])
            + float(base["yes"])
        )
        values.append(margin_delta if row["label"] == "yes" else -margin_delta)
    return values


def terminal_source_gate(
    fold_records: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Apply the original gate plus frozen fold-direction and margin checks."""

    folds = {
        domain: summarize_fold_records(records, ANCHOR_KEY)
        for domain, records in sorted(fold_records.items())
    }
    base_gate = evaluate_source_only_gate(
        folds, SourceOnlyGateConfig(), SOURCE_SELECTION_SCOPE
    )
    nonnegative = sum(fold["delta_pp"] >= 0.0 for fold in folds.values())
    all_signed = [
        value
        for domain in sorted(fold_records)
        for value in signed_correct_margin_deltas(fold_records[domain])
    ]
    pooled_median = statistics.median(all_signed)
    checks = dict(base_gate["checks"])
    checks["at_least_two_thirds_folds_nonnegative"] = (
        3 * nonnegative >= 2 * len(folds)
    )
    checks["pooled_signed_correct_margin_median_positive"] = pooled_median > 0.0
    passed = all(checks.values())
    return {
        **base_gate,
        "status": "passed" if passed else "failed",
        "target_falsification_allowed": passed,
        "checks": checks,
        "folds": folds,
        "terminal_statistics": {
            "nonnegative_fold_count": nonnegative,
            "fold_count": len(folds),
            "pooled_signed_correct_margin_median": pooled_median,
        },
    }


def code_identity() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("rule_attention_mass_anchor.py"),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("models.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=LLAVA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-domain", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_per_domain <= 0:
        raise RunnerError("limit-per-domain must be positive")
    if not transformers.__version__.startswith("5.6."):
        raise RunnerError(
            f"terminal protocol requires Transformers 5.6.x, observed {transformers.__version__}"
        )
    if not args.source_manifest.is_file() or not args.model_path.is_dir():
        raise FileNotFoundError("source manifest or model path is missing")
    manifest = json.loads(args.source_manifest.read_text())
    paths = source_paths(manifest, args.source_manifest)
    selected_train = {}
    selected_dev = {}
    for domain, split_paths in sorted(paths.items()):
        selected_train[domain] = select_source_rows(
            load_jsonl(split_paths["train"]), args.limit_per_domain, args.seed, balanced=False
        )
        selected_dev[domain] = select_source_rows(
            load_jsonl(split_paths["dev"]), args.limit_per_domain, args.seed, balanced=True
        )
        if {row_label(row) for row in selected_dev[domain]} != {"yes", "no"}:
            raise RunnerError(f"source LODO fold {domain} lacks both labels")
    image_identity = {
        f"{domain}/{split}": validate_selected_images(rows)
        for domain, groups in ((name, (selected_train[name], selected_dev[name])) for name in sorted(paths))
        for split, rows in zip(("train", "dev"), groups)
    }
    fingerprint_payload = {
        "version": VERSION,
        "mathematics_protocol": PROTOCOL_VERSION,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "source_manifest_fingerprint": manifest.get("fingerprint"),
        "source_json_sha256": {
            f"{domain}/{split}": file_sha256(path)
            for domain, values in paths.items()
            for split, path in values.items()
        },
        "selected_images": image_identity,
        "model": tree_identity(args.model_path),
        "layers": list(TERMINAL_LAYERS),
        "tau_policy": {
            "name": "enabled_head_source_fit_absolute_gap_q90",
            "quantile": SOURCE_Q90,
            "target_access": False,
        },
        "mad_multiplier": MAD_MULTIPLIER,
        "limit_per_domain": args.limit_per_domain,
        "seed": args.seed,
        "selection_scope": SOURCE_SELECTION_SCOPE,
        "target_labels_accessed": False,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "attention_implementation": "transformers-5.6-eager-mistral",
        },
        "code_sha256": code_identity(),
    }
    fingerprint = stable_sha256(fingerprint_payload)
    if args.dry_run:
        print(json.dumps({"fingerprint": fingerprint, "domains": sorted(paths), "n_train": {k: len(v) for k, v in selected_train.items()}, "n_dev": {k: len(v) for k, v in selected_dev.items()}}, indent=2))
        return
    state_path = args.output_dir / "state.json"
    summary_path = args.output_dir / "summary.json"
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"{state_path} exists; use --resume")
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise RunnerError("resume fingerprint mismatch")
    else:
        state = {
            "version": VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "attention_cache": {domain: {} for domain in paths},
            "fold_records": {domain: {} for domain in paths},
            "fold_centers": {},
            "complete": False,
        }
        atomic_json(state_path, state)

    adapter = LlavaMedAlignmentAdapter(model_path=args.model_path, conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    layers = TERMINAL_LAYERS
    try:
        with MistralAttentionMassHook(adapter.model, layers) as hook:
            for domain in sorted(paths):
                cache = state["attention_cache"][domain]
                for row in tqdm(selected_train[domain], desc=f"anchor-center-{domain}"):
                    key = str(row["id"])
                    if key in cache:
                        continue
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_no_reference_prompt(row_question(row))
                    values = collect_source_log_odds(adapter, hook, image, prompt)
                    cache[key] = {
                        "id": key,
                        "image_sha256": row.get("image_sha256"),
                        "log_odds": values.tolist(),
                    }
                    atomic_json(state_path, state)

            for heldout in sorted(paths):
                training_domains = [domain for domain in sorted(paths) if domain != heldout]
                training_tensors = {
                    domain: torch.tensor(
                        [entry["log_odds"] for entry in state["attention_cache"][domain].values()],
                        dtype=torch.float32,
                    )
                    for domain in training_domains
                }
                fitted = robust_source_attention_center(
                    training_tensors,
                    mad_multiplier=MAD_MULTIPLIER,
                    minimum_domains=2,
                )
                tau_fit = source_q90_tau(
                    fitted.center_log_odds, fitted.head_mask, training_tensors
                )
                state["fold_centers"][heldout] = {
                    "training_domains": training_domains,
                    "center_log_odds": fitted.center_log_odds.tolist(),
                    "domain_mad": fitted.domain_mad.tolist(),
                    "head_mask": fitted.head_mask.tolist(),
                    "mad_threshold": float(fitted.mad_threshold),
                    "source_q90": tau_fit,
                }
                records = state["fold_records"][heldout]
                for row in tqdm(selected_dev[heldout], desc=f"anchor-lodo-{heldout}"):
                    key = str(row["id"])
                    if key in records:
                        continue
                    with Image.open(row["image"]) as handle:
                        image = handle.convert("RGB")
                    prompt = rule_no_reference_prompt(row_question(row))
                    identity = {
                        label: complete_sequence_nll(
                            adapter, hook, image, prompt, label.capitalize() + ".", None, None, 0.0
                        )
                        for label in ("yes", "no")
                    }
                    anchors = {
                        ANCHOR_KEY: {
                            label: complete_sequence_nll(
                                adapter,
                                hook,
                                image,
                                prompt,
                                label.capitalize() + ".",
                                fitted.center_log_odds.to(adapter.model.device),
                                fitted.head_mask.to(adapter.model.device),
                                tau_fit["tau"],
                            )
                            for label in ("yes", "no")
                        }
                    }
                    records[key] = {
                        "id": key,
                        "source_domain": heldout,
                        "label": row_label(row),
                        "identity": identity,
                        "anchor": anchors,
                    }
                    atomic_json(state_path, state)
    finally:
        adapter.close()

    fold_lists = {
        domain: list(state["fold_records"][domain].values()) for domain in sorted(paths)
    }
    terminal_gate = terminal_source_gate(fold_lists)
    state["complete"] = True
    state["terminal_gate"] = terminal_gate
    atomic_json(state_path, state)
    atomic_json(
        summary_path,
        {
            "version": VERSION,
            "fingerprint": fingerprint,
            "scope": "source-only LODO; no target labels or target tuning",
            "terminal_limit": "fixed layers 2--31 and first answer-decision query only",
            "fold_taus": {
                domain: state["fold_centers"][domain]["source_q90"]
                for domain in sorted(paths)
            },
            "terminal_gate": terminal_gate,
        },
    )
    print(json.dumps({"summary": str(summary_path), "target_falsification_allowed": terminal_gate["target_falsification_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
