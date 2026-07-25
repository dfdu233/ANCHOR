"""Train a minimal model-specific visual DG adapter for RULE-style VQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from tqdm import tqdm

from corrected_sgta.frequency_alignment_source_spectrum_release2 import source_spectrum_alignment_release2
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter

VERSION = "rule-preprojector-sequence-dg-adapter-v3"
IGNORE_INDEX = -100


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def rule_label(text: object) -> str:
    """Mirror RULE's public first-sentence Yes/No coercion."""
    sentence = "" if text is None else str(text).replace("\n", " ").strip()
    if "." in sentence:
        sentence = sentence.split(".", 1)[0]
    words = sentence.replace(",", "").split(" ")
    return "No" if any(word in {"No", "no", "not"} for word in words) else "Yes"


def extract_question_only(text: object) -> str:
    value = str(text).replace("<image>", "").strip()
    if "Question:" in value:
        value = value.rsplit("Question:", 1)[1].strip()
    if "\n" in value:
        value = value.split("\n", 1)[0].strip()
    if not value:
        raise ValueError("empty question after removing alignment context")
    return value


def rule_no_reference_prompt(question: object) -> str:
    value = str(question).replace("<image>", "").strip()
    suffix = "Please answer the question based on the image and choose from the following two options: [yes, no]."
    if suffix.lower() in value.lower():
        return value
    return f"{value} {suffix}"


def canonical_answer(text: object) -> str:
    return rule_label(text) + "."


def select_one_qa_per_image(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image = str(row.get("image", "")).strip()
        conversations = row.get("conversations")
        if not image or not isinstance(conversations, list) or len(conversations) != 2:
            continue
        if not str(conversations[1].get("value", "")).strip():
            continue
        grouped.setdefault(image, []).append(row)
    images = sorted(grouped, key=lambda item: stable_digest(f"{seed}:image:{item}"))
    if maximum:
        images = images[:maximum]
    selected = []
    for image in images:
        candidates = sorted(
            grouped[image],
            key=lambda row: stable_digest(f"{seed}:qa:{row.get('id')}:{row['conversations'][0].get('value')}")
        )
        selected.append(candidates[0])
    return selected


class BoundedResidualBottleneck(nn.Module):
    """Zero-init low-rank update with a per-token relative trust region."""
    def __init__(self, width: int, rank: int, max_relative_update: float):
        super().__init__()
        if rank <= 0 or max_relative_update <= 0:
            raise ValueError("rank and max_relative_update must be positive")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.max_relative_update = float(max_relative_update)
        self.last_mean_relative_norm = 0.0
        self.last_max_relative_norm = 0.0
        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original = value.float()
        update = self.up(F.gelu(self.down(original)))
        original_norm = original.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        relative = update.norm(dim=-1, keepdim=True) / original_norm
        scale = (self.max_relative_update / relative.clamp_min(1e-12)).clamp(max=1.0)
        bounded = update * scale
        realized = bounded.norm(dim=-1) / original_norm.squeeze(-1)
        self.last_mean_relative_norm = float(realized.detach().mean().cpu())
        self.last_max_relative_norm = float(realized.detach().max().cpu())
        output = value + bounded.to(value.dtype)
        self.last_input = original
        self.last_output = output.float()
        return output


def normalized_feature_distance(
    reference: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """Non-squared token distance with a stop-gradient clean reference."""
    if reference.shape != candidate.shape:
        raise ValueError("reference/candidate feature shapes differ")
    anchor = F.normalize(reference.detach().float(), dim=-1)
    moved = F.normalize(candidate.float(), dim=-1)
    return (moved - anchor).norm(dim=-1).mean()


def loss_gradient_norm(loss: torch.Tensor, parameters: list[nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared = sum(
        float(gradient.detach().float().square().sum())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(squared)


def vision_width(model) -> int:
    tower = model.get_vision_tower()
    for candidate in (tower, getattr(tower, "vision_tower", None)):
        config = getattr(candidate, "config", None)
        width = getattr(config, "hidden_size", None)
        if width is not None:
            return int(width)
    projector = model.get_model().mm_projector
    for layer in projector.modules():
        if isinstance(layer, nn.Linear):
            return int(layer.in_features)
    raise RuntimeError("could not determine pre-projector visual width")


def projector_output_width(model) -> int:
    """Return the language-embedding width produced by the visual projector."""
    projector = model.get_model().mm_projector
    linear_layers = [layer for layer in projector.modules() if isinstance(layer, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("could not determine visual-projector output width")
    return int(linear_layers[-1].out_features)


@contextmanager
def attach_preprojector_adapter(model, module: nn.Module) -> Iterator[None]:
    projector = model.get_model().mm_projector
    original_forward = projector.forward

    def forward(features: torch.Tensor, *args, **kwargs):
        return original_forward(module(features), *args, **kwargs)

    projector.forward = forward
    try:
        yield
    finally:
        projector.forward = original_forward


@contextmanager
def attach_postprojector_adapter(model, module: nn.Module) -> Iterator[None]:
    """Apply one residual module after the frozen visual projector."""
    projector = model.get_model().mm_projector
    original_forward = projector.forward

    def forward(features: torch.Tensor, *args, **kwargs):
        return module(original_forward(features, *args, **kwargs))

    projector.forward = forward
    try:
        yield
    finally:
        projector.forward = original_forward


def relative_residual_energy(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Differentiable, scale-normalized identity penalty for a residual adapter."""
    if reference.shape != candidate.shape:
        raise ValueError("reference/candidate feature shapes differ")
    denominator = reference.detach().float().square().sum(dim=-1).clamp_min(1e-6)
    numerator = (candidate.float() - reference.detach().float()).square().sum(dim=-1)
    return (numerator / denominator).mean()


def build_teacher_forcing(adapter: LlavaMedAlignmentAdapter, prompt: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    image_token = DEFAULT_IMAGE_TOKEN
    if getattr(adapter.model.config, "mm_use_im_start_end", False):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    conversation = conv_templates[adapter.conv_mode].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + prompt)
    conversation.append_message(conversation.roles[1], answer)
    full_ids = tokenizer_image_token(
        conversation.get_prompt(), adapter.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    prompt_ids = adapter._prompt_ids(prompt)
    prefix = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= prefix:
        raise RuntimeError("assistant target produced no answer tokens")
    if not torch.equal(full_ids[:, :prefix], prompt_ids):
        common = 0
        for left, right in zip(full_ids[0].tolist(), prompt_ids[0].tolist()):
            if left != right:
                break
            common += 1
        raise RuntimeError(f"prompt/full tokenization mismatch at {common}/{prefix}")
    labels = full_ids.clone()
    labels[:, :prefix] = IGNORE_INDEX
    if int((labels != IGNORE_INDEX).sum()) < 2:
        raise RuntimeError("full target must contain answer token plus separator/EOS")
    return full_ids, labels


def process_image(adapter: LlavaMedAlignmentAdapter, image: Image.Image):
    from llava.mm_utils import process_images
    pixels = process_images([image], adapter.image_processor, adapter.model.config)
    if isinstance(pixels, list):
        return [item.to(adapter.model.device, dtype=adapter.model.dtype) for item in pixels]
    return pixels.to(adapter.model.device, dtype=adapter.model.dtype)


def answer_logits(output_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    shifted_logits = output_logits[:, :-1]
    value = shifted_logits[shifted_labels.ne(IGNORE_INDEX)].float()
    if value.ndim != 2 or value.shape[0] == 0:
        raise RuntimeError("no answer-token logits found")
    return value


def sequence_forward(
    adapter,
    image,
    input_ids,
    labels,
    module,
    adapter_location: str = "pre",
    return_token_ids: bool = False,
):
    """Run teacher forcing, optionally returning gold ids aligned to answer logits."""
    ids = input_ids.to(adapter.model.device)
    targets = labels.to(adapter.model.device)
    pixels = process_image(adapter, image)
    if adapter_location not in {"pre", "post"}:
        raise ValueError(f"unknown adapter location: {adapter_location}")
    if module is None:
        context = nullcontext()
    elif adapter_location == "pre":
        context = attach_preprojector_adapter(adapter.model, module)
    else:
        context = attach_postprojector_adapter(adapter.model, module)
    with context:
        _, position_ids, attention_mask, _, inputs_embeds, expanded_labels = (
            adapter.model.prepare_inputs_labels_for_multimodal(
                ids,
                None,
                None,
                None,
                targets,
                pixels,
                image_sizes=[image.size],
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
        vocabulary_weight = adapter.model.get_output_embeddings().weight
        vocabulary_logits = output.last_hidden_state.to(vocabulary_weight.dtype) @ vocabulary_weight.T
    if expanded_labels is None:
        raise RuntimeError("multimodal preparation did not return expanded labels")
    logits = answer_logits(vocabulary_logits, expanded_labels)
    shift_logits = vocabulary_logits[:, :-1].float().contiguous()
    shift_labels = expanded_labels[:, 1:].contiguous()
    token_ids = shift_labels[shift_labels.ne(IGNORE_INDEX)].long()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite sequence loss/logits")
    if return_token_ids:
        return loss, logits, token_ids
    return loss, logits


def forward_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student answer-token shapes differ")
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits.detach(), dim=-1), reduction="batchmean"
    )


def stop_gradient_js(reference_logits: torch.Tensor, view_logits: torch.Tensor) -> torch.Tensor:
    reference = F.softmax(reference_logits.detach(), dim=-1)
    view = F.softmax(view_logits, dim=-1)
    mixture = 0.5 * (reference + view)
    return 0.5 * (
        F.kl_div(mixture.clamp_min(1e-12).log(), reference, reduction="batchmean")
        + F.kl_div(mixture.clamp_min(1e-12).log(), view, reduction="batchmean")
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--center", type=Path, required=True)
    parser.add_argument("--extra-center", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("erm", "dg", "feature_dg", "center_erm", "center_dro"),
        required=True,
    )
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--identity-kl-weight", type=float, default=1.0)
    parser.add_argument("--view-js-weight", type=float, default=1.0)
    parser.add_argument(
        "--feature-pair-gradient-ratio",
        type=float,
        default=0.25,
        help="First-step ||grad pair|| is scaled to this fraction of ||grad view task||.",
    )
    parser.add_argument("--max-feature-pair-weight", type=float, default=10.0)
    parser.add_argument("--max-relative-update", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def code_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(), root / "corrected_sgta/models.py", root / "corrected_sgta/models_alignment.py",
        root / "corrected_sgta/frequency_alignment_source_spectrum_release2.py",
    ]
    return {str(path): file_sha256(path) for path in paths}


def serializable_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serializable_config_value(item) for item in value]
    return value


def checkpoint_payload(args, module, optimizer, selected, history, next_step, width, provenance):
    return {
        "version": VERSION,
        "config": {key: serializable_config_value(value) for key, value in vars(args).items()},
        "width": width,
        "state_dict": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "selected": [
            {"id": row.get("id"), "image": row["image"], "target": canonical_answer(row["conversations"][1]["value"])}
            for row in selected
        ],
        "history": history,
        "next_step": next_step,
        "provenance": provenance,
    }


def source_view_canvas(image: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError("view-size must be positive")
    # Preserve the complete radiograph and avoid an expensive FFT at native resolution.
    return ImageOps.pad(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS, color=(122, 116, 104))


def main() -> None:
    args = parse_args()
    if args.epochs != 1:
        raise ValueError("v2 pilot intentionally supports exactly one epoch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = json.loads(args.train_json.read_text())
    selected = select_one_qa_per_image(rows, args.max_images, args.seed)
    if not selected:
        raise ValueError("no valid independent training images")
    missing = [row["image"] for row in selected if not (args.image_root / row["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} selected images; first={missing[0]}")
    test_rows = [json.loads(line) for line in args.test_json.read_text().splitlines() if line.strip()]
    test_images = {str(row.get("image", "")) for row in test_rows}
    overlap = sorted({row["image"] for row in selected} & test_images)
    if overlap:
        raise RuntimeError(f"train/test image leakage detected: {overlap[:3]}")

    center_paths = [args.center, *args.extra_center]
    if args.mode in {"center_erm", "center_dro"} and len(center_paths) < 2:
        raise ValueError("multi-center modes require at least one --extra-center")
    centers = [np.load(path) for path in center_paths]
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    width = vision_width(adapter.model)
    module = BoundedResidualBottleneck(width, args.rank, args.max_relative_update).to(adapter.model.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    provenance = {
        "train_sha256": file_sha256(args.train_json), "test_sha256": file_sha256(args.test_json),
        "center_sha256": {str(path.resolve()): file_sha256(path) for path in center_paths},
        "train_test_image_overlap": 0,
        "selected_unique_images": len(selected), "code_sha256": code_provenance(),
    }
    history: list[dict[str, float]] = []
    feature_pair_weight: float | None = None
    start = 0
    if args.resume and args.output.is_file():
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        if payload.get("version") != VERSION or payload.get("provenance") != provenance:
            raise RuntimeError("resume checkpoint fingerprint/provenance mismatch")
        if [row["image"] for row in selected] != [row["image"] for row in payload["selected"]]:
            raise RuntimeError("resume selected-image order mismatch")
        module.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload["history"])
        start = int(payload["next_step"])
        if history and args.mode == "feature_dg":
            feature_pair_weight = float(history[0]["feature_pair_weight"])

    progress = tqdm(range(start, len(selected)), desc=f"sequence-{args.mode}")
    for step in progress:
        row = selected[step]
        with Image.open(args.image_root / row["image"]) as handle:
            original = handle.convert("RGB")
        question = extract_question_only(row["conversations"][0]["value"])
        prompt = rule_no_reference_prompt(question)
        target = canonical_answer(row["conversations"][1]["value"])
        input_ids, labels = build_teacher_forcing(adapter, prompt, target)
        view_input = source_view_canvas(original, args.view_size)
        views = [
            source_spectrum_alignment_release2(
                view_input, center, low_frequency_ratio=args.alpha
            )
            for center in centers
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            base_loss, base_logits = sequence_forward(adapter, original, input_ids, labels, None)
        original_loss, original_logits = sequence_forward(adapter, original, input_ids, labels, module)
        if module.last_input is None:
            raise RuntimeError("adapter did not capture clean visual features")
        clean_feature_anchor = module.last_input.detach()
        identity_kl = forward_kl(base_logits, original_logits)
        (0.5 * original_loss + args.identity_kl_weight * identity_kl).backward()
        original_reference = original_logits.detach()
        del original_logits, base_logits
        zero = original_loss.detach().new_zeros(())
        view_js = zero
        feature_pair = zero
        center_losses: list[float] = []
        worst_center = -1
        if args.mode == "center_erm":
            view_loss_values = []
            for view in views:
                current_loss, _ = sequence_forward(adapter, view, input_ids, labels, module)
                (0.5 * current_loss / len(views)).backward()
                view_loss_values.append(current_loss.detach())
                center_losses.append(float(current_loss.detach()))
            view_loss = torch.stack(view_loss_values).mean()
        elif args.mode == "center_dro":
            with torch.no_grad():
                for view in views:
                    current_loss, _ = sequence_forward(adapter, view, input_ids, labels, module)
                    center_losses.append(float(current_loss))
            worst_center = int(np.argmax(center_losses))
            view_loss, _ = sequence_forward(
                adapter, views[worst_center], input_ids, labels, module
            )
            (0.5 * view_loss).backward()
        else:
            view_loss, view_logits = sequence_forward(
                adapter, views[0], input_ids, labels, module
            )
            view_js = stop_gradient_js(original_reference, view_logits)
            if module.last_output is None:
                raise RuntimeError("adapter did not capture view visual features")
            feature_pair = normalized_feature_distance(
                clean_feature_anchor, module.last_output
            )
            if args.mode == "feature_dg" and feature_pair_weight is None:
                parameters = [
                    parameter for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                task_gradient = loss_gradient_norm(0.5 * view_loss, parameters)
                pair_gradient = loss_gradient_norm(feature_pair, parameters)
                if pair_gradient <= 0:
                    raise RuntimeError(
                        "feature-pair gradient is zero; kill criterion triggered"
                    )
                feature_pair_weight = (
                    args.feature_pair_gradient_ratio * task_gradient / pair_gradient
                )
                if feature_pair_weight > args.max_feature_pair_weight:
                    raise RuntimeError(
                        f"feature-pair weight {feature_pair_weight:.6g} exceeds "
                        f"kill threshold {args.max_feature_pair_weight:.6g}"
                    )
            second_loss = 0.5 * view_loss
            if args.mode == "dg":
                second_loss = second_loss + args.view_js_weight * view_js
            elif args.mode == "feature_dg":
                assert feature_pair_weight is not None
                second_loss = second_loss + feature_pair_weight * feature_pair
            second_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.gradient_clip)
        if not math.isfinite(float(gradient_norm)):
            raise FloatingPointError("non-finite adapter gradient norm")
        optimizer.step()
        item = {
            "step": float(step), "base_nll": float(base_loss.detach()), "original_nll": float(original_loss.detach()),
            "view_nll": float(view_loss.detach()), "identity_kl": float(identity_kl.detach()),
            "view_js": float(view_js.detach()), "gradient_norm": float(gradient_norm),
            "feature_pair": float(feature_pair.detach()),
            "feature_pair_weight": float(feature_pair_weight or 0.0),
            "center_loss_spread": (
                float(max(center_losses) - min(center_losses))
                if center_losses else 0.0
            ),
            "worst_center": float(worst_center),
            "mean_relative_update": module.last_mean_relative_norm,
            "max_relative_update": module.last_max_relative_norm,
        }
        if not all(math.isfinite(value) for value in item.values()):
            raise FloatingPointError(f"non-finite training metric: {item}")
        history.append(item)
        progress.set_postfix(nll=f"{item['original_nll']:.3f}", identity=f"{item['identity_kl']:.4f}", js=f"{item['view_js']:.4f}")
        next_step = step + 1
        if next_step % args.save_every == 0 or next_step == len(selected):
            atomic_torch_save(checkpoint_payload(args, module, optimizer, selected, history, next_step, width, provenance), args.output)
        del original_reference, base_loss, original_loss, view_loss
        torch.cuda.empty_cache()
    print(json.dumps({
        "output": str(args.output), "version": VERSION, "mode": args.mode,
        "unique_images": len(selected), "steps_complete": len(history),
        "final": history[-1] if history else None, "provenance": provenance,
    }, indent=2))


if __name__ == "__main__":
    main()
