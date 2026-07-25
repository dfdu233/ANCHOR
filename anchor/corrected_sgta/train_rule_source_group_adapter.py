"""Train one low-rank post-projector adapter with balanced source risks.

Pooled ERM and source GroupDRO consume the exact same deterministic per-source
sample schedule. Their only difference is the source-risk aggregation weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.frequency_alignment_source_spectrum_release2 import (
    source_spectrum_alignment_release2,
)
from corrected_sgta.train_rule_dg_adapter import (
    BoundedResidualBottleneck,
    atomic_torch_save,
    build_teacher_forcing,
    extract_question_only,
    file_sha256,
    projector_output_width,
    relative_residual_energy,
    rule_no_reference_prompt,
    sequence_forward,
    stable_digest,
)

VERSION = "rule-source-group-postprojector-adapter-v1"
SOURCE_INVARIANT_OBJECTIVE = "source_invariant"


def stable_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_named_paths(values: list[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or name in output:
            raise ValueError(f"invalid or duplicate {option} name: {name!r}")
        output[name] = path
    return output


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    payload = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array or JSONL: {path}")
    return payload


def normalize_source_rows(
    name: str, train_json: Path, image_root: Path, maximum: int, seed: int
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    missing: list[Path] = []
    for row in load_rows(train_json):
        image = str(row.get("image", "")).strip()
        conversations = row.get("conversations")
        if not image or not isinstance(conversations, list) or len(conversations) != 2:
            continue
        absolute_image = (image_root / image).resolve()
        if not absolute_image.is_file():
            missing.append(absolute_image)
            continue
        if not str(conversations[1].get("value", "")).strip():
            continue
        grouped.setdefault(str(absolute_image), []).append(row)
    if missing:
        raise FileNotFoundError(
            f"source {name!r} has {len(missing)} missing images; first={missing[0]}"
        )
    images = sorted(
        grouped, key=lambda item: stable_digest(f"{seed}:{name}:image:{item}")
    )
    if maximum:
        images = images[:maximum]
    selected: list[dict[str, str]] = []
    for image in images:
        rows = sorted(
            grouped[image],
            key=lambda row: stable_digest(
                f"{seed}:{name}:qa:{row.get('id')}:{row['conversations'][0].get('value')}"
            ),
        )
        row = rows[0]
        selected.append({
            "domain": name,
            "id": str(row.get("id", "")),
            "image": image,
            # Deliberately remove RULE alignment reference text: this is image DG.
            "question": extract_question_only(row["conversations"][0]["value"]),
            "answer": str(row["conversations"][1]["value"]).strip(),
        })
    if not selected:
        raise ValueError(f"source {name!r} has no usable rows")
    return selected


def balanced_schedule(
    groups: dict[str, list[dict[str, str]]], steps: int
) -> list[dict[str, dict[str, str]]]:
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError("at least two source groups are required")
    if steps <= 0:
        steps = min(len(groups[name]) for name in names)
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [
        {name: groups[name][step % len(groups[name])] for name in names}
        for step in range(steps)
    ]


def update_group_weights(
    previous: torch.Tensor, losses: torch.Tensor, eta: float
) -> torch.Tensor:
    """Exponentiated-gradient update for distributionally robust source weights."""
    if previous.ndim != 1 or losses.shape != previous.shape:
        raise ValueError("group weight/loss shape mismatch")
    if eta <= 0:
        raise ValueError("group DRO eta must be positive")
    logits = previous.clamp_min(1e-30).log() + eta * losses.detach().float()
    return torch.softmax(logits, dim=0)


def source_excess_risks(
    adapted_losses: torch.Tensor, identity_losses: torch.Tensor
) -> torch.Tensor:
    """Cancel source-specific base difficulty with a stop-gradient reference."""
    if adapted_losses.shape != identity_losses.shape or adapted_losses.ndim != 1:
        raise ValueError("adapted/identity source-loss shapes differ")
    return adapted_losses.float() - identity_losses.detach().float()


def centered_source_hull(
    adapted_losses: torch.Tensor, identity_losses: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entropic worst-source excess risk and its exact per-source weights."""
    if tau <= 0:
        raise ValueError("source-hull tau must be positive")
    excess = source_excess_risks(adapted_losses, identity_losses)
    scaled = excess / tau
    objective = tau * (torch.logsumexp(scaled, dim=0) - math.log(excess.numel()))
    weights = torch.softmax(scaled.detach(), dim=0)
    return objective, weights, excess


def source_view_canvas(image: Image.Image, size: int) -> Image.Image:
    """Preserve the full radiograph before the fixed source-spectrum view."""

    if size <= 0:
        raise ValueError("view-size must be positive")
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        color=(122, 116, 104),
    )


def token_cosine_invariance(
    clean_tokens: torch.Tensor, view_tokens: torch.Tensor
) -> torch.Tensor:
    """Align a source-guided view to a stop-gradient clean clinical anchor."""

    if clean_tokens.shape != view_tokens.shape:
        raise ValueError("clean/view token shapes differ")
    clean = torch.nn.functional.normalize(
        clean_tokens.detach().float(), dim=-1
    )
    view = torch.nn.functional.normalize(view_tokens.float(), dim=-1)
    return (1.0 - (clean * view).sum(dim=-1)).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--test-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=(
            "pooled_erm",
            "source_group_dro",
            "centered_source_hull",
            SOURCE_INVARIANT_OBJECTIVE,
        ),
        required=True,
    )
    parser.add_argument(
        "--source-center",
        type=Path,
        help="Fixed leak-safe X-ray source amplitude for source_invariant.",
    )
    parser.add_argument("--source-view-alpha", type=float, default=0.03)
    parser.add_argument("--source-view-size", type=int, default=384)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--max-images-per-source", type=int, default=64)
    parser.add_argument("--steps", type=int, default=0, help="0 uses the smallest source group once")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-relative-update", type=float, default=0.02)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--group-dro-eta", type=float, default=0.1)
    parser.add_argument("--source-hull-tau", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output", "resume"}
    }


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("train_rule_dg_adapter.py"),
        Path(__file__).with_name("models_alignment.py"),
    ]
    return {str(path): file_sha256(path) for path in paths}


def target_image_paths(path: Path, root: Path) -> set[str]:
    return {
        str((root / str(row["image"])).resolve())
        for row in load_rows(path)
        if str(row.get("image", "")).strip()
    }


def main() -> None:
    args = parse_args()
    if args.identity_weight < 0:
        raise ValueError("identity-weight must be nonnegative")
    if args.rank <= 0 or args.max_relative_update <= 0:
        raise ValueError("rank and max-relative-update must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive")
    if args.group_dro_eta <= 0 or args.source_hull_tau <= 0 or args.save_every <= 0:
        raise ValueError("DRO eta, source-hull tau, and save-every must be positive")
    if args.max_images_per_source < 0 or args.steps < 0:
        raise ValueError("sample limits must be nonnegative")
    if args.objective == SOURCE_INVARIANT_OBJECTIVE:
        if args.source_center is None or not args.source_center.is_file():
            raise FileNotFoundError("source_invariant requires --source-center")
        if not 0.0 < args.source_view_alpha < 0.5:
            raise ValueError("source-view-alpha must lie in (0, 0.5)")
        if args.source_view_size <= 0 or args.invariance_weight <= 0:
            raise ValueError(
                "source-view-size and invariance-weight must be positive"
            )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_jsons = parse_named_paths(args.source_json, "--source-json")
    source_roots = parse_named_paths(args.source_image_root, "--source-image-root")
    if set(source_jsons) != set(source_roots) or len(source_jsons) < 2:
        raise ValueError("source JSON/root names must match and contain at least two groups")
    for path in source_jsons.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in source_roots.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.test_json.is_file() or not args.test_image_root.is_dir():
        raise FileNotFoundError("test JSON or image root does not exist")

    groups = {
        name: normalize_source_rows(
            name, source_jsons[name], source_roots[name],
            args.max_images_per_source, args.seed,
        )
        for name in sorted(source_jsons)
    }
    schedule = balanced_schedule(groups, args.steps)
    source_center = (
        np.load(args.source_center)
        if args.objective == SOURCE_INVARIANT_OBJECTIVE
        else None
    )
    target_images = target_image_paths(args.test_json, args.test_image_root.resolve())
    source_images = {row["image"] for rows in groups.values() for row in rows}
    overlap = sorted(source_images & target_images)
    if overlap:
        raise RuntimeError(f"source/test image leakage detected: {overlap[:3]}")

    fingerprint_payload = {
        "version": VERSION,
        "config": serializable_args(args),
        "source_json_sha256": {
            name: file_sha256(path) for name, path in sorted(source_jsons.items())
        },
        "test_json_sha256": file_sha256(args.test_json),
        "source_center_sha256": (
            file_sha256(args.source_center)
            if args.source_center is not None
            else None
        ),
        "source_names": sorted(groups),
        "selected": {
            name: [
                {"id": row["id"], "image": row["image"], "answer": row["answer"]}
                for row in rows
            ]
            for name, rows in sorted(groups.items())
        },
        "code_sha256": code_hashes(),
        "train_test_image_overlap": 0,
    }
    fingerprint = stable_json_sha256(fingerprint_payload)
    if args.output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume for an identical run")

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    width = projector_output_width(adapter.model)
    module = BoundedResidualBottleneck(
        width, args.rank, args.max_relative_update
    ).to(adapter.model.device)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    names = sorted(groups)
    group_weights = torch.full((len(names),), 1.0 / len(names))
    history: list[dict[str, Any]] = []
    start = 0
    if args.resume and args.output.is_file():
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        if payload.get("version") != VERSION or payload.get("fingerprint") != fingerprint:
            raise RuntimeError("resume checkpoint fingerprint mismatch")
        module.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        group_weights = payload["group_weights"].float()
        history = list(payload["history"])
        start = int(payload["next_step"])

    def save(next_step: int) -> None:
        atomic_torch_save({
            "version": VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "width": width,
            "state_dict": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "group_weights": group_weights,
            "history": history,
            "next_step": next_step,
        }, args.output)

    progress = tqdm(range(start, len(schedule)), desc=args.objective)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        detached_losses: list[float] = []
        relative_means: list[float] = []
        relative_maxima: list[float] = []
        group_metrics: dict[str, dict[str, float]] = {}
        baseline_losses: dict[str, float] = {}
        probe_excesses: dict[str, float] = {}
        hull_task_objective = 0.0
        if args.objective == "centered_source_hull":
            probe_adapted: list[float] = []
            probe_identity: list[float] = []
            for name in names:
                row = schedule[step][name]
                with Image.open(row["image"]) as handle:
                    image = handle.convert("RGB")
                prompt = rule_no_reference_prompt(row["question"])
                input_ids, labels = build_teacher_forcing(
                    adapter, prompt, row["answer"]
                )
                with torch.no_grad():
                    identity_nll, _ = sequence_forward(
                        adapter, image, input_ids, labels, None,
                        adapter_location="post",
                    )
                    adapted_nll, _ = sequence_forward(
                        adapter, image, input_ids, labels, module,
                        adapter_location="post",
                    )
                baseline_losses[name] = float(identity_nll)
                probe_adapted.append(float(adapted_nll))
                probe_identity.append(float(identity_nll))
            hull_value, active_weights, excess = centered_source_hull(
                torch.tensor(probe_adapted), torch.tensor(probe_identity),
                args.source_hull_tau,
            )
            group_weights = active_weights.clone()
            hull_task_objective = float(hull_value)
            probe_excesses = {
                name: float(excess[index]) for index, name in enumerate(names)
            }
        else:
            active_weights = (
                torch.full_like(group_weights, 1.0 / len(names))
                if args.objective
                in {"pooled_erm", SOURCE_INVARIANT_OBJECTIVE}
                else group_weights
            )
        for index, name in enumerate(names):
            row = schedule[step][name]
            with Image.open(row["image"]) as handle:
                image = handle.convert("RGB")
            prompt = rule_no_reference_prompt(row["question"])
            input_ids, labels = build_teacher_forcing(adapter, prompt, row["answer"])
            sequence_nll, _ = sequence_forward(
                adapter, image, input_ids, labels, module, adapter_location="post"
            )
            if module.last_input is None or module.last_output is None:
                raise RuntimeError("adapter did not expose its residual")
            clean_input = module.last_input
            clean_output = module.last_output
            clean_identity = relative_residual_energy(clean_input, clean_output)
            view_nll = sequence_nll.detach().new_zeros(())
            invariance = sequence_nll.detach().new_zeros(())
            view_identity = sequence_nll.detach().new_zeros(())
            if args.objective == SOURCE_INVARIANT_OBJECTIVE:
                assert source_center is not None
                canvas = source_view_canvas(image, args.source_view_size)
                view = source_spectrum_alignment_release2(
                    canvas,
                    source_center,
                    low_frequency_ratio=args.source_view_alpha,
                )
                view_nll, _ = sequence_forward(
                    adapter,
                    view,
                    input_ids,
                    labels,
                    module,
                    adapter_location="post",
                )
                if module.last_input is None or module.last_output is None:
                    raise RuntimeError("adapter did not expose view residual")
                view_identity = relative_residual_energy(
                    module.last_input, module.last_output
                )
                invariance = token_cosine_invariance(
                    clean_output, module.last_output
                )
                task_loss = 0.5 * (sequence_nll + view_nll)
                identity = 0.5 * (clean_identity + view_identity)
            else:
                task_loss = sequence_nll
                identity = clean_identity
            task_weight = active_weights[index].to(sequence_nll.device)
            if args.objective == "centered_source_hull":
                risk = (
                    task_weight * task_loss
                    + (args.identity_weight / len(names)) * identity
                )
            else:
                risk = task_weight * (
                    task_loss
                    + args.identity_weight * identity
                    + (
                        args.invariance_weight * invariance
                        if args.objective == SOURCE_INVARIANT_OBJECTIVE
                        else 0.0
                    )
                )
            risk.backward()
            detached_losses.append(float(sequence_nll.detach()))
            relative_means.append(module.last_mean_relative_norm)
            relative_maxima.append(module.last_max_relative_norm)
            group_metrics[name] = {
                "sequence_nll": float(sequence_nll.detach()),
                "identity": float(identity.detach()),
                "view_sequence_nll": (
                    float(view_nll.detach())
                    if args.objective == SOURCE_INVARIANT_OBJECTIVE
                    else None
                ),
                "token_cosine_invariance": (
                    float(invariance.detach())
                    if args.objective == SOURCE_INVARIANT_OBJECTIVE
                    else None
                ),
                "weight": float(active_weights[index]),
                "mean_relative_update": module.last_mean_relative_norm,
                "max_relative_update": module.last_max_relative_norm,
                "identity_nll": baseline_losses.get(name),
                "excess_nll": probe_excesses.get(name),
            }
        if args.objective == "source_group_dro":
            group_weights = update_group_weights(
                group_weights, torch.tensor(detached_losses), args.group_dro_eta
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(), args.gradient_clip
        )
        if not math.isfinite(float(gradient_norm)):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        mean_identity_penalty = float(np.mean([
            values["identity"] for values in group_metrics.values()
        ]))
        source_hull_objective = (
            hull_task_objective + args.identity_weight * mean_identity_penalty
            if args.objective == "centered_source_hull" else 0.0
        )
        item = {
            "step": step,
            "groups": group_metrics,
            "mean_sequence_nll": float(np.mean(detached_losses)),
            "max_sequence_nll": float(np.max(detached_losses)),
            "source_hull_task_objective": hull_task_objective,
            "mean_identity_penalty": mean_identity_penalty,
            "source_hull_objective": source_hull_objective,
            "gradient_norm": float(gradient_norm),
            "next_group_weights": {
                name: float(group_weights[index]) for index, name in enumerate(names)
            },
            "mean_relative_update": float(np.mean(relative_means)),
            "max_relative_update": float(np.max(relative_maxima)),
        }
        if not all(math.isfinite(float(item[key])) for key in (
            "mean_sequence_nll", "max_sequence_nll", "gradient_norm",
            "mean_relative_update", "max_relative_update",
        )):
            raise FloatingPointError(f"non-finite training metric: {item}")
        history.append(item)
        progress.set_postfix(
            mean=f"{item['mean_sequence_nll']:.3f}",
            worst=f"{item['max_sequence_nll']:.3f}",
        )
        next_step = step + 1
        if next_step % args.save_every == 0 or next_step == len(schedule):
            save(next_step)
        torch.cuda.empty_cache()
    print(json.dumps({
        "output": str(args.output), "version": VERSION,
        "fingerprint": fingerprint, "objective": args.objective,
        "source_sizes": {name: len(groups[name]) for name in names},
        "steps_complete": len(history), "final": history[-1] if history else None,
    }, indent=2))


if __name__ == "__main__":
    main()
