"""Train a single post-projector adapter with source-counterfactual evidence invariance."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm
from corrected_sgta.anchor_dg import counterfactual_view, evidence_huber_loss, file_sha256, load_style_bank, raw_logit_consistency_loss, stable_sha256
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.train_rule_dg_adapter import BoundedResidualBottleneck, atomic_torch_save, build_teacher_forcing, projector_output_width, relative_residual_energy, rule_no_reference_prompt, sequence_forward
from corrected_sgta.train_rule_source_group_adapter import balanced_schedule, normalize_source_rows, parse_named_paths

VERSION = "anchor-dg-postprojector-adapter-v2"
OBJECTIVES = ("task_only", "generic_augmentation", "raw_logit_consistency", "anchor_dg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANCHOR-DG using source data only.")
    parser.add_argument("--source-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--forbidden-json", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--forbidden-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--heldout-domain", action="append", default=[])
    parser.add_argument("--style-bank", type=Path)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--style-rho", type=float)
    parser.add_argument("--style-beta", type=float)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-relative-update", type=float, default=0.02)
    parser.add_argument("--max-images-per-source", type=int, default=64)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]


def resolved_images(path: Path, root: Path) -> set[str]:
    result = set()
    for row in load_json_rows(path):
        raw = str(row.get("image", "")).strip()
        if raw:
            candidate = Path(raw)
            result.add(str((candidate if candidate.is_absolute() else root / candidate).resolve()))
    return result


def generic_view(image: Image.Image, sample_id: str, seed: int, step: int) -> Image.Image:
    value = int(stable_sha256({"id": sample_id, "seed": seed, "step": step})[:8], 16)
    contrast = (0.85, 1.15)[value % 2]
    brightness = (0.92, 1.08)[(value // 2) % 2]
    return ImageEnhance.Brightness(ImageEnhance.Contrast(image).enhance(contrast)).enhance(brightness)


def serializable(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in {"resume", "output"}}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source_json = parse_named_paths(args.source_json, "--source-json")
    source_roots = parse_named_paths(args.source_image_root, "--source-image-root")
    forbidden_json = parse_named_paths(args.forbidden_json, "--forbidden-json")
    forbidden_roots = parse_named_paths(args.forbidden_image_root, "--forbidden-image-root")
    if set(source_json) != set(source_roots) or set(forbidden_json) != set(forbidden_roots):
        raise ValueError("manifest/root names must match")
    heldout = set(args.heldout_domain)
    if heldout.intersection(source_json):
        raise ValueError(f"held-out domain used for training: {sorted(heldout.intersection(source_json))}")
    groups = {name: normalize_source_rows(name, source_json[name], source_roots[name], args.max_images_per_source, args.seed) for name in sorted(source_json)}
    source_paths = {row["image"] for rows in groups.values() for row in rows}
    forbidden_paths = set().union(*(resolved_images(forbidden_json[name], forbidden_roots[name]) for name in forbidden_json))
    overlap = source_paths.intersection(forbidden_paths)
    if overlap:
        raise RuntimeError(f"source/forbidden image leakage: {sorted(overlap)[:3]}")
    uses_views = args.objective != "task_only"
    uses_source_style = args.objective in {"raw_logit_consistency", "anchor_dg"}
    bank = None
    if uses_source_style:
        if args.style_bank is None or args.gate_json is None or args.style_rho is None or args.style_beta is None:
            raise ValueError("view objectives require --style-bank, --gate-json, --style-rho, and --style-beta")
        bank = load_style_bank(args.style_bank, heldout)
        if set(bank.domains) != set(groups):
            raise ValueError("style-bank domains must exactly match training domains")
        gate = json.loads(args.gate_json.read_text())
        if (not gate.get("gate_pass") or float(gate.get("selected_rho")) != args.style_rho or float(gate.get("selected_beta")) != args.style_beta):
            raise RuntimeError("training refused: intervention gate did not approve style-rho/style-beta")
    schedule = balanced_schedule(groups, args.steps)
    source_override = any(
        bool(row.get("unverified_source_override"))
        for path in source_json.values() for row in load_json_rows(path)
    )
    provenance = {
        "version": VERSION, "config": serializable(args),
        "source_sha256": {name: file_sha256(path) for name, path in source_json.items()},
        "forbidden_sha256": {name: file_sha256(path) for name, path in forbidden_json.items()},
        "style_bank_sha256": file_sha256(args.style_bank) if args.style_bank else None,
        "gate_sha256": file_sha256(args.gate_json) if args.gate_json else None,
        "code_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in ("train_anchor_dg.py", "anchor_dg.py", "train_rule_dg_adapter.py")},
        "source_forbidden_overlap": 0, "target_labels_accessed": False,
        "unverified_source_override": bool(
            source_override or (bank and bank.metadata.get("unverified_source_override"))
        ),
    }
    fingerprint = stable_sha256(provenance)
    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    adapter.model.eval()
    width = projector_output_width(adapter.model)
    module = BoundedResidualBottleneck(width, args.rank, args.max_relative_update).to(adapter.model.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    start = 0
    if args.resume and args.output.is_file():
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
        module.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload["history"])
        start = int(payload["next_step"])
    elif args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(next_step: int) -> None:
        atomic_torch_save({"version": VERSION, "fingerprint": fingerprint, "fingerprint_payload": provenance, "config": serializable(args), "width": width, "state_dict": module.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "next_step": next_step}, args.output)

    try:
        for step in tqdm(range(start, len(schedule)), desc=args.objective):
            optimizer.zero_grad(set_to_none=True)
            metrics = []
            for domain, row in sorted(schedule[step].items()):
                with Image.open(row["image"]) as handle:
                    image = handle.convert("RGB")
                prompt = rule_no_reference_prompt(row["question"])
                ids, labels = build_teacher_forcing(adapter, prompt, row["answer"])
                clean_nll, clean_logits, token_ids = sequence_forward(adapter, image, ids, labels, module, adapter_location="post", return_token_ids=True)
                identity = relative_residual_energy(module.last_input, module.last_output)
                view_nll = clean_nll.new_zeros(())
                consistency = clean_nll.new_zeros(())
                style_domain = None
                if uses_views:
                    if args.objective == "generic_augmentation":
                        view = generic_view(image, row["id"], args.seed, step)
                        style_domain = "generic"
                    else:
                        assert bank is not None
                        view, style_domain = counterfactual_view(image, bank, domain, row["id"], args.seed, step, args.style_rho, args.style_beta)
                    view_nll, view_logits, view_ids = sequence_forward(adapter, view, ids, labels, module, adapter_location="post", return_token_ids=True)
                    if not torch.equal(token_ids, view_ids):
                        raise RuntimeError("clean/view gold-token alignment failed")
                    view_identity = relative_residual_energy(module.last_input, module.last_output)
                    identity = 0.5 * (identity + view_identity)
                    if args.objective == "raw_logit_consistency":
                        consistency = raw_logit_consistency_loss(clean_logits, view_logits)
                    elif args.objective == "anchor_dg":
                        consistency = evidence_huber_loss(clean_logits, view_logits, token_ids, args.huber_delta)
                    task = 0.5 * (clean_nll + view_nll)
                else:
                    task = clean_nll
                loss = (task + args.identity_weight * identity + args.evidence_weight * consistency) / len(groups)
                loss.backward()
                metrics.append({"domain": domain, "style_domain": style_domain, "clean_nll": float(clean_nll.detach()), "view_nll": float(view_nll.detach()), "consistency": float(consistency.detach()), "identity": float(identity.detach())})
            grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), args.gradient_clip))
            optimizer.step()
            history.append({"step": step, "gradient_norm": grad_norm, "groups": metrics, "mean_relative_update": module.last_mean_relative_norm, "max_relative_update": module.last_max_relative_norm})
            if (step + 1) % args.save_every == 0:
                save(step + 1)
        save(len(schedule))
    finally:
        adapter.close()
    print(json.dumps({"output": str(args.output), "fingerprint": fingerprint, "objective": args.objective, "steps": len(schedule)}, indent=2))


if __name__ == "__main__":
    main()
