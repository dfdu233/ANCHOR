"""Evaluate whether matched image-text alignment learned conditional evidence.

This is a mechanism diagnostic, not a target-domain accuracy benchmark.  It
compares complete-answer sequence NLL on one frozen, source-held-out manifest
while swapping only the visual merger learned by each controlled lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from anchor.corrected_sgta.run_center_native_qwen import (
    ManifestDataset,
    QwenCollator,
    load_merger,
    load_model,
    merger_parameters,
)


VERSION = "alignment-contraction-heldout-nll-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_sample_sequence_nll(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean complete-answer NLL and supervised token count per sample."""

    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_loss = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    token_count = valid.sum(dim=1)
    sequence_nll = (token_loss * valid).sum(dim=1) / token_count.clamp_min(1)
    return sequence_nll, token_count


def parse_condition(value: str) -> tuple[str, Path | None]:
    name, separator, path = value.partition("=")
    if not name:
        raise argparse.ArgumentTypeError("condition name cannot be empty")
    if not separator:
        if name != "base":
            raise argparse.ArgumentTypeError(
                "only the base condition may omit '=CHECKPOINT'"
            )
        return name, None
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint not found: {checkpoint}")
    return name, checkpoint


def cluster_bootstrap_mean_interval(
    values: np.ndarray,
    clusters: list[str],
    draws: int = 5000,
    seed: int = 2027,
) -> list[float]:
    """Bootstrap a paired mean while keeping source figure groups intact."""

    unique = sorted(set(clusters))
    indices = {
        cluster: np.asarray(
            [index for index, value in enumerate(clusters) if value == cluster]
        )
        for cluster in unique
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([indices[cluster] for cluster in selected])
        samples.append(float(values[sample].mean()))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def restore_merger(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    current = dict(model.named_parameters())
    for name, value in state.items():
        current[name].data.copy_(value.to(current[name].device))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    dataset = ManifestDataset(args.manifest)
    indices = [
        index
        for index, row in enumerate(dataset.rows)
        if not args.strict_cxr_only or row.get("is_strict_cxr")
    ]
    if args.limit:
        indices = indices[: args.limit]
    selected = Subset(dataset, indices)
    id_to_group = {
        str(dataset.rows[index]["id"]): str(
            dataset.rows[index].get("group_id", dataset.rows[index]["id"])
        )
        for index in indices
    }
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=QwenCollator(processor, args.max_length),
    )
    model = load_model(args.model, train=False).to("cuda").eval()
    base_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in merger_parameters(model)
    }
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for name, checkpoint in args.condition:
        restore_merger(model, base_state)
        if checkpoint is not None:
            load_merger(model, checkpoint)
        provenance[name] = {
            "checkpoint": (
                str(checkpoint.resolve()) if checkpoint is not None else None
            ),
            "checkpoint_sha256": (
                sha256(checkpoint) if checkpoint is not None else None
            ),
        }
        records: list[dict[str, Any]] = []
        for batch in loader:
            sample_ids = list(batch.pop("anchor_ids"))
            batch.pop("anchor_is_cxr")
            batch = {
                key: value.to("cuda", non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
            nll, token_count = per_sample_sequence_nll(
                output.logits, batch["labels"]
            )
            records.extend(
                {
                    "id": sample_id,
                    "group_id": id_to_group[str(sample_id)],
                    "sequence_nll": float(sample_nll),
                    "supervised_tokens": int(sample_tokens),
                }
                for sample_id, sample_nll, sample_tokens in zip(
                    sample_ids,
                    nll.cpu(),
                    token_count.cpu(),
                    strict=True,
                )
            )
        rows_by_condition[name] = records
    expected_ids = {
        name: [row["id"] for row in records]
        for name, records in rows_by_condition.items()
    }
    if len({tuple(ids) for ids in expected_ids.values()}) != 1:
        raise RuntimeError("conditions did not evaluate identical source samples")
    summary = {
        name: {
            "n": len(records),
            "mean_sequence_nll": float(
                np.mean([row["sequence_nll"] for row in records])
            ),
            "median_sequence_nll": float(
                np.median([row["sequence_nll"] for row in records])
            ),
            "mean_supervised_tokens": float(
                np.mean([row["supervised_tokens"] for row in records])
            ),
        }
        for name, records in rows_by_condition.items()
    }
    paired = {}
    for first_name, second_name in itertools.combinations(
        rows_by_condition, 2
    ):
        first = rows_by_condition[first_name]
        second = rows_by_condition[second_name]
        differences = np.asarray(
            [
                left["sequence_nll"] - right["sequence_nll"]
                for left, right in zip(first, second, strict=True)
            ]
        )
        paired[f"{first_name}_minus_{second_name}"] = {
            "mean_sequence_nll_difference": float(differences.mean()),
            "source_group_cluster_bootstrap_ci95": (
                cluster_bootstrap_mean_interval(
                    differences,
                    [str(row["group_id"]) for row in first],
                    draws=args.bootstrap_draws,
                )
            ),
        }
    result = {
        "version": VERSION,
        "definition": (
            "mean teacher-forced NLL over every supervised answer token"
        ),
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "strict_cxr_only": args.strict_cxr_only,
        "limit": args.limit,
        "n": len(indices),
        "conditions": provenance,
        "summary": summary,
        "paired_comparisons": paired,
        "records": rows_by_condition,
        "claim_ceiling": (
            "source-held-out conditional-information diagnostic; not "
            "target-domain generation accuracy"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_condition)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-cxr-only", action="store_true")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if not args.condition:
        parser.error("at least one --condition is required")
    names = [name for name, _ in args.condition]
    if len(names) != len(set(names)):
        parser.error("condition names must be unique")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
