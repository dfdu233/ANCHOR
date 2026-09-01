#!/usr/bin/env python3
"""Within-sentence causal probe for source-marker versus finding tokens."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from scipy.stats import wilcoxon

from corrected_sgta.audit_information_mismatch import TIER_A, labels_for, split_sentences
from corrected_sgta.run_huatuo_evidence_dg_probe import (
    IGNORE_INDEX,
    REPORT_PROMPT,
    import_huatuo,
    load_report_rows,
    sha256_file,
)
from corrected_sgta.run_huatuo_information_dependence import (
    DEFAULT_HUATUO,
    DEFAULT_MODEL,
    DEFAULT_REPO,
    bootstrap_paired,
    load_finding_aliases,
    stable_key,
)


VERSION = "huatuo-information-token-dependence-v1"
DEFAULT_OUTPUT = Path(
    "/home/dbw/ANCHOR/corrected_runs/information_mismatch/huatuo_token_dependence_v1"
)


def overlap(offset: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = offset
    return end > start and any(start < span_end and end > span_start for span_start, span_end in spans)


def source_spans(sentence: str) -> list[tuple[int, int]]:
    spans = {
        match.span()
        for patterns in TIER_A.values()
        for pattern in patterns
        for match in pattern.finditer(sentence)
    }
    return sorted(spans)


def finding_spans(sentence: str, aliases: tuple[str, ...]) -> list[tuple[int, int]]:
    spans = []
    for alias in aliases:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
        spans.extend(match.span() for match in pattern.finditer(sentence))
    # Longest aliases are already first; discard spans contained in another hit.
    unique = sorted(set(spans), key=lambda span: (span[0], -(span[1] - span[0])))
    return [
        span
        for index, span in enumerate(unique)
        if not any(other[0] <= span[0] and other[1] >= span[1] for other in unique[:index])
    ]


def token_masks(
    sentence: str,
    offsets: list[tuple[int, int]],
    aliases: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    source = source_spans(sentence)
    finding = finding_spans(sentence, aliases)
    source_mask = np.asarray([overlap(offset, source) for offset in offsets], dtype=bool)
    finding_mask = np.asarray([overlap(offset, finding) for offset in offsets], dtype=bool)
    # A token cannot serve as both the tested source marker and control finding.
    finding_mask &= ~source_mask
    return source_mask, finding_mask


def select_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    aliases: tuple[str, ...],
    per_domain: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    audit = {}
    for domain, rows in rows_by_domain.items():
        candidates = []
        for row in rows:
            if not Path(row["image"]).is_file():
                continue
            valid = []
            for sentence in split_sentences(str(row["reference"])):
                if labels_for(sentence, TIER_A) and source_spans(sentence) and finding_spans(sentence, aliases):
                    valid.append(sentence)
            if valid:
                sentence = min(valid, key=lambda value: (len(value.split()), value))
                candidates.append(
                    {
                        **row,
                        "sentence": sentence,
                        "tier_a_labels": labels_for(sentence, TIER_A),
                    }
                )
        candidates.sort(key=lambda row: stable_key(seed, domain, str(row["id"])))
        chosen = candidates[:per_domain]
        if len(chosen) < min(4, per_domain):
            continue
        for index, row in enumerate(chosen):
            row["shuffled_image"] = chosen[(index + 1) % len(chosen)]["image"]
        selected.extend(chosen)
        audit[domain] = {"rows": len(rows), "eligible": len(candidates), "selected": len(chosen)}
    if len(selected) < 8:
        raise RuntimeError(f"only {len(selected)} token-paired reports")
    return selected, audit


@torch.inference_mode()
def token_category_nll(
    bot: Any,
    sentence: str,
    image_tensor: torch.Tensor,
    aliases: tuple[str, ...],
    maximum: int,
) -> dict[str, Any]:
    encoded = bot.tokenizer(
        sentence,
        add_special_tokens=False,
        truncation=True,
        max_length=maximum,
        return_offsets_mapping=True,
    )
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    source_mask, finding_mask = token_masks(sentence, offsets, aliases)
    if not source_mask.any() or not finding_mask.any():
        raise RuntimeError("sentence has no non-overlapping source/finding tokens")
    targets = torch.tensor(encoded["input_ids"], device=bot.model.device, dtype=torch.long)
    prompt = bot.insert_image_placeholder(REPORT_PROMPT, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt), return_tensors="pt"
    ).to(bot.model.device)
    full = torch.cat((prompt_ids, targets), dim=0)
    labels = torch.full_like(full, IGNORE_INDEX)
    labels[-targets.numel() :] = targets
    attention = torch.ones_like(full, dtype=torch.bool)
    _, positions, expanded_attention, _, embeddings, expanded_labels = (
        bot.model.prepare_inputs_labels_for_multimodal_new(
            [full], None, [attention], None, [labels], image_tensor
        )
    )
    output = bot.model.model(
        input_ids=None,
        attention_mask=expanded_attention,
        position_ids=positions,
        inputs_embeds=embeddings,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    target_positions = expanded_labels[:, 1:].ne(IGNORE_INDEX)
    target_ids = expanded_labels[:, 1:][target_positions]
    hidden = output.last_hidden_state[:, :-1][target_positions]
    logits = hidden.to(bot.model.get_output_embeddings().weight.dtype) @ bot.model.get_output_embeddings().weight.T
    losses = F.cross_entropy(logits.float(), target_ids, reduction="none").cpu().numpy()
    if len(losses) != len(offsets):
        raise RuntimeError(f"token alignment mismatch: losses={len(losses)} offsets={len(offsets)}")
    return {
        "token_count": len(losses),
        "source_token_count": int(source_mask.sum()),
        "finding_token_count": int(finding_mask.sum()),
        "source_mean_nll": float(losses[source_mask].mean()),
        "finding_mean_nll": float(losses[finding_mask].mean()),
        "source_text": [sentence[start:end] for start, end in source_spans(sentence)],
        "finding_text": [sentence[start:end] for start, end in finding_spans(sentence, aliases)],
    }


def deltas(scores: dict[str, dict[str, Any]], category: str) -> dict[str, float]:
    key = f"{category}_mean_nll"
    correct = float(scores["correct"][key])
    return {
        "shuffle_minus_correct": float(scores["shuffled"][key]) - correct,
        "zero_minus_correct": float(scores["zero"][key]) - correct,
    }


def analyze(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [row for row in records if row.get("status") == "ok"]
    output: dict[str, Any] = {"n": len(ok), "domains": {}}
    for condition in ("shuffle_minus_correct", "zero_minus_correct"):
        source = np.asarray([row["source_delta"][condition] for row in ok])
        finding = np.asarray([row["finding_delta"][condition] for row in ok])
        paired = finding - source
        try:
            test = wilcoxon(paired, alternative="greater")
            test_result = {"statistic": float(test.statistic), "pvalue_one_sided": float(test.pvalue)}
        except ValueError:
            test_result = {"statistic": None, "pvalue_one_sided": None}
        output[condition] = {
            "source_marker": bootstrap_paired(source, seed + 11),
            "finding": bootstrap_paired(finding, seed + 12),
            "paired_finding_minus_source": bootstrap_paired(paired, seed + 13),
            "wilcoxon_finding_greater": test_result,
        }
    for domain in sorted({row["domain"] for row in ok}):
        part = [row for row in ok if row["domain"] == domain]
        output["domains"][domain] = {
            "n": len(part),
            "mean_paired_shuffle_delta": float(np.mean([
                row["finding_delta"]["shuffle_minus_correct"] - row["source_delta"]["shuffle_minus_correct"]
                for row in part
            ])),
        }
    output["interpretation_guard"] = (
        "Within-sentence token likelihood isolates lexical source markers from finding terms. "
        "It is a causal representation diagnostic, not a clinical truth label."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-domain", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    ontology = args.repo_root / "configs/missing_third_state_vindr_ontology.json"
    aliases = load_finding_aliases(ontology)
    rows, audit = select_rows(load_report_rows(args.repo_root), aliases, args.samples_per_domain, args.seed)
    config = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model_dir),
        "n": len(rows),
        "selection": audit,
        "primary_contrast": "within-sentence finding vs source-marker dependence on correct over same-domain shuffled image",
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "ontology_sha256": sha256_file(ontology),
        "code_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    for index, row in enumerate(rows):
        record: dict[str, Any] = {
            "version": VERSION,
            "id": row["id"],
            "domain": row["domain"],
            "image": str(row["image"]),
            "shuffled_image": str(row["shuffled_image"]),
            "sentence": row["sentence"],
            "tier_a_labels": row["tier_a_labels"],
            "status": "error",
        }
        try:
            correct = torch.stack(bot.get_image_tensors([str(row["image"])] )).to(bot.model.device, dtype=torch.bfloat16)
            shuffled = torch.stack(bot.get_image_tensors([str(row["shuffled_image"])] )).to(bot.model.device, dtype=torch.bfloat16)
            tensors = {"correct": correct, "shuffled": shuffled, "zero": torch.zeros_like(correct)}
            scores = {
                name: token_category_nll(bot, row["sentence"], tensor, aliases, args.max_tokens)
                for name, tensor in tensors.items()
            }
            record.update({
                "status": "ok",
                "scores": scores,
                "source_delta": deltas(scores, "source"),
                "finding_delta": deltas(scores, "finding"),
            })
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        with raw_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{index + 1}/{len(rows)}] {row['domain']} {record['status']}", flush=True)
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    summary = {"version": VERSION, **analyze(records, args.seed), "config": config}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
