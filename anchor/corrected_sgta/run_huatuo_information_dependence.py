#!/usr/bin/env python3
"""Test whether unavailable target content is less image-dependent.

For every report we pair one explicit out-of-bundle sentence (Tier A) with a
finding-bearing sentence from the same report.  We then compare teacher-forced
NLL under the correct image, a natural shuffled image from the same corpus, and
a zero-visual control.  The within-report pairing controls patient/report style;
the shuffled image is the primary causal contrast.

This is a mechanism probe, not a report-quality benchmark.  Regex indicators
define an auditable target class but do not define clinical factuality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from scipy.stats import wilcoxon

from corrected_sgta.audit_information_mismatch import (
    NON_VISUAL,
    TIER_A,
    TIER_B,
    labels_for,
    split_sentences,
)
from corrected_sgta.run_huatuo_evidence_dg_probe import (
    IGNORE_INDEX,
    REPORT_PROMPT,
    answer_ids,
    import_huatuo,
    load_report_rows,
    sha256_file,
)


VERSION = "huatuo-information-dependence-v1"
DEFAULT_REPO = Path("/home/dbw/ANCHOR")
DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_OUTPUT = Path(
    "/home/dbw/ANCHOR/corrected_runs/information_mismatch/huatuo_dependence_v1"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(seed: int, domain: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{domain}:{identifier}".encode()).hexdigest()


def load_finding_aliases(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    aliases = {
        alias.lower()
        for values in payload["findings"].values()
        for alias in values
    }
    return tuple(sorted(aliases, key=lambda value: (-len(value), value)))


def contains_finding(sentence: str, aliases: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", sentence, re.IGNORECASE)
        for alias in aliases
    )


def choose_sentence_pair(
    report: str, aliases: tuple[str, ...]
) -> tuple[str, str, list[str]] | None:
    sentences = split_sentences(report)
    unavailable = [
        (sentence, labels_for(sentence, TIER_A))
        for sentence in sentences
        if labels_for(sentence, TIER_A)
    ]
    visible = [
        sentence
        for sentence in sentences
        if not labels_for(sentence, TIER_A)
        and not labels_for(sentence, TIER_B)
        and not labels_for(sentence, NON_VISUAL)
        and contains_finding(sentence, aliases)
    ]
    if not unavailable or not visible:
        return None
    # Prefer the shortest unavailable span to reduce mixed visual content, then
    # length-match its within-report control.
    tier_sentence, labels = min(
        unavailable, key=lambda item: (len(item[0].split()), item[0])
    )
    visible_sentence = min(
        visible,
        key=lambda sentence: (
            abs(len(sentence.split()) - len(tier_sentence.split())),
            len(sentence.split()),
            sentence,
        ),
    )
    return tier_sentence, visible_sentence, labels


def select_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    aliases: tuple[str, ...],
    per_domain: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for domain, rows in rows_by_domain.items():
        candidates = []
        for row in rows:
            image = Path(row["image"])
            pair = choose_sentence_pair(str(row["reference"]), aliases)
            if image.is_file() and pair is not None:
                tier, visible, labels = pair
                candidates.append(
                    {
                        **row,
                        "tier_a_sentence": tier,
                        "visible_sentence": visible,
                        "tier_a_labels": labels,
                    }
                )
        candidates.sort(
            key=lambda row: stable_key(seed, domain, str(row["id"]))
        )
        chosen = candidates[:per_domain]
        if len(chosen) < min(4, per_domain):
            continue
        # A deterministic cyclic derangement provides a natural same-domain
        # negative image while never using the paired report's image.
        for index, row in enumerate(chosen):
            row["shuffled_image"] = chosen[(index + 1) % len(chosen)]["image"]
        selected.extend(chosen)
        audit[domain] = {
            "rows": len(rows),
            "eligible": len(candidates),
            "selected": len(chosen),
        }
    if len(selected) < 8:
        raise RuntimeError(f"only {len(selected)} eligible paired reports")
    return selected, audit


@torch.inference_mode()
def sequence_nll(
    bot: Any,
    prompt: str,
    text: str,
    image_tensor: torch.Tensor,
    maximum: int,
) -> dict[str, float | int]:
    prompt = bot.insert_image_placeholder(prompt, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt), return_tensors="pt"
    ).to(bot.model.device)
    if int((prompt_ids < 0).sum()) != 1:
        raise RuntimeError("prompt must contain exactly one image token")
    targets = answer_ids(bot, text, maximum)
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
    mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
    target_ids = expanded_labels[:, 1:][mask]
    hidden = output.last_hidden_state[:, :-1][mask]
    logits = hidden.to(bot.model.get_output_embeddings().weight.dtype) @ (
        bot.model.get_output_embeddings().weight.T
    )
    losses = F.cross_entropy(logits.float(), target_ids, reduction="none")
    return {
        "token_count": int(target_ids.numel()),
        "mean_nll": float(losses.mean().cpu()),
        "total_nll": float(losses.sum().cpu()),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def score_delta(scores: dict[str, dict[str, float | int]]) -> dict[str, float]:
    correct = float(scores["correct"]["mean_nll"])
    return {
        "shuffle_minus_correct": float(scores["shuffled"]["mean_nll"]) - correct,
        "zero_minus_correct": float(scores["zero"]["mean_nll"]) - correct,
    }


def bootstrap_paired(values: np.ndarray, seed: int, draws: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def analyze(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [row for row in records if row.get("status") == "ok"]
    output: dict[str, Any] = {"n": len(ok), "domains": {}}
    for condition in ("shuffle_minus_correct", "zero_minus_correct"):
        tier = np.asarray([row["tier_a_delta"][condition] for row in ok])
        visible = np.asarray([row["visible_delta"][condition] for row in ok])
        paired = visible - tier
        try:
            test = wilcoxon(paired, alternative="greater")
            wilcoxon_result = {
                "statistic": float(test.statistic),
                "pvalue_one_sided": float(test.pvalue),
            }
        except ValueError:
            wilcoxon_result = {"statistic": None, "pvalue_one_sided": None}
        output[condition] = {
            "tier_a": bootstrap_paired(tier, seed + 1),
            "visible_control": bootstrap_paired(visible, seed + 2),
            "paired_visible_minus_tier_a": bootstrap_paired(paired, seed + 3),
            "wilcoxon_visible_greater": wilcoxon_result,
        }
    for domain in sorted({row["domain"] for row in ok}):
        part = [row for row in ok if row["domain"] == domain]
        output["domains"][domain] = {
            "n": len(part),
            "mean_shuffle_delta_tier_a": float(
                np.mean([row["tier_a_delta"]["shuffle_minus_correct"] for row in part])
            ),
            "mean_shuffle_delta_visible": float(
                np.mean([row["visible_delta"]["shuffle_minus_correct"] for row in part])
            ),
        }
    output["interpretation_guard"] = (
        "A positive paired delta means finding-bearing controls depend more on the "
        "correct image than explicit out-of-bundle sentences. Token NLL is a "
        "mechanism diagnostic, not clinical correctness."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-domain", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    ontology = args.repo_root / "configs/missing_third_state_vindr_ontology.json"
    aliases = load_finding_aliases(ontology)
    rows, selection_audit = select_rows(
        load_report_rows(args.repo_root), aliases, args.samples_per_domain, args.seed
    )
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "model": str(args.model_dir),
        "huatuo_root": str(args.huatuo_root),
        "n": len(rows),
        "selection": selection_audit,
        "prompt": REPORT_PROMPT,
        "primary_contrast": "same-corpus natural shuffled image minus correct image",
        "secondary_contrast": "zero visual tensor minus correct image",
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
            "tier_a_sentence": row["tier_a_sentence"],
            "tier_a_labels": row["tier_a_labels"],
            "visible_sentence": row["visible_sentence"],
            "status": "error",
        }
        try:
            correct = torch.stack(bot.get_image_tensors([str(row["image"])]))
            shuffled = torch.stack(
                bot.get_image_tensors([str(row["shuffled_image"])]))
            correct = correct.to(bot.model.device, dtype=torch.bfloat16)
            shuffled = shuffled.to(bot.model.device, dtype=torch.bfloat16)
            tensors = {"correct": correct, "shuffled": shuffled, "zero": torch.zeros_like(correct)}
            tier_scores = {
                name: sequence_nll(bot, REPORT_PROMPT, row["tier_a_sentence"], tensor, args.max_tokens)
                for name, tensor in tensors.items()
            }
            visible_scores = {
                name: sequence_nll(bot, REPORT_PROMPT, row["visible_sentence"], tensor, args.max_tokens)
                for name, tensor in tensors.items()
            }
            record.update(
                {
                    "status": "ok",
                    "tier_a_scores": tier_scores,
                    "visible_scores": visible_scores,
                    "tier_a_delta": score_delta(tier_scores),
                    "visible_delta": score_delta(visible_scores),
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(f"[{index + 1}/{len(rows)}] {row['domain']} {record['status']}", flush=True)

    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    summary = {"version": VERSION, **analyze(records, args.seed), "config": config}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
