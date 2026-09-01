#!/usr/bin/env python3
"""Probe whether Findings text substitutes for image evidence at Impression.

The target Impression is fixed.  We score it under correct versus naturally
shuffled images and three assistant-prefix conditions: no Findings, its own
Findings, and length-matched Findings from another report.  The mismatched
prefix is the position/extra-language control; it prevents a generic long-
sequence visual-forgetting effect from being mislabeled evidence substitution.

This is a teacher-forced mechanism probe, not a clinical quality benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
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

from corrected_sgta.run_huatuo_evidence_dg_probe import (
    IGNORE_INDEX,
    REPORT_PROMPT,
    answer_ids,
    import_huatuo,
    load_report_rows,
    sha256_file,
)


VERSION = "huatuo-section-substitution-v1"
DEFAULT_REPO = Path("/home/dbw/ANCHOR")
DEFAULT_MODEL = Path("/home/dbw/models/HuatuoGPT-Vision-7B")
DEFAULT_HUATUO = Path("/home/dbw/HuatuoGPT-Vision")
DEFAULT_OUTPUT = Path(
    "/home/dbw/ANCHOR/corrected_runs/section_substitution/huatuo_mimic_n24_v1"
)
SECTION_RE = re.compile(r"(?i)\b(impression|findings?)\s*:")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def split_sections(report: str) -> dict[str, str]:
    """Return explicit Findings/Impression spans independent of their order."""
    matches = list(SECTION_RE.finditer(str(report)))
    output: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        key = "findings" if name.startswith("finding") else "impression"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        value = re.sub(r"\s+", " ", report[match.end() : end]).strip(" .\n\t")
        if value and key not in output:
            output[key] = value
    return output


def select_rows(repo: Path, sample_count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = load_report_rows(repo)["mimic"]
    eligible: list[dict[str, Any]] = []
    for row in source:
        sections = split_sections(str(row["reference"]))
        finding_words = len(sections.get("findings", "").split())
        impression_words = len(sections.get("impression", "").split())
        if (
            Path(row["image"]).is_file()
            and 5 <= finding_words <= 140
            and 3 <= impression_words <= 80
        ):
            eligible.append({**row, **sections})
    eligible.sort(key=lambda row: stable_key(seed, str(row["id"])))
    chosen = eligible[:sample_count]
    if len(chosen) < sample_count:
        raise RuntimeError(f"requested {sample_count}, found {len(chosen)} eligible reports")

    # Same-corpus natural image negatives.  Findings negatives are selected by
    # closest word count, with hash as the deterministic tie-breaker.
    for index, row in enumerate(chosen):
        row["shuffled_image"] = chosen[(index + 1) % len(chosen)]["image"]
        alternatives = [candidate for candidate in chosen if candidate["id"] != row["id"]]
        mismatch = min(
            alternatives,
            key=lambda candidate: (
                abs(len(candidate["findings"].split()) - len(row["findings"].split())),
                stable_key(seed + 1, str(candidate["id"])),
            ),
        )
        row["mismatched_findings"] = mismatch["findings"]
        row["mismatched_findings_id"] = mismatch["id"]
    return chosen, {
        "source_rows": len(source),
        "eligible": len(eligible),
        "selected": len(chosen),
    }


@torch.inference_mode()
def continuation_nll(
    bot: Any,
    prompt: str,
    assistant_prefix: str,
    target: str,
    image_tensor: torch.Tensor,
    max_prefix_tokens: int,
    max_target_tokens: int,
) -> dict[str, float | int]:
    """Score only target tokens while teacher-forcing an assistant prefix."""
    prompt = bot.insert_image_placeholder(prompt, 1)
    prompt_ids = bot.preprocess(
        bot.get_conv_without_history(prompt), return_tensors="pt"
    ).to(bot.model.device)
    if int((prompt_ids < 0).sum()) != 1:
        raise RuntimeError("prompt must contain exactly one image token")
    prefix_ids = answer_ids(bot, assistant_prefix, max_prefix_tokens)
    target_ids = answer_ids(bot, target, max_target_tokens)
    full = torch.cat((prompt_ids, prefix_ids, target_ids), dim=0)
    labels = torch.full_like(full, IGNORE_INDEX)
    labels[-target_ids.numel() :] = target_ids
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
    target_labels = expanded_labels[:, 1:][mask]
    hidden = output.last_hidden_state[:, :-1][mask]
    logits = hidden.to(bot.model.get_output_embeddings().weight.dtype) @ (
        bot.model.get_output_embeddings().weight.T
    )
    losses = F.cross_entropy(logits.float(), target_labels, reduction="none")
    return {
        "prefix_token_count": int(prefix_ids.numel()),
        "target_token_count": int(target_labels.numel()),
        "mean_nll": float(losses.mean().cpu()),
        "total_nll": float(losses.sum().cpu()),
    }


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(draws)]
    )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def analyze(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    ok = [row for row in records if row.get("status") == "ok"]
    if not ok:
        raise RuntimeError("no successful records")
    image_delta: dict[str, np.ndarray] = {}
    for prefix in ("none", "matched", "mismatched"):
        image_delta[prefix] = np.asarray(
            [
                row["scores"][prefix]["shuffled_image"]["mean_nll"]
                - row["scores"][prefix]["correct_image"]["mean_nll"]
                for row in ok
            ],
            dtype=float,
        )
    attenuation = image_delta["none"] - image_delta["matched"]
    semantic_specific = image_delta["mismatched"] - image_delta["matched"]
    generic_position = image_delta["none"] - image_delta["mismatched"]
    matched_prefix_gain = np.asarray(
        [
            row["scores"]["mismatched"]["correct_image"]["mean_nll"]
            - row["scores"]["matched"]["correct_image"]["mean_nll"]
            for row in ok
        ]
    )
    statistics = {
        "image_identity_benefit": {
            key: bootstrap(value, seed + index)
            for index, (key, value) in enumerate(image_delta.items(), start=1)
        },
        "matched_attenuation_vs_no_prefix": bootstrap(attenuation, seed + 10),
        "semantic_specific_attenuation_vs_mismatched": bootstrap(
            semantic_specific, seed + 11
        ),
        "generic_position_attenuation": bootstrap(generic_position, seed + 12),
        "matched_prefix_nll_gain_vs_mismatched": bootstrap(matched_prefix_gain, seed + 13),
    }
    gate = {
        "matched_attenuation_ci_above_zero": statistics[
            "matched_attenuation_vs_no_prefix"
        ]["ci_low"]
        > 0,
        "semantic_specific_ci_above_zero": statistics[
            "semantic_specific_attenuation_vs_mismatched"
        ]["ci_low"]
        > 0,
    }
    gate["pass"] = all(gate.values())
    return {
        "n": len(ok),
        "statistics": statistics,
        "frozen_gate": gate,
        "interpretation_guard": (
            "Passing supports prefix-conditioned image dependence, not clinical "
            "hallucination reduction. Failure prunes the section-substitution candidate."
        ),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--max-prefix-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / "raw.jsonl"
    rows, audit = select_rows(args.repo_root, args.samples, args.seed)
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "model": str(args.model_dir),
        "huatuo_root": str(args.huatuo_root),
        "selection": audit,
        "prompt": REPORT_PROMPT,
        "primary_estimand": (
            "(shuffle-correct image NLL delta with mismatched Findings) minus "
            "(shuffle-correct image NLL delta with matched Findings)"
        ),
        "frozen_gate": (
            "bootstrap 95% lower CI > 0 for matched-vs-none attenuation and "
            "matched-vs-length-matched-mismatched semantic attenuation"
        ),
        "seed": args.seed,
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
            "image": str(row["image"]),
            "shuffled_image": str(row["shuffled_image"]),
            "mismatched_findings_id": row["mismatched_findings_id"],
            "findings": row["findings"],
            "mismatched_findings": row["mismatched_findings"],
            "impression": row["impression"],
            "status": "error",
        }
        try:
            correct = torch.stack(bot.get_image_tensors([str(row["image"])]))
            shuffled = torch.stack(bot.get_image_tensors([str(row["shuffled_image"])]))
            tensors = {
                "correct_image": correct.to(bot.model.device, dtype=torch.bfloat16),
                "shuffled_image": shuffled.to(bot.model.device, dtype=torch.bfloat16),
            }
            prefixes = {
                "none": "Impression:",
                "matched": f"Findings: {row['findings']}\nImpression:",
                "mismatched": (
                    f"Findings: {row['mismatched_findings']}\nImpression:"
                ),
            }
            record["scores"] = {
                prefix_name: {
                    image_name: continuation_nll(
                        bot,
                        REPORT_PROMPT,
                        prefix,
                        row["impression"],
                        tensor,
                        args.max_prefix_tokens,
                        args.max_target_tokens,
                    )
                    for image_name, tensor in tensors.items()
                }
                for prefix_name, prefix in prefixes.items()
            }
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        append_jsonl(raw_path, record)
        print(f"[{index + 1}/{len(rows)}] {row['id']} {record['status']}", flush=True)

    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    summary = {"version": VERSION, **analyze(records, args.seed), "config": config}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
