#!/usr/bin/env python3
"""Fatal canary for atomic-correct / binding-wrong medical VLM errors.

The unit of analysis is an image with two reader-supported findings on
opposite sides of the chest.  The probe first checks that Huatuo admits both
finding atoms, then compares two sentences with exactly the same atoms and
length but opposite finding-to-side bindings.  The matched permutation score
cannot improve by deleting findings, shortening the answer, or changing the
positive rate.

This is a phenomenon/mechanism gate, not a paper result.  It deliberately
fails closed before any large experiment or method naming.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from corrected_sgta.run_huatuo_vindr_commitment_probe import (
    dicom_to_pil,
    import_huatuo,
    label_ids,
    layer_logits,
    prepared_embeddings,
    hidden_trajectory,
)


VERSION = "huatuo-binding-conservation-probe-v1"
FOCAL_FINDINGS = {
    "Atelectasis",
    "Calcification",
    "Clavicle fracture",
    "Consolidation",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Mediastinal shift",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Rib fracture",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + ":" + ":".join(values)).encode()).hexdigest()


def normalize_finding(value: str) -> str:
    return value.lower().replace("/", " or ").replace("_", " ").strip()


def build_cases(csv_path: Path, image_root: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []
    image_ids: set[str] = set()
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] not in FOCAL_FINDINGS or not row["x_min"]:
                continue
            rows.append(row)
            image_ids.add(row["image_id"])

    widths: dict[str, int] = {}
    for image_id in sorted(image_ids):
        path = image_root / f"{image_id}.dicom"
        if not path.is_file():
            raise FileNotFoundError(path)
        widths[image_id] = int(pydicom.dcmread(str(path), stop_before_pixels=True).Columns)

    # DICOMs use radiological display: image-right is the patient's left.
    support: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for row in rows:
        center_x = (float(row["x_min"]) + float(row["x_max"])) / 2.0
        side = "left" if center_x >= widths[row["image_id"]] / 2.0 else "right"
        support[row["image_id"]][row["class_name"]][side].add(row["rad_id"])

    candidates: list[dict[str, Any]] = []
    for image_id, findings in support.items():
        unilateral: list[tuple[str, str, int]] = []
        for finding, sides in findings.items():
            for side in ("left", "right"):
                other = "right" if side == "left" else "left"
                if len(sides.get(side, set())) >= 2 and not sides.get(other, set()):
                    unilateral.append((finding, side, len(sides[side])))
        for index, first in enumerate(unilateral):
            for second in unilateral[index + 1 :]:
                if first[0] == second[0] or first[1] == second[1]:
                    continue
                ordered = sorted((first, second), key=lambda item: item[1])
                left = ordered[0] if ordered[0][1] == "left" else ordered[1]
                right = ordered[1] if ordered[1][1] == "right" else ordered[0]
                if left[1] != "left" or right[1] != "right":
                    raise AssertionError("laterality ordering failed")
                candidates.append(
                    {
                        "case_key": f"{image_id}|{left[0]}|{right[0]}",
                        "image_id": image_id,
                        "image": str((image_root / f"{image_id}.dicom").resolve()),
                        "left_finding": left[0],
                        "right_finding": right[0],
                        "left_reader_support": left[2],
                        "right_reader_support": right[2],
                        "columns": widths[image_id],
                    }
                )

    # Avoid letting a single frequent pair type dominate the canary.
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = tuple(sorted((row["left_finding"], row["right_finding"])))
        by_pair[key].append(row)
    for key in by_pair:
        by_pair[key].sort(key=lambda row: stable_key(seed, row["case_key"]))
    pair_order = sorted(by_pair, key=lambda key: stable_key(seed, *key))
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    offset = 0
    while len(selected) < limit:
        added = False
        for key in pair_order:
            if offset >= len(by_pair[key]):
                continue
            row = by_pair[key][offset]
            if row["image_id"] not in used_images:
                selected.append(row)
                used_images.add(row["image_id"])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            raise ValueError(f"only found {len(selected)} image-disjoint cases, requested {limit}")
        offset += 1
    return selected


def presence_prompt(finding: str) -> str:
    return (
        f"Does this chest X-ray show {normalize_finding(finding)}? "
        "Answer with exactly one word: Yes, No, or Maybe."
    )


def oe_prompt(left_finding: str, right_finding: str) -> str:
    return (
        f"This chest X-ray contains {normalize_finding(left_finding)} and "
        f"{normalize_finding(right_finding)}. State which finding is on the patient's "
        "left and which is on the patient's right. Answer in one short sentence."
    )


def scoring_prompt(first: str, second: str) -> str:
    return (
        f"The chest X-ray contains {normalize_finding(first)} and {normalize_finding(second)}. "
        "Complete the laterality statement accurately:"
    )


@torch.inference_mode()
def presence_margin(bot: Any, ids: dict[str, int], image_tensor: torch.Tensor, finding: str) -> float:
    embeddings, attention, positions, _ = prepared_embeddings(
        bot, presence_prompt(finding), image_tensor
    )
    hidden = hidden_trajectory(bot, embeddings, attention, positions)
    logits = layer_logits(bot, hidden, (), ids)[len(hidden) - 1]
    return float(logits["supported"] - logits["refuted"])


@torch.inference_mode()
def continuation_logprob(
    bot: Any,
    image_tensor: torch.Tensor,
    prompt: str,
    continuation: str,
) -> dict[str, Any]:
    embeddings, attention, positions, _ = prepared_embeddings(bot, prompt, image_tensor)
    token_ids = bot.tokenizer.encode(" " + continuation, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("empty continuation tokenization")
    ids = torch.tensor(token_ids, device=embeddings.device, dtype=torch.long).unsqueeze(0)
    candidate_embeddings = bot.model.get_input_embeddings()(ids)
    joined = torch.cat((embeddings, candidate_embeddings), dim=1)
    joined_attention = torch.ones(joined.shape[:2], dtype=attention.dtype, device=attention.device)
    if positions is None:
        joined_positions = None
    else:
        start = int(positions[0, -1]) + 1
        tail = torch.arange(start, start + len(token_ids), device=positions.device).unsqueeze(0)
        joined_positions = torch.cat((positions, tail), dim=1)
    output = bot.model.model(
        input_ids=None,
        attention_mask=joined_attention,
        position_ids=joined_positions,
        inputs_embeds=joined,
        use_cache=False,
        return_dict=True,
    )
    logits = bot.model.lm_head(output.last_hidden_state)
    begin = embeddings.shape[1] - 1
    selected = logits[:, begin : begin + len(token_ids), :]
    logp = F.log_softmax(selected.float(), dim=-1)
    values = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return {
        "sum_logprob": float(values.sum()),
        "mean_logprob": float(values.mean()),
        "token_count": len(token_ids),
        "token_ids": token_ids,
    }


def pair_score(bot: Any, image_tensor: torch.Tensor, left: str, right: str) -> dict[str, Any]:
    # Both hypotheses contain the exact same lexical atoms.  Averaging the two
    # clause orders removes the remaining autoregressive clause-order nuisance.
    hypotheses = {
        "correct": [
            (left, right, f"{normalize_finding(left)} is on the patient's left; {normalize_finding(right)} is on the patient's right."),
            (right, left, f"{normalize_finding(right)} is on the patient's right; {normalize_finding(left)} is on the patient's left."),
        ],
        "swapped": [
            (left, right, f"{normalize_finding(left)} is on the patient's right; {normalize_finding(right)} is on the patient's left."),
            (right, left, f"{normalize_finding(right)} is on the patient's left; {normalize_finding(left)} is on the patient's right."),
        ],
    }
    output: dict[str, Any] = {}
    for name, variants in hypotheses.items():
        cells = [
            continuation_logprob(bot, image_tensor, scoring_prompt(a, b), text)
            for a, b, text in variants
        ]
        output[name] = {
            "variants": cells,
            "mean_sum_logprob": float(np.mean([cell["sum_logprob"] for cell in cells])),
        }
    output["correct_minus_swapped"] = (
        output["correct"]["mean_sum_logprob"] - output["swapped"]["mean_sum_logprob"]
    )
    output["prediction"] = "correct" if output["correct_minus_swapped"] > 0 else "swapped"
    return output


@torch.inference_mode()
def generate_oe(bot: Any, image: Any, prompt: str, max_new_tokens: int) -> str:
    old = dict(bot.gen_kwargs)
    bot.gen_kwargs.update(
        {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "repetition_penalty": 1.0,
            "eos_token_id": bot.tokenizer.eos_token_id,
            "pad_token_id": bot.tokenizer.pad_token_id or bot.tokenizer.eos_token_id,
        }
    )
    try:
        response = bot.inference(prompt, [image])
        return str(response[0] if response else "").strip()
    finally:
        bot.gen_kwargs.clear()
        bot.gen_kwargs.update(old)


def parse_binding(text: str, left: str, right: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    left_name = normalize_finding(left)
    right_name = normalize_finding(right)
    expected = (
        (left_name in normalized and re.search(rf"{re.escape(left_name)}.{{0,45}}\bleft\b", normalized))
        or (left_name in normalized and re.search(rf"\bleft\b.{{0,45}}{re.escape(left_name)}", normalized))
    ) and (
        (right_name in normalized and re.search(rf"{re.escape(right_name)}.{{0,45}}\bright\b", normalized))
        or (right_name in normalized and re.search(rf"\bright\b.{{0,45}}{re.escape(right_name)}", normalized))
    )
    swapped = (
        (left_name in normalized and re.search(rf"{re.escape(left_name)}.{{0,45}}\bright\b", normalized))
        or (left_name in normalized and re.search(rf"\bright\b.{{0,45}}{re.escape(left_name)}", normalized))
    ) and (
        (right_name in normalized and re.search(rf"{re.escape(right_name)}.{{0,45}}\bleft\b", normalized))
        or (right_name in normalized and re.search(rf"\bleft\b.{{0,45}}{re.escape(right_name)}", normalized))
    )
    if expected and not swapped:
        return "correct"
    if swapped and not expected:
        return "swapped"
    return "unparsed"


def bootstrap_accuracy_delta(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, float]:
    # Matched score accuracy minus parsed OE accuracy, restricted to atomic-correct
    # and parseable cases.  The phenomenon gate itself is reported separately.
    eligible = [
        row for row in rows
        if row["status"] == "ok" and row["atomic_correct"] and row["oe_parse"] != "unparsed"
    ]
    if not eligible:
        return {"n": 0, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    delta = np.asarray([
        float(row["permutation_score"]["prediction"] == "correct")
        - float(row["oe_parse"] == "correct")
        for row in eligible
    ])
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(draws)
    ])
    return {
        "n": len(eligible),
        "estimate": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def analyze(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    atomic = [row for row in ok if row["atomic_correct"]]
    parsed = [row for row in atomic if row["oe_parse"] != "unparsed"]
    score_acc = float(np.mean([row["permutation_score"]["prediction"] == "correct" for row in atomic])) if atomic else float("nan")
    oe_acc = float(np.mean([row["oe_parse"] == "correct" for row in parsed])) if parsed else float("nan")
    binding_error = 1.0 - oe_acc if parsed else float("nan")
    phenomenon_pass = len(atomic) >= 8 and len(parsed) >= 6 and binding_error >= 0.10
    improvement = bootstrap_accuracy_delta(ok, draws, seed)
    method_pass = bool(
        phenomenon_pass
        and improvement["n"] >= 6
        and improvement["estimate"] >= 0.10
        and improvement["ci_low"] > 0
        # Relative improvement over a systematically mirrored native answer is
        # not enough: a mitigation must also beat chance in absolute terms.
        and score_acc >= 0.70
    )
    return {
        "version": VERSION,
        "status": "GO_BINDING" if method_pass else ("GO_PHENOMENON_ONLY" if phenomenon_pass else "NO_GO_BINDING_PHENOMENON"),
        "n_ok": len(ok),
        "n_atomic_correct": len(atomic),
        "atomic_correct_rate": len(atomic) / len(ok) if ok else 0.0,
        "n_atomic_correct_parseable_oe": len(parsed),
        "oe_parse_rate_within_atomic": len(parsed) / len(atomic) if atomic else 0.0,
        "oe_binding_accuracy": oe_acc,
        "oe_binding_error": binding_error,
        "permutation_score_accuracy": score_acc,
        "matched_accuracy_improvement": improvement,
        "phenomenon_gate_passed": phenomenon_pass,
        "method_canary_passed": method_pass,
        "radiological_frame_audit": {
            "all_parseable_native_answers_are_exact_mirrors": bool(
                parsed and all(row["oe_parse"] == "swapped" for row in parsed)
            ),
            "interpretation": (
                "if true, first audit patient-centric versus screen-centric laterality; "
                "do not call relative improvement a binding correction"
            ),
        },
        "gate": {
            "phenomenon": "at least 8 atomic-correct, 6 parseable OE, and >=10% native binding error",
            "method_canary": ">=10pp matched improvement with bootstrap CI low >0 and >=70% absolute accuracy",
            "scientific_boundary": "failure closes this Huatuo canary, not all compositional hallucination",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cases = build_cases(args.csv, args.image_root, args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "created_at": now(),
        "model": "HuatuoGPT-Vision-7B",
        "model_dir": str(args.model_dir.resolve()),
        "seed": args.seed,
        "limit": args.limit,
        "cases": cases,
        "selection": "two distinct focal unilateral findings on opposite sides; >=2 bbox annotators each; image-disjoint",
        "intervention": "same-atom matched permutation scoring; average both clause orders",
        "claim_conservation": "finding multiset, laterality multiset, sentence length, and positive claim count are identical",
        "research_role": "fatal phenomenon/mechanism canary; not a mitigation result",
    }
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        old = json.loads(config_path.read_text())
        for key in ("version", "model", "model_dir", "seed", "limit", "cases", "selection", "intervention"):
            if old[key] != config[key]:
                raise RuntimeError(f"resume config drift: {key}")
    else:
        atomic_json(config_path, config)

    raw_path = args.output_dir / "raw.jsonl"
    completed: set[str] = set()
    if raw_path.exists() and args.resume:
        completed = {
            json.loads(line)["case_key"] for line in raw_path.read_text().splitlines() if line.strip()
        }

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    bot.debug = False
    ids = label_ids(bot)
    for index, case in enumerate(cases):
        if case["case_key"] in completed:
            continue
        row: dict[str, Any] = {**case, "version": VERSION, "status": "error"}
        try:
            image = dicom_to_pil(Path(case["image"]))
            image_tensor = torch.stack(bot.get_image_tensors([image])).to(
                device=bot.model.device, dtype=torch.bfloat16
            )
            margins = {
                "left_finding": presence_margin(bot, ids, image_tensor, case["left_finding"]),
                "right_finding": presence_margin(bot, ids, image_tensor, case["right_finding"]),
            }
            answer = generate_oe(
                bot,
                image,
                oe_prompt(case["left_finding"], case["right_finding"]),
                args.max_new_tokens,
            )
            row.update(
                {
                    "status": "ok",
                    "presence_margins": margins,
                    "atomic_correct": all(value > 0 for value in margins.values()),
                    "oe_prompt": oe_prompt(case["left_finding"], case["right_finding"]),
                    "oe_answer": answer,
                    "oe_parse": parse_binding(answer, case["left_finding"], case["right_finding"]),
                    "permutation_score": pair_score(
                        bot, image_tensor, case["left_finding"], case["right_finding"]
                    ),
                    "completed_at": now(),
                }
            )
        except Exception as error:
            row.update({"error": repr(error), "traceback": traceback.format_exc(), "completed_at": now()})
        append_jsonl(raw_path, row)
        completed.add(case["case_key"])
        print(
            f"[{len(completed)}/{len(cases)}] {case['case_key']} status={row['status']} "
            f"atomic={row.get('atomic_correct')} oe={row.get('oe_parse')} "
            f"pair={row.get('permutation_score', {}).get('prediction')}",
            flush=True,
        )

    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    atomic_json(args.output_dir / "analysis.json", analyze(rows, args.bootstrap_draws, args.seed))


if __name__ == "__main__":
    main()
