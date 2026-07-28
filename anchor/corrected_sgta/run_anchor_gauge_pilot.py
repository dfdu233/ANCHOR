#!/usr/bin/env python3
"""ANCHOR-Gauge pilot: source/DG motivated visual-attention steering.

This runner is intentionally small and independent.  It tests whether a
VISTA/ClearSight-style visual information gain can change complete generated
answers on RULE/MIMIC samples without using yes/no logits as predictions.

The DG story is diagnostic here: SGTA/FedDG views can be used to expose output
instability, but the intervention is a single training-free visual-flow gain in
the LLM attention stack.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.evaluate_medheval_answers import evaluate_rows, rule_pope_prediction
from corrected_sgta.models_oe import LlavaMedOEAdapter
from corrected_sgta.run_anchor_flow_sgta_gate import normalize_target_record

VERSION = "anchor-gauge-pilot-v1"
DEFAULT_OUTPUT = Path("corrected_runs/final_anchor_gauge_pilot_v1")


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "questions", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported manifest shape: {path}")


def normalize_rows(path: Path, *, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_json_or_jsonl(path):
        if "prompt" in row and "image" in row and ("reference" in row or "answer" in row):
            rows.append(
                {
                    "id": str(row.get("id")),
                    "image": str(row.get("image")),
                    "prompt": str(row.get("prompt")),
                    "answer": str(row.get("reference", row.get("answer"))),
                    "patient_id": str(row.get("patient_id", row.get("id"))),
                    "question_type": "binary",
                    "raw": row,
                }
            )
        else:
            rows.append(
                normalize_target_record(
                    row,
                    task="ce",
                    default_domain="mimic",
                    default_prompt="",
                    require_answer=True,
                )
            )
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def parse_ce(text: str, reference: str, prompt: str) -> dict[str, Any]:
    result = evaluate_rows(
        [
            {
                "qid": "sample",
                "question": prompt,
                "ground_truth": reference,
                "text": text,
                "question_type": "binary",
            }
        ]
    )["details"][0]
    parsed = result.get("prediction")
    gt = result.get("ground_truth")
    if parsed is None:
        parsed = rule_pope_prediction(text)
        gt = rule_pope_prediction(reference)
    return {
        "parsed_answer": parsed,
        "ground_truth": gt,
        "correct": bool(parsed is not None and gt is not None and parsed == gt),
        "parseable": parsed is not None,
    }


def _image_token_start(adapter: LlavaMedOEAdapter, prompt: str) -> int:
    from llava.constants import IMAGE_TOKEN_INDEX

    input_ids = adapter._prompt_ids(prompt)[0].tolist()
    try:
        return input_ids.index(IMAGE_TOKEN_INDEX)
    except ValueError as exc:
        raise RuntimeError("prompt does not contain an image token") from exc


def _image_token_len(adapter: LlavaMedOEAdapter, image: Image.Image, prompt: str) -> int:
    ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = adapter._process_images([image])
    if isinstance(pixels, list):
        pixels = [value.to(adapter.model.device, dtype=adapter.model.dtype) for value in pixels]
    else:
        pixels = pixels.to(adapter.model.device, dtype=adapter.model.dtype)
    _, _, _, _, embeds, _ = adapter.model.prepare_inputs_labels_for_multimodal(
        ids, None, None, None, None, pixels, image_sizes=[image.size]
    )
    # One image placeholder token is replaced by the full visual token block.
    return int(embeds.shape[1] - ids.shape[1] + 1)


@contextmanager
def visual_attention_gain(adapter: LlavaMedOEAdapter, *, image_start: int, image_len: int, gain: float):
    """Add log(gain) attention bias from generated/text queries to image keys."""

    if gain <= 0:
        raise ValueError("gain must be positive")
    bias = float(math.log(gain))
    patched: list[tuple[Any, Any]] = []

    def make_forward(module, original_forward):
        def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            if abs(bias) > 1e-12 and image_len > 0:
                q_len = hidden_states.shape[-2]
                # In prefill, kv_len == q_len.  During decoding with cache, kv_len
                # is the full cached prefix plus one new query.
                if past_key_values is not None:
                    try:
                        kv_len = int(past_key_values.get_seq_length(self.layer_idx)) + q_len
                    except Exception:
                        kv_len = image_start + image_len
                else:
                    kv_len = q_len
                end = min(image_start + image_len, kv_len)
                if image_start < end:
                    if attention_mask is None:
                        attention_mask = torch.zeros(
                            (hidden_states.shape[0], 1, q_len, kv_len),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    else:
                        attention_mask = attention_mask.to(dtype=hidden_states.dtype).clone()
                        if attention_mask.shape[-2] == 1 and q_len != 1:
                            attention_mask = attention_mask.expand(-1, -1, q_len, -1).clone()
                    # Only generated/text queries are biased in prefill; single-token
                    # decode queries are always text queries.
                    q_start = max(0, end) if q_len == kv_len else 0
                    attention_mask[:, :, q_start:, image_start:end] += bias
            return original_forward(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        return types.MethodType(forward, module)

    for module in adapter.model.model.layers:
        attn = module.self_attn
        original = attn.forward
        patched.append((attn, original))
        attn.forward = make_forward(attn, original)
    try:
        yield
    finally:
        for attn, original in patched:
            attn.forward = original


@torch.inference_mode()
def generate_with_gain(
    adapter: LlavaMedOEAdapter,
    image: Image.Image,
    prompt: str,
    *,
    gain: float,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    image_start = _image_token_start(adapter, prompt)
    image_len = _image_token_len(adapter, image, prompt)
    with visual_attention_gain(adapter, image_start=image_start, image_len=image_len, gain=gain):
        generation = adapter._generate_once(
            image=image,
            prompt=prompt,
            count=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )[0]
    return {
        "text": generation.text,
        "token_count": generation.token_count,
        "uncertainty": generation.uncertainty,
        "image_start": image_start,
        "image_len": image_len,
        "gain": gain,
    }


def summarize(records: Iterable[dict[str, Any]], gains: list[float]) -> dict[str, Any]:
    rows = list(records)
    out: dict[str, Any] = {"version": VERSION, "n": len(rows), "methods": {}}
    base = [r["methods"]["gain_1.0"]["eval"]["correct"] for r in rows]
    for gain in gains:
        key = f"gain_{gain}"
        vals = [r["methods"][key]["eval"]["correct"] for r in rows]
        parse = [r["methods"][key]["eval"]["parseable"] for r in rows]
        out["methods"][key] = {
            "accuracy": float(np.mean(vals)) if vals else 0.0,
            "parse_rate": float(np.mean(parse)) if parse else 0.0,
            "delta_vs_greedy": float(np.mean(vals) - np.mean(base)) if vals else 0.0,
            "rescue": int(sum((not b) and v for b, v in zip(base, vals))),
            "harm": int(sum(b and (not v) for b, v in zip(base, vals))),
            "mean_words": float(np.mean([len(r["methods"][key]["text"].split()) for r in rows])) if rows else 0.0,
        }
    out["best_gain_by_accuracy"] = max(
        out["methods"], key=lambda name: (out["methods"][name]["accuracy"], -out["methods"][name]["harm"])
    ) if rows else None
    out["continue_gate"] = bool(
        rows
        and any(
            item["delta_vs_greedy"] >= 0.03 and item["rescue"] > item["harm"]
            for name, item in out["methods"].items()
            if name != "gain_1.0"
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("corrected_runs/final_anchor_flow_sgta_gate_v1_ce8_strong_v2/ce_raw.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/medheval/images"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "ce_raw.jsonl"
    summary_path = args.output_dir / "ce_summary.json"

    rows = normalize_rows(args.manifest, max_samples=args.max_samples)
    fingerprint = stable_json_sha256(
        {
            "version": VERSION,
            "manifest": str(args.manifest),
            "image_root": str(args.image_root),
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "gains": args.gains,
            "no_yes_no_logits_as_results": True,
            "training_free": True,
        }
    )
    adapter = LlavaMedOEAdapter(conv_mode="mistral_instruct")
    records: list[dict[str, Any]] = []
    with raw_path.open("w") as handle:
        for idx, row in enumerate(tqdm(rows, desc="ANCHOR-Gauge CE")):
            image_path = resolve_image_path(row["image"], args.image_root)
            with Image.open(image_path) as src:
                image = src.convert("RGB")
            methods: dict[str, Any] = {}
            for gain in args.gains:
                gen = generate_with_gain(
                    adapter,
                    image,
                    row["prompt"],
                    gain=gain,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed + idx,
                )
                gen["eval"] = parse_ce(gen["text"], row["answer"], row["prompt"])
                methods[f"gain_{gain}"] = gen
            record = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "id": row["id"],
                "patient_id": row["patient_id"],
                "image": row["image"],
                "prompt": row["prompt"],
                "reference": row["answer"],
                "methods": methods,
                "target_labels_used_for_generation": False,
                "uses_yes_no_logits_for_prediction": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    summary = summarize(records, args.gains)
    summary["fingerprint"] = fingerprint
    summary["raw"] = str(raw_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
