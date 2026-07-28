#!/usr/bin/env python3
"""ANCHOR layer-expert pilot.

Training-free full-vocabulary decoding inspired by VISTA/MoLE: mix the final
decoder distribution with an early/mid-layer "second opinion" distribution.
This is not a yes/no logit evaluator; it changes full natural-language greedy
generation and then evaluates generated sentences with the official parser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.evaluate_medheval_answers import evaluate_rows, rule_pope_prediction
from corrected_sgta.models_oe import Generation, LlavaMedOEAdapter
from corrected_sgta.run_anchor_gauge_pilot import load_json_or_jsonl
from corrected_sgta.run_anchor_flow_sgta_gate import normalize_target_record

VERSION = "anchor-layer-expert-pilot-v1"


def normalize_rows(path: Path, *, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_json_or_jsonl(path):
        if "prompt" in row and "image" in row and ("answer" in row or "reference" in row):
            rows.append(
                {
                    "id": str(row.get("id", row.get("qid", len(rows)))),
                    "image": str(row["image"]),
                    "prompt": str(row["prompt"]).replace("<image>", "").strip(),
                    "answer": str(row.get("answer", row.get("reference"))).strip(),
                    "domain": str(row.get("domain", row.get("dataset", "mimic"))),
                    "patient_id": str(row.get("patient_id", row.get("subject_id", row.get("id", len(rows))))),
                    "question_type": str(row.get("question_type", "binary")),
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
    detail = evaluate_rows(
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
    parsed = detail.get("prediction")
    gt = detail.get("ground_truth")
    if parsed is None:
        parsed = rule_pope_prediction(text)
        gt = rule_pope_prediction(reference)
    return {
        "parsed_answer": parsed,
        "ground_truth": gt,
        "correct": bool(parsed is not None and gt is not None and parsed == gt),
        "parseable": parsed is not None,
    }


@torch.inference_mode()
def generate_layer_expert(
    adapter: LlavaMedOEAdapter,
    image: Image.Image,
    prompt: str,
    *,
    alpha: float,
    expert_layer: int,
    max_new_tokens: int,
    seed: int,
) -> Generation:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = adapter._process_images([image])
    if isinstance(pixels, list):
        pixels = [value.to(adapter.model.device, dtype=adapter.model.dtype) for value in pixels]
    else:
        pixels = pixels.to(adapter.model.device, dtype=adapter.model.dtype)
    _, position_ids, attention_mask, _, embeds, _ = adapter.model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, pixels, image_sizes=[image.size]
    )
    output = adapter.model.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=embeds,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = output.past_key_values
    generated: list[int] = []
    nll: list[float] = []
    eos = adapter.tokenizer.eos_token_id
    weight = adapter.model.get_output_embeddings().weight
    norm = adapter.model.model.norm
    for step in range(max_new_tokens):
        final_h = output.last_hidden_state[:, -1]
        hidden_tuple = output.hidden_states
        layer_index = expert_layer if expert_layer >= 0 else len(hidden_tuple) + expert_layer
        layer_index = max(0, min(layer_index, len(hidden_tuple) - 1))
        expert_h = norm(hidden_tuple[layer_index][:, -1])
        final_logits = final_h.to(weight.dtype) @ weight.T
        expert_logits = expert_h.to(weight.dtype) @ weight.T
        logits = (1.0 - alpha) * final_logits.float() + alpha * expert_logits.float()
        logp = torch.log_softmax(logits[0], dim=-1)
        token = int(torch.argmax(logp).item())
        nll.append(float(-logp[token].item()))
        if eos is not None and token == eos:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        token_ids = torch.tensor([[token]], dtype=torch.long, device=adapter.model.device)
        past_length = int(past.get_seq_length() if hasattr(past, "get_seq_length") else past[0][0].shape[-2])
        next_attention = torch.ones((1, past_length + 1), dtype=torch.long, device=adapter.model.device)
        next_position = torch.full((1, 1), past_length, dtype=torch.long, device=adapter.model.device)
        output = adapter.model.model(
            input_ids=token_ids,
            attention_mask=next_attention,
            position_ids=next_position,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values
    text = adapter.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return Generation(text=text, uncertainty=float(np.mean(nll)) if nll else float("inf"), token_count=len(generated))


def summarize(records: Iterable[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    rows = list(records)
    base = [r["methods"]["alpha_0.0_layer_-1"]["eval"]["correct"] for r in rows]
    out: dict[str, Any] = {"version": VERSION, "n": len(rows), "methods": {}}
    for method in methods:
        vals = [r["methods"][method]["eval"]["correct"] for r in rows]
        parse = [r["methods"][method]["eval"]["parseable"] for r in rows]
        out["methods"][method] = {
            "accuracy": float(np.mean(vals)) if vals else 0.0,
            "parse_rate": float(np.mean(parse)) if parse else 0.0,
            "delta_vs_greedy": float(np.mean(vals) - np.mean(base)) if vals else 0.0,
            "rescue": int(sum((not b) and v for b, v in zip(base, vals))),
            "harm": int(sum(b and (not v) for b, v in zip(base, vals))),
            "mean_words": float(np.mean([len(r["methods"][method]["text"].split()) for r in rows])) if rows else 0.0,
        }
    out["best_method"] = max(
        out["methods"], key=lambda name: (out["methods"][name]["accuracy"], -out["methods"][name]["harm"])
    ) if rows else None
    out["continue_gate"] = bool(
        rows
        and any(
            m["delta_vs_greedy"] >= 0.03 and m["rescue"] > m["harm"]
            for name, m in out["methods"].items()
            if name != "alpha_0.0_layer_-1"
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/mmedrag/test/vqa/mimic_test.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/medheval/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("corrected_runs/final_anchor_layer_expert_pilot_v1"))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--layers", type=int, nargs="+", default=[-8, -12, -16])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "ce_raw.jsonl"
    summary_path = args.output_dir / "ce_summary.json"
    rows = normalize_rows(args.manifest, max_samples=args.max_samples)
    methods: list[tuple[str, float, int]] = [("alpha_0.0_layer_-1", 0.0, -1)]
    for layer in args.layers:
        for alpha in args.alphas:
            if alpha > 0:
                methods.append((f"alpha_{alpha}_layer_{layer}", alpha, layer))
    fingerprint = stable_json_sha256(
        {
            "version": VERSION,
            "manifest": str(args.manifest),
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "layers": args.layers,
            "alphas": args.alphas,
            "full_vocab_generation": True,
            "no_yes_no_logits_as_results": True,
        }
    )
    adapter = LlavaMedOEAdapter(conv_mode="mistral_instruct")
    records: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record.get("id"))] = record
        records.extend(completed.values())
    mode = "a" if args.resume else "w"
    with raw_path.open(mode) as handle:
        for idx, row in enumerate(tqdm(rows, desc="ANCHOR-LayerExpert CE")):
            if str(row["id"]) in completed:
                continue
            with Image.open(resolve_image_path(row["image"], args.image_root)) as src:
                image = src.convert("RGB")
            method_payload: dict[str, Any] = {}
            for name, alpha, layer in methods:
                gen = generate_layer_expert(
                    adapter,
                    image,
                    row["prompt"],
                    alpha=alpha,
                    expert_layer=layer,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed + idx,
                )
                method_payload[name] = {
                    "text": gen.text,
                    "token_count": gen.token_count,
                    "uncertainty": gen.uncertainty,
                    "alpha": alpha,
                    "layer": layer,
                    "eval": parse_ce(gen.text, row["answer"], row["prompt"]),
                }
            record = {
                "version": VERSION,
                "fingerprint": fingerprint,
                "id": row["id"],
                "patient_id": row["patient_id"],
                "image": row["image"],
                "prompt": row["prompt"],
                "reference": row["answer"],
                "methods": method_payload,
                "target_labels_used_for_generation": False,
                "uses_yes_no_logits_for_prediction": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    summary = summarize(records, [name for name, _, _ in methods])
    summary["fingerprint"] = fingerprint
    summary["raw"] = str(raw_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
