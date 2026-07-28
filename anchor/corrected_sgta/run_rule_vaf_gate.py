#!/usr/bin/env python3
"""Small RULE/MIMIC gate for visual-odds amplification.

The intervention is the minimal mathematical version of ClearSight/VISTA-style
visual information steering: in selected middle layers, add the same scalar eta
to every visual key logit. This multiplies the total visual-vs-nonvisual
attention odds by exp(eta) while preserving the normalized distribution inside
the image tokens exactly.

The script generates full natural-language answers and evaluates them with the
RULE sentence parser. It never uses yes/no label logits for prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows
from corrected_sgta.infer_rule_dg_adapter import (
    load_rows,
    official_prompt,
    repair_jsonl_tail,
    successful_qids,
)
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.run_rule_attention_mass_anchor import visual_interval
from corrected_sgta.train_rule_dg_adapter import file_sha256, process_image


VERSION = "rule-vaf-visual-odds-gate-v1"
DEFAULT_LAYERS = tuple(range(9, 15))


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class VisualOddsHook(AbstractContextManager):
    def __init__(self, model: Any, layers: tuple[int, ...], eta: float):
        self.model = model
        self.layers = layers
        self.eta = float(eta)
        self.original: dict[int, Any] = {}
        self.image_start = 0
        self.image_end = 0

    def configure(self, image_start: int, image_end: int) -> None:
        self.image_start = int(image_start)
        self.image_end = int(image_end)

    def __enter__(self):
        from transformers.models.mistral import modeling_mistral

        model_layers = self.model.model.layers
        for layer_index in self.layers:
            attn = model_layers[layer_index].self_attn
            self.original[layer_index] = attn.forward

            def forward(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                _layer_index=layer_index,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                query, key = modeling_mistral.apply_rotary_pos_emb(query, key, cos, sin)
                if past_key_values is not None:
                    key, value = past_key_values.update(key, value, module.layer_idx)
                key = modeling_mistral.repeat_kv(key, module.num_key_value_groups)
                value = modeling_mistral.repeat_kv(value, module.num_key_value_groups)
                weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
                if attention_mask is not None:
                    weights = weights + attention_mask
                if self.eta and 0 <= self.image_start < self.image_end <= weights.shape[-1]:
                    weights = weights.clone()
                    weights[..., self.image_start : self.image_end] += self.eta
                weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
                weights = F.dropout(
                    weights,
                    p=0.0 if not module.training else module.attention_dropout,
                    training=module.training,
                )
                output = torch.matmul(weights, value).transpose(1, 2).contiguous()
                output = output.reshape(*input_shape, -1).contiguous()
                return module.o_proj(output), weights

            attn.forward = types.MethodType(forward, attn)
        return self

    def __exit__(self, exc_type, exc, tb):
        for layer_index, forward in self.original.items():
            self.model.model.layers[layer_index].self_attn.forward = forward
        self.original = {}
        return False


@torch.inference_mode()
def decode(adapter, hook, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria

    input_ids = adapter._prompt_ids(prompt).to(adapter.model.device)
    pixels = process_image(adapter, image)
    _, _, _, _, embeds, _ = adapter.model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        pixels,
        image_sizes=[image.size],
    )
    image_start, image_end = visual_interval(input_ids, embeds.shape[1])
    hook.configure(image_start, image_end)
    conversation = conv_templates[adapter.conv_mode].copy()
    stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
    stopping = KeywordsStoppingCriteria([stop], adapter.tokenizer, input_ids)
    output_ids = adapter.model.generate(
        input_ids,
        images=pixels,
        image_sizes=[image.size],
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=[stopping],
        pad_token_id=adapter.tokenizer.eos_token_id,
    )
    text = adapter.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text[: -len(stop)].strip() if stop and text.endswith(stop) else text


def parse_layers(value: str) -> tuple[int, ...]:
    if value == "middle":
        return DEFAULT_LAYERS
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/mimic_cxr_rule/questions.target.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("/root/autodl-tmp/MedHEval/images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eta", type=float, action="append", default=[0.0, 0.25, 0.5])
    parser.add_argument("--layers", type=parse_layers, default=DEFAULT_LAYERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.questions)[: args.max_samples]
    fingerprint_payload = {
        "version": VERSION,
        "questions_sha256": file_sha256(args.questions),
        "image_root": str(args.image_root.resolve()),
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "eta": args.eta,
        "layers": list(args.layers),
        "prediction": "full generated sentence -> RULE parser",
        "uses_yes_no_logits": False,
    }
    fingerprint = hashlib.sha256(stable_json(fingerprint_payload).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.resume and meta_path.is_file():
        if json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
    elif args.output.exists() and args.output.stat().st_size:
        raise FileExistsError("output exists; use --resume")
    atomic_json(meta_path, {"fingerprint": fingerprint, **fingerprint_payload})
    repair_jsonl_tail(args.output)
    done = successful_qids(args.output) if args.resume else set()

    adapter = LlavaMedAlignmentAdapter(conv_mode="vicuna_v1")
    for p in adapter.model.parameters():
        p.requires_grad_(False)
    adapter.model.eval()
    try:
        with args.output.open("a") as handle:
            for row in tqdm(rows, desc="vaf-gate"):
                qid = str(row.get("question_id", row.get("qid")))
                if qid in done:
                    continue
                image_path = args.image_root / str(row["image"])
                with Image.open(image_path) as im:
                    image = im.convert("RGB")
                prompt = official_prompt(row, "no_reference")
                texts = {}
                for eta in args.eta:
                    with VisualOddsHook(adapter.model, args.layers, eta) as hook:
                        texts[str(eta)] = decode(adapter, hook, image, prompt, args.max_new_tokens)
                record = {
                    "status": "ok",
                    "question_id": qid,
                    "question": row["question"],
                    "answer": row["answer"],
                    "image": row["image"],
                    "texts": texts,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        adapter.close()

    outputs = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    summary = {
        "version": VERSION,
        "fingerprint": fingerprint,
        "n": len(outputs),
        "metrics": {},
    }
    for eta in args.eta:
        answers = [
            {"question_id": row["question_id"], "text": row["texts"][str(eta)]}
            for row in outputs
        ]
        metrics, records = evaluate_rule_rows(rows[: len(outputs)], answers)
        summary["metrics"][str(eta)] = metrics
        summary["metrics"][str(eta)]["rescue_vs_0"] = None
        if str(eta) != "0.0" and "0.0" in summary["metrics"]:
            by_id = {
                str(r["question_id"]): r
                for r in records
            }
            base_answers = [
                {"question_id": row["question_id"], "text": row["texts"]["0.0"]}
                for row in outputs
            ]
            _, base_records = evaluate_rule_rows(rows[: len(outputs)], base_answers)
            base = {str(r["question_id"]): r for r in base_records}
            rescue = sum(
                (not bool(base[k]["explicit_ground_truth_correct"]))
                and bool(by_id[k]["explicit_ground_truth_correct"])
                for k in by_id
            )
            harm = sum(
                bool(base[k]["explicit_ground_truth_correct"])
                and (not bool(by_id[k]["explicit_ground_truth_correct"]))
                for k in by_id
            )
            summary["metrics"][str(eta)]["rescue_vs_0"] = {"rescue": rescue, "harm": harm, "net": rescue - harm}
    atomic_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps({"summary": str(args.output.with_suffix(".summary.json")), "n": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
