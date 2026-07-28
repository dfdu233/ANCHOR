#!/usr/bin/env python3
"""Evaluate MiniMind-V on RULE/MIMIC binary VQA with generated sentences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

ANCHOR_ROOT = Path(__file__).resolve().parents[2]
MINIMIND_ROOT_DEFAULT = Path('/root/autodl-tmp/minimind-v')
if str(ANCHOR_ROOT) not in sys.path:
    sys.path.insert(0, str(ANCHOR_ROOT))

from anchor.corrected_sgta.evaluate_rule_vqa import evaluate_rule_rows  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    image = row.get('image') or row.get('img_name')
    if image is None:
        raise ValueError('missing image field')
    path = image_root / str(image)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def init_minimind(minimind_root: Path, weight: str, save_dir: str, device: str):
    sys.path.insert(0, str(minimind_root))
    from eval_vlm import init_model  # type: ignore

    class Args:
        load_from = 'model'
        save_dir = 'out'
        hidden_size = 768
        num_hidden_layers = 8
        use_moe = 0
        max_new_tokens = 64
        temperature = 0.0
        top_p = 1.0
        image_dir = './dataset/eval_images/'
        show_speed = 0
        open_thinking = 0

    args = Args()
    args.weight = weight
    args.save_dir = save_dir
    args.device = device
    old = Path.cwd()
    try:
        import os
        os.chdir(minimind_root)
        model, tokenizer, preprocess = init_model(args)
    finally:
        import os
        os.chdir(old)
    return model, tokenizer, preprocess


@torch.inference_mode()
def generate_one(model, tokenizer, preprocess, image: Image.Image, question: str, max_new_tokens: int, device: str) -> str:
    from model.model_vlm import MiniMindVLM  # type: ignore

    prompt = '<image>\n' + question.strip() + '\nAnswer in one concise sentence.'
    image_token = model.config.image_special_token * model.config.image_token_len
    messages = [{'role': 'user', 'content': prompt.replace('<image>', image_token)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
    pixel_values = MiniMindVLM.image2tensor(image.convert('RGB'), preprocess)
    pixel_values = {k: v.to(device) for k, v in pixel_values.items()} if hasattr(pixel_values, 'keys') else pixel_values.to(device)
    out = model.generate(
        inputs=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pixel_values=pixel_values,
    )
    return tokenizer.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimind-root', type=Path, default=MINIMIND_ROOT_DEFAULT)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='out')
    parser.add_argument('--questions', type=Path, default=Path('/root/autodl-tmp/ANCHOR/data/rule/test/mimic_test.jsonl'))
    parser.add_argument('--image-root', type=Path, default=Path('/root/autodl-tmp/MedHEval/images'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    rows = load_jsonl(args.questions)
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, preprocess = init_minimind(args.minimind_root, args.weight, args.save_dir, args.device)

    answers = []
    for row in tqdm(rows, desc='MiniMind RULE/MIMIC'):
        try:
            image = Image.open(resolve_image(row, args.image_root))
            text = generate_one(model, tokenizer, preprocess, image, str(row.get('question') or ''), args.max_new_tokens, args.device)
        except Exception as exc:
            text = f'[ERROR] {type(exc).__name__}: {exc}'
        answers.append({'question_id': row.get('question_id'), 'answer': text, 'text': text})
        with (args.out_dir / 'answers.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(answers[-1], ensure_ascii=False) + '\n')

    metrics, records = evaluate_rule_rows(rows, answers)
    metrics.update({'weight': args.weight, 'limit': args.limit, 'max_new_tokens': args.max_new_tokens})
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.out_dir / 'records.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metrics['rule_normalized'], indent=2))


if __name__ == '__main__':
    main()
