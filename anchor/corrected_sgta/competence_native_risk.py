"""Competence-native risk: estimate VLM reliable visual support from calibration correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.protocol_v2 import ProtocolError, file_sha256, ground_truth_index, resolve_image, task_kind

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = 'competence-native-risk-v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--greedy-eval', required=True, type=Path)
    parser.add_argument('--mitigation-eval', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-samples', type=int, default=128)
    parser.add_argument('--max-image-side', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--question-type', default='binary')
    parser.add_argument('--train-frac', type=float, default=0.4)
    parser.add_argument('--max-token-bank', type=int, default=12000)
    parser.add_argument('--knn-k', type=int, default=7)
    return parser.parse_args()


def unit(x: np.ndarray, axis=-1, eps=1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
    bu = unit(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
    return float(1.0 - np.dot(au, bu))


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def acc(values):
    return float(np.mean(values)) if values else None


def load_eval(path: Path):
    payload = json.loads(path.read_text())
    return {str(d['question_id']): d for d in payload['details']}, payload


def eligible_rows(rows, question_type: str | None, seed: int, max_samples: int):
    kept = []
    for row in rows:
        try:
            if task_kind(row) == 'open':
                continue
            if question_type and row.get('question_type') != question_type:
                continue
            ground_truth_index(row)
            if resolve_image(row.get('img_name', '')) is not None:
                kept.append(row)
        except ProtocolError:
            continue
    kept.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest())
    return kept[:max_samples] if max_samples else kept


def sample_token_bank(token_lists, max_tokens: int, seed: int):
    if not token_lists:
        return None
    tokens = np.concatenate(token_lists, axis=0).astype(np.float32)
    if len(tokens) > max_tokens:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tokens), size=max_tokens, replace=False)
        tokens = tokens[idx]
    return unit(tokens)


def token_bank_risk(tokens: np.ndarray, bank_unit: np.ndarray | None):
    if bank_unit is None or len(bank_unit) == 0:
        return None
    sims = unit(tokens.astype(np.float32)) @ bank_unit.T
    max_sims = sims.max(axis=1)
    return {
        'one_minus_native_token_mean_sim': float(1.0 - max_sims.mean()),
        'one_minus_native_token_p10_sim': float(1.0 - np.quantile(max_sims, 0.10)),
        'one_minus_native_token_p50_sim': float(1.0 - np.quantile(max_sims, 0.50)),
    }


def best_threshold_router(rows, key, train_qids):
    values = sorted(set(float(r['risk_scores'][key]) for r in rows if r['qid'] in train_qids and r['risk_scores'].get(key) is not None))
    if not values:
        return None
    candidates = [min(values) - 1e-9] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [max(values) + 1e-9]
    best = None
    for thr in candidates:
        train = [r for r in rows if r['qid'] in train_qids]
        correct = [r['mitigation_correct'] if float(r['risk_scores'][key]) >= thr else r['greedy_correct'] for r in train]
        value = acc(correct)
        routed = sum(float(r['risk_scores'][key]) >= thr for r in train)
        rescue_auc = auroc([r['risk_scores'][key] for r in train], [r['rescue'] for r in train]) or -1.0
        item = (value, -routed, rescue_auc, float(thr))
        if best is None or item > best:
            best = item
    return best[3]


def evaluate_risks(rows, score_keys, train_qids, qids):
    metrics = {}
    selected = None
    for key in score_keys:
        valid = [r for r in rows if r['risk_scores'].get(key) is not None]
        scores = [float(r['risk_scores'][key]) for r in valid]
        item = {
            'auroc_greedy_error': auroc(scores, [not r['greedy_correct'] for r in valid]),
            'auroc_mitigation_rescue': auroc(scores, [r['rescue'] for r in valid]),
            'auroc_mitigation_harmful': auroc(scores, [r['harmful'] for r in valid]),
        }
        thr = best_threshold_router(valid, key, train_qids)
        if thr is not None:
            for split_name, split_qids in [('train', train_qids), ('test', set(qids) - train_qids), ('all', set(qids))]:
                split = [r for r in valid if r['qid'] in split_qids]
                correct = [r['mitigation_correct'] if float(r['risk_scores'][key]) >= thr else r['greedy_correct'] for r in split]
                routed = [r for r in split if float(r['risk_scores'][key]) >= thr]
                item[f'router_{split_name}'] = {
                    'threshold': float(thr),
                    'n': len(split),
                    'routed': len(routed),
                    'coverage_mitigation': float(len(routed) / len(split)) if split else None,
                    'accuracy': acc(correct),
                    'rescues_kept': sum(r['rescue'] and float(r['risk_scores'][key]) >= thr for r in split),
                    'harmful_introduced': sum(r['harmful'] and float(r['risk_scores'][key]) >= thr for r in split),
                }
            train = item['router_train']
            objective = (float(train['accuracy']), -int(train['routed']), float(item['auroc_mitigation_rescue'] or -1.0))
            candidate = {'score_key': key, 'objective': list(objective), 'train': train, 'test': item['router_test'], 'all': item['router_all']}
            if selected is None or tuple(candidate['objective']) > tuple(selected['objective']):
                selected = candidate
        metrics[key] = item
    return metrics, selected


def main():
    args = parse_args()
    rows_raw = json.loads(args.dataset.read_text())
    greedy, greedy_meta = load_eval(args.greedy_eval)
    mitigation, mitigation_meta = load_eval(args.mitigation_eval)
    selected_rows = eligible_rows(rows_raw, args.question_type, args.seed, args.max_samples)
    qids = [str(r['qid']) for r in selected_rows]
    train_cut = int(round(args.train_frac * len(qids)))
    train_qids = set(qids[:train_cut])

    adapter = LlavaLocalSourceAdapter()
    records = []
    for row in tqdm(selected_rows, desc='competence native features'):
        qid = str(row['qid'])
        if qid not in greedy or qid not in mitigation:
            continue
        path = resolve_image(row.get('img_name', ''))
        if path is None:
            continue
        with Image.open(path) as raw:
            image = resize_image(raw.convert('RGB'), args.max_image_side)
        tokens = adapter.visual_tokens([image])[0].astype(np.float32)
        pooled = tokens.mean(axis=0).astype(np.float32)
        g = greedy[qid]
        m = mitigation[qid]
        records.append({
            'qid': qid,
            'img_name': row.get('img_name'),
            'answer': row.get('answer'),
            'greedy_correct': bool(g['correct']),
            'mitigation_correct': bool(m['correct']),
            'rescue': (not g['correct']) and bool(m['correct']),
            'harmful': bool(g['correct']) and (not m['correct']),
            'both_wrong': (not g['correct']) and (not m['correct']),
            'pooled': pooled,
            'tokens': tokens,
        })

    train = [r for r in records if r['qid'] in train_qids]
    train_correct = [r for r in train if r['greedy_correct']]
    train_wrong = [r for r in train if not r['greedy_correct']]
    if len(train_correct) < 2:
        raise RuntimeError('not enough correct calibration samples for native center')
    correct_center = np.mean([r['pooled'] for r in train_correct], axis=0)
    all_center = np.mean([r['pooled'] for r in train], axis=0)
    wrong_center = np.mean([r['pooled'] for r in train_wrong], axis=0) if train_wrong else None
    correct_bank = sample_token_bank([r['tokens'] for r in train_correct], args.max_token_bank, args.seed)
    all_bank = sample_token_bank([r['tokens'] for r in train], args.max_token_bank, args.seed + 1)

    rng = np.random.default_rng(args.seed)
    random_train = list(train)
    rng.shuffle(random_train)
    random_subset = random_train[:len(train_correct)]
    random_center = np.mean([r['pooled'] for r in random_subset], axis=0)
    random_bank = sample_token_bank([r['tokens'] for r in random_subset], args.max_token_bank, args.seed + 2)

    output_rows = []
    for r in records:
        risk = {
            'native_correct_distance': cosine_distance(r['pooled'], correct_center),
            'native_all_distance': cosine_distance(r['pooled'], all_center),
            'random_center_distance': cosine_distance(r['pooled'], random_center),
        }
        if wrong_center is not None:
            d_correct = risk['native_correct_distance']
            d_wrong = cosine_distance(r['pooled'], wrong_center)
            risk['native_correct_minus_wrong_distance'] = float(d_correct - d_wrong)
        risk.update(token_bank_risk(r['tokens'], correct_bank) or {})
        all_token = token_bank_risk(r['tokens'], all_bank) or {}
        risk.update({k.replace('native_', 'all_'): v for k, v in all_token.items()})
        rand_token = token_bank_risk(r['tokens'], random_bank) or {}
        risk.update({k.replace('native_', 'random_'): v for k, v in rand_token.items()})
        out = {k: v for k, v in r.items() if k not in {'pooled', 'tokens'}}
        out['split'] = 'train' if r['qid'] in train_qids else 'test'
        out['risk_scores'] = risk
        output_rows.append(out)

    score_keys = sorted(output_rows[0]['risk_scores']) if output_rows else []
    metrics, selected = evaluate_risks(output_rows, score_keys, train_qids, qids)
    summary = {
        'version': VERSION,
        'dataset': str(args.dataset.resolve()),
        'dataset_sha256': file_sha256(args.dataset),
        'n': len(output_rows),
        'train_n': len(train),
        'test_n': len(output_rows) - len(train),
        'train_correct_n': len(train_correct),
        'train_wrong_n': len(train_wrong),
        'greedy_accuracy': acc([r['greedy_correct'] for r in output_rows]),
        'mitigation_accuracy': acc([r['mitigation_correct'] for r in output_rows]),
        'rescues': sum(r['rescue'] for r in output_rows),
        'harmful': sum(r['harmful'] for r in output_rows),
        'max_token_bank': args.max_token_bank,
        'train_frac': args.train_frac,
    }
    payload = {'summary': summary, 'selected_by_calibration': selected, 'risk_metrics': metrics, 'rows': output_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({'summary': summary, 'selected_by_calibration': selected}, indent=2))


if __name__ == '__main__':
    main()
