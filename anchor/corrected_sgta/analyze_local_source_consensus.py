"""Analyze Source-Local Consensus cache without semantic-prototype transduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.methods import softmax_np

VERSION = 'sgta-local-source-consensus-analysis-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--laplacian-lambda-grid', type=float, nargs='+', default=(0.1, 0.3, 1.0, 3.0))
    return parser.parse_args()


def acc(preds: list[int], gt: list[int]) -> dict:
    correct = sum(int(p == y) for p, y in zip(preds, gt))
    return {'n': len(gt), 'correct': correct, 'accuracy': correct / len(gt) if gt else None}


def flips(base: list[int], other: list[int], gt: list[int]) -> dict:
    rescues = harmful = changed = 0
    for b, o, y in zip(base, other, gt):
        if b != o:
            changed += 1
            if b != y and o == y:
                rescues += 1
            if b == y and o != y:
                harmful += 1
    return {'rescues': rescues, 'harmful': harmful, 'net': rescues - harmful, 'changed': changed}


def main() -> None:
    args = parse_args()
    meta = json.loads(args.cache.with_suffix(args.cache.suffix + '.meta.json').read_text())
    records = list(iter_successes(args.cache, meta['fingerprint']))
    if not records:
        raise RuntimeError('no successful records')
    gt = [int(r['gt_index']) for r in records]
    logits = np.asarray([r['style_logits'] for r in records], dtype=np.float32)
    original = logits[:, 0, :]
    matched = logits[:, 1, :]
    wrong = logits[:, 2, :]
    pred_original = original.argmax(axis=1).astype(int).tolist()
    pred_matched = matched.argmax(axis=1).astype(int).tolist()
    pred_wrong = wrong.argmax(axis=1).astype(int).tolist()
    reports = {
        'original_surface': acc(pred_original, gt),
        'matched_local_surface': acc(pred_matched, gt),
        'wrong_control_surface': acc(pred_wrong, gt),
    }
    reports['matched_flips_vs_original'] = flips(pred_original, pred_matched, gt)
    reports['wrong_flips_vs_original'] = flips(pred_original, pred_wrong, gt)
    lap = {}
    probs = softmax_np(logits)
    for lam in args.laplacian_lambda_grid:
        # Conservative complete-graph smoothing with the original view as anchor.
        # It is intentionally simple: source-local views can only help if their
        # average evidence overcomes the original prediction.
        fused = (probs[:, 0, :] + float(lam) * probs.mean(axis=1)) / (1.0 + float(lam))
        pred = fused.argmax(axis=1).astype(int).tolist()
        lap[f'lambda_{lam:g}'] = {**acc(pred, gt), 'flips_vs_original': flips(pred_original, pred, gt)}
    oracle = []
    for row, y in zip(logits, gt):
        oracle.append(any(int(view.argmax()) == y for view in row))
    rows = []
    for r, bo, bm, bw, y in zip(records, pred_original, pred_matched, pred_wrong, gt):
        rows.append({
            'qid': r['qid'],
            'gt_index': y,
            'original': bo,
            'matched': bm,
            'wrong': bw,
            'source_ids': r.get('style_source_ids'),
            'decoded': r.get('style_decoded_text'),
        })
    report = {
        'version': VERSION,
        'cache': str(args.cache),
        'fingerprint': meta['fingerprint'],
        'n': len(records),
        'config': meta.get('config'),
        'point': reports,
        'laplacian': lap,
        'style_oracle': {'accuracy': float(np.mean(oracle)), 'headroom': float(np.mean(oracle) - reports['original_surface']['accuracy'])},
        'gate': {
            'oracle_headroom_ge_2pp': float(np.mean(oracle) - reports['original_surface']['accuracy']) >= 0.02,
            'matched_rescues_ge_harmful': reports['matched_flips_vs_original']['rescues'] >= reports['matched_flips_vs_original']['harmful'],
            'matched_beats_wrong_control': reports['matched_local_surface']['accuracy'] > reports['wrong_control_surface']['accuracy'],
            'matched_noninferior_0_5pp': reports['matched_local_surface']['accuracy'] >= reports['original_surface']['accuracy'] - 0.005,
        },
        'bad_case_candidates': [row for row in rows if row['original'] == row['gt_index'] and row['matched'] != row['gt_index']][:20],
        'rescue_candidates': [row for row in rows if row['original'] != row['gt_index'] and row['matched'] == row['gt_index']][:20],
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'point': report['point'], 'style_oracle': report['style_oracle'], 'gate': report['gate']}, indent=2))


if __name__ == '__main__':
    main()
