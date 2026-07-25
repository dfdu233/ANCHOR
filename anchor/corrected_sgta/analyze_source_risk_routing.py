"""Analyze whether Source Bank domain-risk predicts mitigation gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk-jsonl', required=True, type=Path)
    parser.add_argument('--greedy-eval', required=True, type=Path)
    parser.add_argument('--mitigation-eval', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mitigation-name', default='PAI')
    return parser.parse_args()


def load_eval(path: Path):
    payload = json.loads(path.read_text())
    return {str(d['question_id']): d for d in payload['details']}, payload


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U / pairwise probability, with tie=0.5.
    wins = 0.0
    for value in pos:
        wins += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def acc(correct):
    return float(np.mean(correct)) if len(correct) else None


def split_train_test(qids):
    # The subset order is qid-hash selected; preserve order for a simple 40/60 validation split.
    n = len(qids)
    cut = int(round(0.4 * n))
    return set(qids[:cut]), set(qids[cut:])


def best_threshold_router(rows, score_key, train_qids):
    values = sorted(set(float(r['risk_scores'][score_key]) for r in rows if r['qid'] in train_qids))
    if not values:
        return None
    candidates = [min(values) - 1e-9] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [max(values) + 1e-9]
    best = None
    for thr in candidates:
        train = [r for r in rows if r['qid'] in train_qids]
        correct = [r['mitigation_correct'] if r['risk_scores'][score_key] >= thr else r['greedy_correct'] for r in train]
        value = acc(correct)
        # tie-breaker: prefer fewer routed samples, i.e. simpler/more conservative.
        routed = sum(r['risk_scores'][score_key] >= thr for r in train)
        item = (value, -routed, float(thr))
        if best is None or item > best:
            best = item
    return best[2]


def main():
    args = parse_args()
    greedy, greedy_meta = load_eval(args.greedy_eval)
    mit, mit_meta = load_eval(args.mitigation_eval)
    rows = []
    for line in args.risk_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = str(r.get('qid'))
        if r.get('status') == 'error' or qid not in greedy or qid not in mit:
            continue
        g = greedy[qid]
        m = mit[qid]
        r['qid'] = qid
        r['greedy_correct'] = bool(g['correct'])
        r['mitigation_correct'] = bool(m['correct'])
        r['rescue'] = (not g['correct']) and bool(m['correct'])
        r['harmful'] = bool(g['correct']) and (not m['correct'])
        r['both_wrong'] = (not g['correct']) and (not m['correct'])
        rows.append(r)
    qids = [r['qid'] for r in rows]
    train_qids, test_qids = split_train_test(qids)
    score_keys = sorted(rows[0]['risk_scores']) if rows else []
    metrics = {}
    for key in score_keys:
        scores = [float(r['risk_scores'][key]) for r in rows]
        metrics[key] = {
            'auroc_greedy_error': auroc(scores, [not r['greedy_correct'] for r in rows]),
            'auroc_mitigation_rescue': auroc(scores, [r['rescue'] for r in rows]),
            'auroc_mitigation_harmful': auroc(scores, [r['harmful'] for r in rows]),
        }
        thr = best_threshold_router(rows, key, train_qids)
        if thr is not None:
            for split_name, split_qids in [('train', train_qids), ('test', test_qids), ('all', set(qids))]:
                split = [r for r in rows if r['qid'] in split_qids]
                routed = [r for r in split if float(r['risk_scores'][key]) >= thr]
                correct = [r['mitigation_correct'] if float(r['risk_scores'][key]) >= thr else r['greedy_correct'] for r in split]
                metrics[key][f'router_{split_name}'] = {
                    'threshold': float(thr),
                    'n': len(split),
                    'routed': len(routed),
                    'coverage_mitigation': float(len(routed) / len(split)) if split else None,
                    'accuracy': acc(correct),
                    'rescues_kept': sum(r['rescue'] and float(r['risk_scores'][key]) >= thr for r in split),
                    'harmful_introduced': sum(r['harmful'] and float(r['risk_scores'][key]) >= thr for r in split),
                }
    selected = None
    for key, item in metrics.items():
        train = item.get('router_train')
        test = item.get('router_test')
        if not train or not test:
            continue
        # Calibration objective: maximize train accuracy; tie-break by fewer routed samples,
        # then by higher rescue AUROC. This yields a compact plug-in router.
        objective = (
            float(train['accuracy']),
            -int(train['routed']),
            float(item.get('auroc_mitigation_rescue') or -1.0),
        )
        candidate = {'score_key': key, 'objective': objective, 'train': train, 'test': test, 'all': item.get('router_all')}
        if selected is None or objective > selected['objective']:
            selected = candidate
    if selected is not None:
        selected['objective'] = list(selected['objective'])

    summary = {
        'n': len(rows),
        'mitigation_name': args.mitigation_name,
        'greedy_correct': sum(r['greedy_correct'] for r in rows),
        'mitigation_correct': sum(r['mitigation_correct'] for r in rows),
        'greedy_accuracy': acc([r['greedy_correct'] for r in rows]),
        'mitigation_accuracy': acc([r['mitigation_correct'] for r in rows]),
        'rescues': sum(r['rescue'] for r in rows),
        'harmful': sum(r['harmful'] for r in rows),
        'both_wrong': sum(r['both_wrong'] for r in rows),
        'train_n': len(train_qids),
        'test_n': len(test_qids),
    }
    payload = {'summary': summary, 'selected_by_calibration': selected, 'risk_metrics': metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
