"""Score Source Bank domain-risk for MedHEval rows without generation."""

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
from corrected_sgta.source_bank_v2 import entries_for_modality, load_manifest, sha256_file

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = 'source-domain-risk-v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-bank', required=True, type=Path)
    parser.add_argument('--local-prototypes', required=True, type=Path)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-image-side', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--question-type', default='binary')
    return parser.parse_args()


def unit(x: np.ndarray, axis=-1, eps=1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
    bu = unit(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
    return float(1.0 - np.dot(au, bu))


def softmax_neg_distance(distances: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    logits = -np.asarray(distances, dtype=np.float64) / max(temperature, 1e-6)
    logits -= logits.max()
    e = np.exp(logits)
    return (e / e.sum()).astype(np.float64)


def entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(len(p)))


def load_prototypes(path: Path, source_hash: str) -> tuple[dict, dict[str, np.ndarray]]:
    meta = json.loads(path.with_suffix(path.suffix + '.meta.json').read_text())
    if meta.get('version') != 'sgta-local-source-prototypes-v1':
        raise RuntimeError('unsupported prototype version')
    if meta.get('source_bank_sha256') != source_hash:
        raise RuntimeError('local prototype/source-bank mismatch')
    payload = np.load(path, allow_pickle=False)
    out = {}
    for entry in meta.get('entries', []):
        out[entry['source_id']] = payload[entry['array_key']].astype(np.float32)
    return meta, out


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


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    source_hash = sha256_file(args.source_bank)
    manifest = load_manifest(args.source_bank)
    proto_meta, prototypes = load_prototypes(args.local_prototypes, source_hash)
    entries = [e for e in entries_for_modality(manifest, 'xray', formal_only=True) if e['source_id'] in prototypes]
    source_ids = [e['source_id'] for e in entries]
    source_centers = {sid: prototypes[sid].mean(axis=0).astype(np.float32) for sid in source_ids}

    selected = eligible_rows(rows, args.question_type, args.seed, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'version': VERSION,
        'dataset': str(args.dataset.resolve()),
        'dataset_sha256': file_sha256(args.dataset),
        'source_bank': str(args.source_bank.resolve()),
        'source_bank_sha256': source_hash,
        'local_prototypes': str(args.local_prototypes.resolve()),
        'local_prototypes_sha256': sha256_file(args.local_prototypes),
        'source_ids': source_ids,
        'max_samples': args.max_samples,
        'selected': len(selected),
        'max_image_side': args.max_image_side,
        'seed': args.seed,
        'question_type': args.question_type,
    }
    args.output.with_suffix(args.output.suffix + '.meta.json').write_text(json.dumps(meta, indent=2))

    adapter = LlavaLocalSourceAdapter()
    with args.output.open('w') as handle:
        for row in tqdm(selected, desc='source domain risk'):
            path = resolve_image(row.get('img_name', ''))
            if path is None:
                continue
            try:
                with Image.open(path) as raw:
                    image = resize_image(raw.convert('RGB'), args.max_image_side)
                tokens = adapter.visual_tokens([image])[0].astype(np.float32)
                pooled = tokens.mean(axis=0)
                center_distances = {sid: cosine_distance(pooled, source_centers[sid]) for sid in source_ids}
                ordered = sorted(center_distances.items(), key=lambda kv: kv[1])
                dvals = np.array([center_distances[sid] for sid in source_ids], dtype=np.float64)
                probs = softmax_neg_distance(dvals)
                token_scores = {}
                token_unit = unit(tokens)
                for sid in source_ids:
                    proto_unit = unit(prototypes[sid])
                    sims = token_unit @ proto_unit.T
                    max_sims = sims.max(axis=1)
                    token_scores[sid] = {
                        'mean_max_sim': float(max_sims.mean()),
                        'p10_max_sim': float(np.quantile(max_sims, 0.10)),
                        'p50_max_sim': float(np.quantile(max_sims, 0.50)),
                    }
                row_out = {
                    'qid': row['qid'],
                    'img_name': row.get('img_name'),
                    'gt_index': ground_truth_index(row),
                    'answer': row.get('answer'),
                    'source_ids': source_ids,
                    'center_distances': center_distances,
                    'nearest_source': ordered[0][0],
                    'nearest_distance': ordered[0][1],
                    'second_distance': ordered[1][1] if len(ordered) > 1 else None,
                    'distance_margin': (ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else None,
                    'source_entropy': entropy(probs),
                    'source_probabilities': {sid: float(prob) for sid, prob in zip(source_ids, probs)},
                    'token_source_scores': token_scores,
                    'risk_scores': {
                        'nearest_distance': ordered[0][1],
                        'negative_margin': -((ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else 0.0),
                        'source_entropy': entropy(probs),
                        'one_minus_best_source_probability': float(1.0 - probs.max()),
                        'one_minus_nearest_token_mean_sim': float(1.0 - token_scores[ordered[0][0]]['mean_max_sim']),
                        'one_minus_nearest_token_p10_sim': float(1.0 - token_scores[ordered[0][0]]['p10_max_sim']),
                    },
                }
                handle.write(json.dumps(row_out) + '\n')
                handle.flush()
            except Exception as exc:
                handle.write(json.dumps({'qid': row.get('qid'), 'status': 'error', 'error': repr(exc)}) + '\n')
                handle.flush()


if __name__ == '__main__':
    main()
