"""Build token-level local prototypes from audited Source Bank entries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.source_bank_v2 import entries_for_modality, load_index, load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import load_descriptor_image, verify_source_artifacts

VERSION = 'sgta-local-source-prototypes-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('llava',), default='llava')
    parser.add_argument('--source-bank', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--modality', default='xray')
    parser.add_argument('--max-images-per-source', type=int, default=64)
    parser.add_argument('--max-tokens-per-source', type=int, default=20000)
    parser.add_argument('--prototypes-per-source', type=int, default=64)
    parser.add_argument('--kmeans-iters', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def ordered(items: list[dict], seed: int, source_id: str) -> list[dict]:
    return sorted(items, key=lambda row: hashlib.sha256(f'{seed}:{source_id}:{row}'.encode()).hexdigest())


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6, None)


def kmeans(tokens: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = normalize(tokens.astype(np.float32))
    if len(x) <= k:
        return x.copy()
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        sims = x @ centers.T
        labels = sims.argmax(axis=1)
        new = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                new[idx] = x[mask].mean(axis=0)
        centers = normalize(new)
    return centers.astype(np.float32)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.source_bank)
    source_hash = sha256_file(args.source_bank)
    verified = verify_source_artifacts(manifest)
    entries = entries_for_modality(manifest, args.modality, formal_only=True)
    if not entries:
        raise RuntimeError(f'no formal entries for modality {args.modality}')
    adapter = LlavaLocalSourceAdapter()
    arrays = {}
    meta_entries = []
    try:
        for entry in entries:
            source_id = entry['source_id']
            rows = ordered(load_index(Path(entry['image_index'])), args.seed, source_id)
            rows = rows[: args.max_images_per_source]
            token_chunks = []
            for row in tqdm(rows, desc=f'local prototypes {source_id}'):
                image = load_descriptor_image(row)
                tokens = adapter.visual_tokens([image])[0]
                token_chunks.append(tokens)
                image.close()
            tokens = np.concatenate(token_chunks, axis=0).astype(np.float32)
            if len(tokens) > args.max_tokens_per_source:
                rng = np.random.default_rng(args.seed)
                pick = rng.choice(len(tokens), size=args.max_tokens_per_source, replace=False)
                tokens = tokens[pick]
            proto = kmeans(tokens, args.prototypes_per_source, args.kmeans_iters, args.seed)
            key = f'proto_{len(arrays)}'
            arrays[key] = proto
            meta_entries.append({
                'source_id': source_id,
                'modality': entry.get('modality'),
                'dataset': entry.get('dataset'),
                'formal': bool(entry.get('formal')),
                'n_images': len(rows),
                'n_tokens_sampled': int(len(tokens)),
                'array_key': key,
                'prototype_shape': list(proto.shape),
            })
    finally:
        adapter.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        'version': VERSION,
        'model': args.model,
        'source_bank': str(args.source_bank.resolve()),
        'source_bank_sha256': source_hash,
        'verified_source_artifacts': verified,
        'seed': args.seed,
        'modality': args.modality,
        'max_images_per_source': args.max_images_per_source,
        'max_tokens_per_source': args.max_tokens_per_source,
        'prototypes_per_source': args.prototypes_per_source,
        'kmeans_iters': args.kmeans_iters,
        'entries': meta_entries,
    }
    args.output.with_suffix(args.output.suffix + '.meta.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'output': str(args.output), 'entries': len(meta_entries)}, indent=2))


if __name__ == '__main__':
    main()
