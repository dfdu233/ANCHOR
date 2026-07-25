"""Run Source-Local Consensus smoke for fixed close-ended MedHEval rows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.cache import encode_array, load_successful_qids, repair_truncated_jsonl_tail
from corrected_sgta.infer_ce import decoded_label_index, resize_image
from corrected_sgta.models_local_source import LlavaLocalSourceAdapter
from corrected_sgta.protocol_v2 import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    ProtocolError,
    build_prompt,
    file_sha256,
    ground_truth_index,
    labels_for_sample,
    protocol_fingerprint,
    resolve_image,
    task_kind,
    validate_dataset,
)
from corrected_sgta.source_bank_v2 import cosine_distance, entries_for_modality, load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION = 'sgta-local-source-consensus-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('llava',), default='llava')
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-bank', required=True, type=Path)
    parser.add_argument('--local-prototypes', required=True, type=Path)
    parser.add_argument('--max-samples', type=int, default=32)
    parser.add_argument('--max-image-side', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--confidence-power', type=float, default=2.0)
    parser.add_argument('--decode-max-new-tokens', type=int, default=8)
    parser.add_argument('--question-type', default='binary')
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


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


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text())
    validation = validate_dataset(rows)
    manifest = load_manifest(args.source_bank)
    source_hash = sha256_file(args.source_bank)
    verified = verify_source_artifacts(manifest)
    proto_meta, prototypes = load_prototypes(args.local_prototypes, source_hash)
    entries = [entry for entry in entries_for_modality(manifest, 'xray', formal_only=True) if entry['source_id'] in prototypes]
    if len(entries) < 2:
        raise RuntimeError('need at least two formal xray source prototypes')
    config = {
        'cache_version': VERSION,
        'protocol_version': PROTOCOL_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'model': args.model,
        'dataset': str(args.dataset.resolve()),
        'dataset_sha256': file_sha256(args.dataset),
        'source_bank': str(args.source_bank.resolve()),
        'source_bank_sha256': source_hash,
        'verified_source_artifacts': verified,
        'local_prototypes': str(args.local_prototypes.resolve()),
        'local_prototypes_sha256': sha256_file(args.local_prototypes),
        'local_prototypes_meta_sha256': sha256_file(args.local_prototypes.with_suffix(args.local_prototypes.suffix + '.meta.json')),
        'max_samples': args.max_samples,
        'max_image_side': args.max_image_side,
        'seed': args.seed,
        'subset_order': 'sha256(seed:qid)',
        'source_selection': 'nearest same-modality local prototype pooled center',
        'wrong_control': 'other formal xray source with lexicographically smallest id',
        'beta': args.beta,
        'confidence_power': args.confidence_power,
        'decode_max_new_tokens': args.decode_max_new_tokens,
        'question_type': args.question_type,
    }
    fingerprint = protocol_fingerprint(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(args.output.suffix + '.meta.json')
    metadata = {
        'protocol_version': PROTOCOL_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'cache_version': VERSION,
        'fingerprint': fingerprint,
        'config': config,
        'dataset_validation': validation,
        'source_bank': manifest,
        'local_prototype_metadata': proto_meta,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get('fingerprint') != fingerprint:
            raise RuntimeError(f'metadata mismatch; choose a new output: {args.output}')
    else:
        atomic_json(meta_path, metadata)
    repair_truncated_jsonl_tail(args.output)
    saved = load_successful_qids(args.output, fingerprint)

    target_rows = []
    for row in rows:
        try:
            if task_kind(row) == 'open':
                continue
            if args.question_type and row.get('question_type') != args.question_type:
                continue
            labels = labels_for_sample(row)
            if args.question_type == 'binary' and labels != ('Yes', 'No') and labels != ['Yes', 'No']:
                # Existing protocol normalizes binary rows to Yes/No labels.
                pass
            ground_truth_index(row)
            if resolve_image(row.get('img_name', '')) is not None:
                target_rows.append(row)
        except ProtocolError:
            continue
    target_rows.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['qid']}".encode()).hexdigest())
    if args.max_samples:
        target_rows = target_rows[: args.max_samples]
    eligible = [row for row in target_rows if str(row['qid']) not in saved]
    print(f'local-source fingerprint={fingerprint[:12]} eligible={len(eligible)}', flush=True)
    if not eligible:
        return

    proto_centers = {sid: proto.mean(axis=0) for sid, proto in prototypes.items()}
    adapter = LlavaLocalSourceAdapter()
    errors = 0
    try:
        with args.output.open('a') as output:
            for sample in tqdm(eligible, desc='local source consensus'):
                try:
                    image_path = resolve_image(sample.get('img_name', ''))
                    with Image.open(image_path) as source:
                        image = resize_image(source, args.max_image_side)
                    labels = labels_for_sample(sample)
                    prompt = build_prompt(sample)
                    tokens = adapter.visual_tokens([image])[0]
                    pooled = tokens.mean(axis=0)
                    matched_entry = min(entries, key=lambda entry: cosine_distance(pooled, proto_centers[entry['source_id']]))
                    wrong_entry = min([entry for entry in entries if entry['source_id'] != matched_entry['source_id']], key=lambda entry: entry['source_id'])
                    matched_proto = prototypes[matched_entry['source_id']]
                    wrong_proto = prototypes[wrong_entry['source_id']]
                    original = adapter.forward_ce([image], prompt, labels)[0]
                    matched = adapter.forward_ce_local_transport(image, prompt, labels, matched_proto, args.beta, args.confidence_power)
                    wrong = adapter.forward_ce_local_transport(image, prompt, labels, wrong_proto, args.beta, args.confidence_power)
                    decoded = [
                        adapter.decode_ce([image], prompt, max_new_tokens=args.decode_max_new_tokens)[0],
                        adapter.decode_ce_local_transport(image, prompt, matched_proto, args.beta, args.decode_max_new_tokens, args.confidence_power),
                        adapter.decode_ce_local_transport(image, prompt, wrong_proto, args.beta, args.decode_max_new_tokens, args.confidence_power),
                    ]
                    evidence = [original, matched, wrong]
                    row = {
                        'protocol_version': PROTOCOL_VERSION,
                        'cache_version': VERSION,
                        'fingerprint': fingerprint,
                        'status': 'ok',
                        'qid': sample['qid'],
                        'img_name': sample.get('img_name', ''),
                        'question_type': task_kind(sample),
                        'labels': list(labels),
                        'gt_index': ground_truth_index(sample),
                        'style_names': ['original', f"matched_local_{matched_entry['source_id']}_b{args.beta:g}", f"wrong_local_{wrong_entry['source_id']}_b{args.beta:g}"],
                        'style_roles': ['original', 'matched_local', 'wrong_control'],
                        'style_source_ids': ['original', matched_entry['source_id'], wrong_entry['source_id']],
                        'style_logits': [item.logits.tolist() for item in evidence],
                        'style_probabilities': [softmax(item.logits).tolist() for item in evidence],
                        'style_sequence_nll': [item.sequence_nll.tolist() if item.sequence_nll is not None else None for item in evidence],
                        'style_language_features': encode_array(np.stack([item.features for item in evidence])),
                        'pooled_visual_feature': encode_array(pooled.astype(np.float32)),
                        'source_distances': {entry['source_id']: cosine_distance(pooled, proto_centers[entry['source_id']]) for entry in entries},
                        'style_decoded_text': decoded,
                        'style_decoded_prediction': [decoded_label_index(text, labels, sample) for text in decoded],
                    }
                except Exception as exc:
                    errors += 1
                    traceback.print_exc()
                    row = {
                        'protocol_version': PROTOCOL_VERSION,
                        'cache_version': VERSION,
                        'fingerprint': fingerprint,
                        'status': 'error',
                        'qid': sample.get('qid'),
                        'error': f'{type(exc).__name__}: {exc}'[:500],
                    }
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        gc.collect(); torch.cuda.empty_cache()
                output.write(json.dumps(row, separators=(',', ':')) + '\n')
                output.flush()
    finally:
        adapter.close()
    print(f'finished rows={len(eligible)} errors={errors}', flush=True)


if __name__ == '__main__':
    main()
