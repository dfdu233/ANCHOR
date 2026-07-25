"""Create deterministic MedHEval JSON subsets by qid hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image-folder', type=Path, default=None)
    parser.add_argument('--max-samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--require-image', action='store_true')
    parser.add_argument('--question-type', default=None)
    return parser.parse_args()


def qid_of(row: dict) -> str:
    return str(row.get('qid', row.get('id', row.get('question_id'))))


def is_binary_row(row: dict) -> bool:
    qtype = str(row.get('question_type') or '').lower()
    if qtype == 'binary':
        return True
    choices = row.get('choices')
    answer = str(row.get('answer', row.get('gt_ans', ''))).strip().lower()
    return answer in {'yes', 'no'} and (choices is None or str(choices).strip() == '')


def normalized_question_type(value: object) -> str:
    return ''.join(character for character in str(value).lower() if character.isalnum())


def image_name(row: dict) -> str | None:
    value = row.get('img_name', row.get('image', row.get('img_id')))
    return None if value is None else str(value)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text())
    kept = []
    skipped_missing_image = 0
    skipped_question_type = 0
    for row in rows:
        if args.question_type:
            if args.question_type == 'binary':
                if not is_binary_row(row):
                    skipped_question_type += 1
                    continue
                if row.get('question_type') != 'binary':
                    row = dict(row)
                    row['question_type'] = 'binary'
            else:
                requested = normalized_question_type(args.question_type)
                observed = normalized_question_type(row.get('question_type', ''))
                aliases = {'multiplechoice': 'multichoice'}
                requested = aliases.get(requested, requested)
                observed = aliases.get(observed, observed)
                if observed != requested:
                    skipped_question_type += 1
                    continue
        if args.require_image:
            if args.image_folder is None:
                raise ValueError('--require-image needs --image-folder')
            name = image_name(row)
            if not name or not (args.image_folder / name).exists():
                skipped_missing_image += 1
                continue
        kept.append(row)
    kept.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{qid_of(row)}".encode()).hexdigest())
    selected = kept[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2))
    meta = {
        'input': str(args.input),
        'output': str(args.output),
        'seed': args.seed,
        'max_samples': args.max_samples,
        'selected': len(selected),
        'eligible': len(kept),
        'skipped_missing_image': skipped_missing_image,
        'skipped_question_type': skipped_question_type,
        'question_type': args.question_type,
        'order': 'sha256(seed:qid)',
        'qids': [qid_of(row) for row in selected],
    }
    args.output.with_suffix(args.output.suffix + '.meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
