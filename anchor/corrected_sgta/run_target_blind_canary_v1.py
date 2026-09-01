#!/usr/bin/env python3
"""Target-blind, resumable native-generation runner for small medical VLM canaries.

The input manifest must contain no target/answer fields.  The runner deliberately
emits no reference answer, label, or copied manifest metadata.  GPU locking is an
external orchestration responsibility; this module never creates or acquires a
lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageFile, UnidentifiedImageError

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.run_native_oe_vqa import stable_seed


VERSION = "target-blind-canary-v1"
BASE_SEED = 42
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
}
FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "answer",
        "answers",
        "correct_answer",
        "expected_answer",
        "gold",
        "ground_truth",
        "gt_ans",
        "label",
        "labels",
        "reference",
        "target",
        "targets",
        "truth",
    }
)
IMAGE_KEYS = ("img_name", "image_path", "image", "dicom_relpath")
SUPPORTED_RASTER_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
SUPPORTED_DICOM_SUFFIXES = frozenset({".dcm", ".dicom"})

# Match the canonical OE runner's handling of recoverable truncated JPEGs.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}"
            if key_text.strip().lower() in FORBIDDEN_TARGET_KEYS:
                paths.append(child_path)
            paths.extend(_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def item_qid(row: Mapping[str, Any]) -> str:
    for key in ("qid", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("each manifest row requires a non-empty qid/question_id/id")


def image_reference(row: Mapping[str, Any]) -> str:
    values = [str(row[key]) for key in IMAGE_KEYS if row.get(key) is not None and str(row[key])]
    if not values:
        raise ValueError(f"manifest row {item_qid(row)!r} has no image reference")
    if len(set(values)) != 1:
        raise ValueError(f"manifest row {item_qid(row)!r} has conflicting image references")
    return values[0]


def load_target_blind_manifest(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden = _forbidden_paths(payload)
    if forbidden:
        preview = ", ".join(forbidden[:5])
        raise ValueError(f"target-bearing field(s) forbidden in canary manifest: {preview}")
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must be a JSON list or an object with a records list")
    selected = rows[:limit] if limit else rows
    if not selected:
        raise ValueError("target-blind canary manifest is empty")
    qids: list[str] = []
    for index, row in enumerate(selected):
        try:
            qids.append(item_qid(row))
            question = row.get("question")
            if question is None or not str(question).strip():
                raise ValueError("question is empty")
            image_reference(row)
        except ValueError as exc:
            raise ValueError(f"invalid manifest row {index}: {exc}") from exc
    if len(qids) != len(set(qids)):
        raise ValueError("manifest qids are not unique")
    return selected


def resolve_image_path(row: Mapping[str, Any], image_root: Path) -> Path:
    reference = Path(image_reference(row))
    return reference if reference.is_absolute() else image_root / reference


def load_input_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_DICOM_SUFFIXES:
        # This is the existing VinDr renderer used by the commitment probes:
        # modality LUT, robust 0.5/99.5 percentile window, and MONOCHROME1 fix.
        from anchor.corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil

        return dicom_to_pil(path)
    if suffix not in SUPPORTED_RASTER_SUFFIXES:
        raise ValueError(f"unsupported medical image suffix {suffix!r}: {path}")
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"unreadable raster image {path}: {type(exc).__name__}: {exc}") from exc


def preflight_inputs(rows: list[dict[str, Any]], image_root: Path) -> dict[str, Any]:
    counts = {"raster": 0, "dicom": 0}
    for row in rows:
        path = resolve_image_path(row, image_root)
        if not path.is_file():
            raise FileNotFoundError(f"missing input for {item_qid(row)!r}: {path}")
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_DICOM_SUFFIXES:
            counts["dicom"] += 1
        elif suffix in SUPPORTED_RASTER_SUFFIXES:
            counts["raster"] += 1
        else:
            raise ValueError(f"unsupported medical image suffix {suffix!r}: {path}")
        image = load_input_image(path)
        try:
            if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
                raise ValueError(f"renderer returned invalid image for {path}")
        finally:
            image.close()
    return {"n": len(rows), "image_types": counts, "target_fields_present": 0}


def build_config(
    *,
    model: str,
    manifest: Path,
    image_root: Path,
    limit: int,
    max_new_tokens: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    package_root = runner.parents[1]
    return {
        "protocol": VERSION,
        "model": model,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "image_root": str(image_root.resolve()),
        "limit": limit,
        "row_count": len(rows),
        "ordered_qids_sha256": sha256_json([item_qid(row) for row in rows]),
        "max_new_tokens": max_new_tokens,
        "base_seed": BASE_SEED,
        "prompt": "exact source question; model-native image placeholder and chat template",
        "generation": GENERATION,
        "target_blind": True,
        "output_fields": ["question_id", "text", "model_id", "metadata"],
        "gpu_lock": "external",
        "runner_sha256": sha256_file(runner),
        "adapter_code_sha256": sha256_file(package_root / "corrected_sgta" / "models_oe.py"),
        "dicom_renderer_sha256": sha256_file(
            package_root / "corrected_sgta" / "run_huatuo_vindr_commitment_probe.py"
        ),
    }


def freeze_config(output_dir: Path, config: dict[str, Any], resume: bool) -> str:
    fingerprint = sha256_json(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "generation_config.json"
    answers_path = output_dir / "answers.jsonl"
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError("--resume requires generation_config.json")
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("refusing to resume an incompatible target-blind canary")
    else:
        if config_path.exists() or answers_path.exists():
            raise FileExistsError("canary output exists; use --resume after verifying identity")
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({**config, "fingerprint": fingerprint}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(config_path)
    return fingerprint


def load_strict_resume(
    path: Path,
    expected_qids: list[str],
    fingerprint: str,
    model: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"answer line {line_number} is not an object")
        forbidden = _forbidden_paths(row)
        if forbidden:
            raise ValueError(f"target field leaked into answer line {line_number}: {forbidden[0]}")
        if row.get("model_id") != model:
            raise ValueError(f"model drift in answer line {line_number}")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"fingerprint drift in answer line {line_number}")
        rows.append(row)
    observed = [item_qid(row) for row in rows]
    if observed != expected_qids[: len(observed)] or len(observed) != len(set(observed)):
        raise ValueError("existing answers are not an exact unique manifest prefix")
    return rows


def build_output_record(
    *,
    qid: str,
    model: str,
    result: Any,
    max_new_tokens: int,
    sample_seed: int,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "question_id": qid,
        "text": result.text,
        "model_id": model,
        "metadata": {
            "generated_token_count": result.token_count,
            "generated_token_ids": list(result.token_ids),
            "hit_max_new_tokens": result.token_count >= max_new_tokens,
            "stop_reason": "length" if result.token_count >= max_new_tokens else "eos_or_template",
            "empty_generation": not bool(result.text.strip()),
            "mean_token_nll": result.uncertainty,
            "base_seed": BASE_SEED,
            "sample_seed": sample_seed,
            "fingerprint": fingerprint,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("huatuo", "hulu", "llava", "qwen"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    rows = load_target_blind_manifest(args.manifest, args.limit)
    preflight = preflight_inputs(rows, args.image_root)
    config = build_config(
        model=args.model,
        manifest=args.manifest,
        image_root=args.image_root,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        rows=rows,
    )
    fingerprint = sha256_json(config)
    if args.preflight_only:
        print(json.dumps({**preflight, "fingerprint": fingerprint}, indent=2), flush=True)
        return

    fingerprint = freeze_config(args.output_dir, config, args.resume)
    expected = [item_qid(row) for row in rows]
    answers_path = args.output_dir / "answers.jsonl"
    completed = load_strict_resume(answers_path, expected, fingerprint, args.model)

    from anchor.corrected_sgta.models_oe import load_oe_adapter

    adapter = load_oe_adapter(args.model, llava_conv_mode="mistral_instruct")
    try:
        with answers_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                current_qid = expected[index]
                sample_seed = stable_seed(BASE_SEED, current_qid)
                image = load_input_image(resolve_image_path(row, args.image_root))
                try:
                    result = adapter.generate_control(
                        image=image,
                        prompt=str(row["question"]),
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        seed=sample_seed,
                    )
                finally:
                    image.close()
                record = build_output_record(
                    qid=current_qid,
                    model=args.model,
                    result=result,
                    max_new_tokens=args.max_new_tokens,
                    sample_seed=sample_seed,
                    fingerprint=fingerprint,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index + 1}/{len(rows)}] {current_qid}", flush=True)
    finally:
        adapter.close()

    final = load_strict_resume(answers_path, expected, fingerprint, args.model)
    if len(final) != len(rows):
        raise RuntimeError(f"target-blind canary incomplete: {len(final)}/{len(rows)}")


if __name__ == "__main__":
    main()
