#!/usr/bin/env python3
"""Fast target-blind Yes/No/Maybe margin runner for cached CXR prompts.

The runner calls only ``HuatuoScorer.score`` or ``HuluScorer.score``. It never
calls generation, never loads a reference answer, and refuses recursively nested
target fields before touching a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


VERSION = "target-blind-tristate-margin-v1"
DEFAULT_INPUT = Path(
    "corrected_runs/matched_retrieval_polarity_pilot_v1/target_blind_pilot_v2.json"
)
DEFAULT_IMAGE_ROOT = Path("/home/dbw/ANCHOR/data/medheval/images")
DEFAULT_MODELS = {
    "huatuo": Path("/home/dbw/models/HuatuoGPT-Vision-7B"),
    "hulu": Path("/home/dbw/models/Hulu-Med-4B"),
}
DEFAULT_HUATUO_ROOT = Path("/home/dbw/HuatuoGPT-Vision")
EXPECTED_ARMS = {"present", "absent", "neutral", "random_deletion", "plain"}
INPUT_SUFFIX = "Answer with exactly one of: Yes, No, Uncertain."
SCORING_SUFFIX = "Answer with exactly one word: Yes, No, or Maybe."
FORBIDDEN_INPUT_KEYS = {
    "answer", "answers", "gtans", "gtanswer", "groundtruth", "label", "labels",
    "target", "targets", "reference", "referenceanswer", "referenceanswers",
    "gold", "goldanswer", "readerlabels", "readervotes", "positivevotes",
    "readersupport", "classid", "answeridx", "answerindex",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_key(key: object) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def reject_target_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = normalized_key(key)
            if normalized in FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"forbidden target field at {path}.{key}")
            if normalized == "selectionusestargetlabel" and child is not False:
                raise ValueError(f"target-blind audit flag must be false at {path}.{key}")
            reject_target_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_target_fields(child, f"{path}[{index}]")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_target_blind_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reject_target_fields(payload)
    if not isinstance(payload, list) or not payload:
        raise ValueError("input must be a nonempty JSON list")
    rows = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"row {index} is not an object")
        row = dict(raw)
        required = {"qid", "pair_id", "arm", "finding", "img_name", "question"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"row {index} missing fields: {missing}")
        if row.get("selection_uses_target_label") is not False:
            raise ValueError(f"row {index} lacks an explicit false target-use audit flag")
        if row["arm"] not in EXPECTED_ARMS:
            raise ValueError(f"row {index} has unexpected arm: {row['arm']!r}")
        if not str(row["question"]).endswith(INPUT_SUFFIX):
            raise ValueError(f"row {index} violates frozen input answer suffix")
        rows.append(row)
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("input qids are not unique")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, members in by_pair.items():
        if {str(row["arm"]) for row in members} != EXPECTED_ARMS or len(members) != 5:
            raise ValueError(f"pair {pair_id} does not contain exactly five frozen arms")
        for field in ("img_name", "finding"):
            if len({str(row[field]) for row in members}) != 1:
                raise ValueError(f"pair {pair_id} disagrees on {field}")
    return rows


def scoring_prompt(question: str) -> str:
    if question.count(INPUT_SUFFIX) != 1 or not question.endswith(INPUT_SUFFIX):
        raise ValueError("input prompt must end in exactly one frozen Uncertain suffix")
    return question[: -len(INPUT_SUFFIX)] + SCORING_SUFFIX


def resolve_cxr_image(image_root: Path, relative: str) -> Path:
    root = image_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"image escapes image root: {relative!r}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_cxr_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    return image


def image_inventory(rows: list[dict[str, Any]], image_root: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for relative in sorted({str(row["img_name"]) for row in rows}):
        path = resolve_cxr_image(image_root, relative)
        with Image.open(path) as source:
            width, height, mode, image_format = source.width, source.height, source.mode, source.format
        inventory[relative] = {
            "relative_path": relative,
            "resolved_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
            "mode": mode,
            "format": image_format,
        }
    return inventory


def cheap_model_inventory(model_dir: Path) -> dict[str, Any]:
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    weights = sorted(model_dir.glob("*.safetensors"))
    if missing or not weights:
        raise FileNotFoundError(f"incomplete model directory: missing={missing}, weights={len(weights)}")
    return {
        "path": str(model_dir.resolve()),
        "required_asset_sha256": {name: sha256_file(model_dir / name) for name in required},
        "weight_inventory": [
            {"name": path.name, "size_bytes": path.stat().st_size} for path in weights
        ],
    }


def preflight(rows: list[dict[str, Any]], input_path: Path, image_root: Path, model_dir: Path) -> dict[str, Any]:
    inventory = image_inventory(rows, image_root)
    prompts = [scoring_prompt(str(row["question"])) for row in rows]
    return {
        "status": "passed_target_blind_cpu_preflight",
        "version": VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "rows": len(rows),
        "pairs": len({str(row["pair_id"]) for row in rows}),
        "arms": sorted({str(row["arm"]) for row in rows}),
        "findings": sorted({str(row["finding"]) for row in rows}),
        "unique_images": len(inventory),
        "image_inventory_fingerprint": canonical_hash(inventory),
        "model_cheap_inventory": cheap_model_inventory(model_dir),
        "scoring_prompt_fingerprint": canonical_hash(prompts),
        "recursive_target_field_guard": "passed",
        "generation": "forbidden; score() only",
    }


def full_config(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    inspection: dict[str, Any],
) -> dict[str, Any]:
    # Import here so --self-test and --preflight-only never initialize a model.
    from corrected_sgta.run_cecd_factorial_v1 import (
        environment_fingerprint,
        full_model_artifact_fingerprint,
        python_source_tree_fingerprint,
    )

    model_artifact = full_model_artifact_fingerprint(args.model_dir)
    external_runtime = (
        python_source_tree_fingerprint(args.huatuo_root)
        if args.model_family == "huatuo" else None
    )
    source_files = {
        "runner": Path(__file__),
        "scorer": Path(__file__).with_name("run_cecd_factorial_v1.py"),
    }
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "model_family": args.model_family,
        "model_dir": str(args.model_dir.resolve()),
        "model_artifact": model_artifact,
        "external_huatuo_runtime": external_runtime,
        "device": args.device,
        "max_visual_tokens": args.max_visual_tokens,
        "input": inspection,
        "row_order_fingerprint": canonical_hash([str(row["qid"]) for row in rows]),
        "safe_row_fingerprint": canonical_hash(rows),
        "prompt_conversion": {"from": INPUT_SUFFIX, "to": SCORING_SUFFIX},
        "image_root": str(args.image_root.resolve()),
        "source_sha256": {name: sha256_file(path) for name, path in source_files.items()},
        "environment": environment_fingerprint(),
        "score_contract": (
            "HuatuoScorer/HuluScorer.score only; FP32 final-hidden @ Yes/No/Maybe lm-head; "
            "no generation and no reference-data access"
        ),
    }
    immutable = {key: value for key, value in candidate.items() if key not in {"created_at", "command"}}
    candidate["fingerprint"] = canonical_hash(immutable)
    return candidate


def freeze_or_resume(candidate: dict[str, Any], path: Path, resume: bool) -> dict[str, Any]:
    if not resume:
        if path.exists():
            raise FileExistsError(f"config already exists; use --resume: {path}")
        atomic_json(path, candidate)
        return candidate
    if not path.is_file():
        raise FileNotFoundError("--resume requires the frozen config.json")
    existing = json.loads(path.read_text(encoding="utf-8"))
    ignored = {"created_at", "command", "fingerprint"}
    left = {key: value for key, value in existing.items() if key not in ignored}
    right = {key: value for key, value in candidate.items() if key not in ignored}
    if left != right:
        changed = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"refusing resume after config drift: {changed}")
    expected = canonical_hash({key: value for key, value in existing.items() if key not in {"created_at", "command", "fingerprint"}})
    if existing.get("fingerprint") != expected:
        raise ValueError("stored config fingerprint is internally invalid")
    return existing


def public_scores(raw: Mapping[str, Any]) -> dict[str, Any]:
    internal = raw.get("logits")
    if not isinstance(internal, Mapping):
        raise ValueError("scorer returned no tristate logits")
    logits = {
        "Yes": float(internal["supported"]),
        "No": float(internal["refuted"]),
        "Maybe": float(internal["undetermined"]),
    }
    if not all(math.isfinite(value) for value in logits.values()):
        raise ValueError(f"non-finite FP32 logit: {logits}")
    polarity = logits["Yes"] - logits["No"]
    commitment = max(logits["Yes"], logits["No"]) - logits["Maybe"]
    if abs(polarity - float(raw["polarity"])) > 1e-5:
        raise ValueError("scorer polarity contract mismatch")
    if abs(commitment - float(raw["commitment"])) > 1e-5:
        raise ValueError("scorer commitment contract mismatch")
    return {
        "tristate_logits_fp32": logits,
        "polarity_yes_minus_no": polarity,
        "commitment_max_yes_no_minus_maybe": commitment,
        "argmax_state": max(logits, key=logits.get),
        "tristate_entropy_nats": float(raw["tristate_entropy"]),
        "wrapped_input_audit": raw.get("wrapped_input_audit"),
        "dtype_contract": "FP32 readout",
    }


def shard_name(index: int, qid: str) -> str:
    return f"{index:04d}-{hashlib.sha256(qid.encode()).hexdigest()[:16]}.json"


def valid_shard(path: Path, fingerprint: str, qid: str, row_hash: str, image_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        row.get("status") == "complete"
        and row.get("config_fingerprint") == fingerprint
        and row.get("qid") == qid
        and row.get("input_row_sha256") == row_hash
        and row.get("image_sha256") == image_hash
        and set(row.get("tristate_logits_fp32", {})) == {"Yes", "No", "Maybe"}
    )


def build_scorer(args: argparse.Namespace):
    from corrected_sgta.run_cecd_factorial_v1 import HuatuoScorer, HuluScorer

    if args.model_family == "huatuo":
        return HuatuoScorer(args.model_dir, args.huatuo_root, args.device)
    return HuluScorer(args.model_dir, args.max_visual_tokens)


def run(args: argparse.Namespace, rows: list[dict[str, Any]], inspection: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = full_config(args, rows, inspection)
    config = freeze_or_resume(candidate, args.output_dir / "config.json", args.resume)
    fingerprint = str(config["fingerprint"])
    inventory = image_inventory(rows, args.image_root)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for index, row in enumerate(rows):
        qid = str(row["qid"])
        row_hash = canonical_hash(row)
        image_hash = inventory[str(row["img_name"])]["sha256"]
        path = shard_root / shard_name(index, qid)
        if path.exists() and not valid_shard(path, fingerprint, qid, row_hash, image_hash):
            raise ValueError(f"invalid or drifted existing shard: {path}")
        if not path.exists():
            pending.append((index, row, path, row_hash, image_hash))

    scorer = None if not pending else build_scorer(args)
    for completed, (index, row, path, row_hash, image_hash) in enumerate(pending, 1):
        image_path = resolve_cxr_image(args.image_root, str(row["img_name"]))
        image = load_cxr_image(image_path)
        prompt = scoring_prompt(str(row["question"]))
        raw_scores = scorer.score(image, prompt)
        scores = public_scores(raw_scores)
        record = {
            "status": "complete",
            "version": VERSION,
            "config_fingerprint": fingerprint,
            "index": index,
            "qid": str(row["qid"]),
            "pair_id": str(row["pair_id"]),
            "arm": str(row["arm"]),
            "finding": str(row["finding"]),
            "source_qid": str(row.get("source_qid", "")),
            "img_name": str(row["img_name"]),
            "image_sha256": image_hash,
            "input_row_sha256": row_hash,
            "input_question_sha256": hashlib.sha256(str(row["question"]).encode()).hexdigest(),
            "scoring_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_suffix_conversion": {"from": INPUT_SUFFIX, "to": SCORING_SUFFIX},
            **scores,
            "generation_called": False,
            "reference_data_accessed": False,
        }
        reject_target_fields(record)
        atomic_json(path, record)
        print(f"[{completed}/{len(pending)}] {row['qid']}", flush=True)

    packed = []
    shard_hashes = []
    for index, row in enumerate(rows):
        path = shard_root / shard_name(index, str(row["qid"]))
        row_hash = canonical_hash(row)
        image_hash = inventory[str(row["img_name"])]["sha256"]
        if not valid_shard(path, fingerprint, str(row["qid"]), row_hash, image_hash):
            raise RuntimeError(f"missing valid shard after scoring: {path}")
        packed.append(json.loads(path.read_text(encoding="utf-8")))
        shard_hashes.append({"name": path.name, "sha256": sha256_file(path)})
    atomic_jsonl(args.output_dir / "tristate_margins.jsonl", packed)
    summary = {
        "status": "complete",
        "version": VERSION,
        "config_fingerprint": fingerprint,
        "rows": len(packed),
        "pairs": len({row["pair_id"] for row in packed}),
        "model_family": args.model_family,
        "tristate_margins_sha256": sha256_file(args.output_dir / "tristate_margins.jsonl"),
        "shards": shard_hashes,
        "generation_called": False,
        "reference_data_accessed": False,
    }
    reject_target_fields(summary)
    atomic_json(args.output_dir / "summary.json", summary)


def run_self_tests() -> None:
    accepted = {"selection_uses_target_label": False, "nested": [{"score": 1.0}]}
    reject_target_fields(accepted)
    for forbidden in ({"answer": "Yes"}, {"nested": [{"ground_truth": 1}]}, {"label": 0}):
        try:
            reject_target_fields(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target guard accepted: {forbidden}")
    converted = scoring_prompt("Question?\n" + INPUT_SUFFIX)
    assert converted.endswith(SCORING_SUFFIX) and "Uncertain" not in converted
    fake = {
        "logits": {"supported": 3.0, "refuted": 1.0, "undetermined": 2.0},
        "polarity": 2.0, "commitment": 1.0, "tristate_entropy": 0.5,
    }
    public = public_scores(fake)
    assert public["polarity_yes_minus_no"] == 2.0
    assert public["commitment_max_yes_no_minus_maybe"] == 1.0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        Image.new("L", (3, 2), 127).save(root / "image.png")
        assert resolve_cxr_image(root, "image.png") == (root / "image.png").resolve()
        assert load_cxr_image(root / "image.png").mode == "RGB"
        try:
            resolve_cxr_image(root, "../escape.png")
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("image traversal guard failed")
    print(json.dumps({"status": "passed", "tests": 8, "gpu_used": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-family", choices=("huatuo", "hulu"), default="huatuo")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--huatuo-root", type=Path, default=DEFAULT_HUATUO_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    args.model_dir = args.model_dir or DEFAULT_MODELS[args.model_family]
    rows = load_target_blind_rows(args.input)
    inspection = preflight(rows, args.input, args.image_root, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "preflight.json", inspection)
    if args.preflight_only:
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return
    run(args, rows, inspection)


if __name__ == "__main__":
    main()
