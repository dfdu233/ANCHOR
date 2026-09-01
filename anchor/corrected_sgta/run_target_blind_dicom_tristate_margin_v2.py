#!/usr/bin/env python3
"""Target-blind Yes/No/Maybe margin scoring for controlled VinDr DICOM prompts.

This runner is deliberately narrower than a general evaluator.  It consumes only
the frozen treatment prompt and image reference, renders VinDr DICOMs with the
project's established renderer, and calls ``HuatuoScorer.score`` or
``HuluScorer.score``.  It never generates text and never reads target votes,
labels, or reference answers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image

from corrected_sgta.run_target_blind_tristate_margin_v1 import (
    DEFAULT_HUATUO_ROOT,
    DEFAULT_MODELS,
    INPUT_SUFFIX,
    SCORING_SUFFIX,
    atomic_json,
    atomic_jsonl,
    build_scorer,
    canonical_hash,
    cheap_model_inventory,
    public_scores,
    scoring_prompt,
    sha256_file,
)


VERSION = "target-blind-dicom-tristate-margin-v2"
DEFAULT_INPUT = Path(
    "corrected_runs/reader_grounded_controlled_source_injection_v1/"
    "target_blind_discovery.json"
)
DEFAULT_IMAGE_ROOT = Path("/workspace/vinbigdata")
DEFAULT_GPU_LOCK = Path(
    "/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock"
)
EXPECTED_ARMS = {
    "current_present",
    "current_absent",
    "current_uncertain",
    "other_present",
    "other_absent",
    "other_uncertain",
    "plain",
    "random_unrelated_state",
}
FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "correctanswer",
    "expectedanswer",
    "gtans",
    "gtanswer",
    "groundtruth",
    "gold",
    "goldanswer",
    "label",
    "labels",
    "target",
    "targets",
    "reference",
    "referenceanswer",
    "referenceanswers",
    "truth",
    "readervote",
    "readervotes",
    "readerlabel",
    "readerlabels",
    "readersupport",
    "positivevotes",
    "supportcount",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_key(key: object) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def reject_target_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = normalized_key(key)
            if normalized in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden target-bearing field at {path}.{key}")
            if normalized in {"selectionusestargetvote", "selectionusestargetlabel"}:
                if child is not False:
                    raise ValueError(f"target-blind audit flag must be false at {path}.{key}")
            reject_target_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_target_fields(child, f"{path}[{index}]")


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reject_target_fields(payload)
    if not isinstance(payload, list) or not payload:
        raise ValueError("manifest must be a non-empty JSON list")
    rows: list[dict[str, Any]] = []
    required = {
        "qid",
        "pair_id",
        "arm",
        "finding",
        "img_name",
        "question",
        "selection_uses_target_vote",
    }
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest row {index} is not an object")
        row = dict(raw)
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"manifest row {index} missing fields: {missing}")
        if row["selection_uses_target_vote"] is not False:
            raise ValueError(f"row {index} is not explicitly target-vote blind")
        if row["arm"] not in EXPECTED_ARMS:
            raise ValueError(f"row {index} has unexpected arm: {row['arm']!r}")
        if row.get("experiment_split") != "discovery":
            raise ValueError(f"row {index} is not in the frozen discovery split")
        if row.get("controlled_source_injection_not_natural_rag") is not True:
            raise ValueError(f"row {index} lacks the controlled-injection audit flag")
        if not str(row["question"]).endswith(INPUT_SUFFIX):
            raise ValueError(f"row {index} violates the frozen answer suffix")
        rows.append(row)

    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("manifest qids are not unique")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    for pair_id, members in pairs.items():
        if len(members) != len(EXPECTED_ARMS):
            raise ValueError(f"pair {pair_id} does not contain exactly eight rows")
        if {str(row["arm"]) for row in members} != EXPECTED_ARMS:
            raise ValueError(f"pair {pair_id} does not contain the frozen eight arms")
        for field in ("img_name", "finding"):
            if len({str(row[field]) for row in members}) != 1:
                raise ValueError(f"pair {pair_id} disagrees on {field}")
    return rows


def resolve_dicom(image_root: Path, relative: str) -> Path:
    root = image_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"DICOM escapes image root: {relative!r}") from error
    if not candidate.is_file() or candidate.suffix.lower() not in {".dcm", ".dicom"}:
        raise ValueError(f"expected a real DICOM file: {candidate}")
    return candidate


def standard_renderer():
    from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil

    return dicom_to_pil


def renderer_fingerprint() -> dict[str, Any]:
    import pydicom

    renderer = standard_renderer()
    source_path = Path(inspect.getsourcefile(renderer) or renderer.__code__.co_filename).resolve()
    callable_source = inspect.getsource(renderer)
    return {
        "callable": f"{renderer.__module__}.{renderer.__name__}",
        "source_path": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "callable_source_sha256": hashlib.sha256(callable_source.encode()).hexdigest(),
        "contract": (
            "rescale slope/intercept; finite-pixel 0.5/99.5 percentile window; "
            "MONOCHROME1 inversion; uint8 RGB"
        ),
        "numpy_version": str(np.__version__),
        "pydicom_version": str(pydicom.__version__),
        "pillow_version": str(getattr(sys.modules.get("PIL"), "__version__", "unknown")),
    }


def load_dicom(path: Path) -> Image.Image:
    image = standard_renderer()(path)
    if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
        raise ValueError(f"standard renderer returned invalid image: {path}")
    return image


def dicom_inventory(rows: list[dict[str, Any]], image_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for relative in sorted({str(row["img_name"]) for row in rows}):
        path = resolve_dicom(image_root, relative)
        inventory[relative] = {
            "relative_path": relative,
            "resolved_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def render_one_audit(path: Path) -> dict[str, Any]:
    image = load_dicom(path)
    try:
        pixels = np.asarray(image)
        return {
            "path": str(path),
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "rendered_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "minimum": int(pixels.min()),
            "maximum": int(pixels.max()),
        }
    finally:
        image.close()


def preflight(
    rows: list[dict[str, Any]], input_path: Path, image_root: Path, model_dir: Path
) -> dict[str, Any]:
    inventory = dicom_inventory(rows, image_root)
    first_relative = sorted(inventory)[0]
    prompts = [scoring_prompt(str(row["question"])) for row in rows]
    return {
        "status": "passed_target_blind_dicom_cpu_preflight",
        "version": VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "rows": len(rows),
        "pairs": len({str(row["pair_id"]) for row in rows}),
        "arms": sorted({str(row["arm"]) for row in rows}),
        "findings": sorted({str(row["finding"]) for row in rows}),
        "unique_dicoms": len(inventory),
        "dicom_inventory": inventory,
        "dicom_inventory_fingerprint": canonical_hash(inventory),
        "renderer": renderer_fingerprint(),
        "one_real_dicom_render": render_one_audit(Path(inventory[first_relative]["resolved_path"])),
        "model_cheap_inventory": cheap_model_inventory(model_dir),
        "scoring_prompt_fingerprint": canonical_hash(prompts),
        "recursive_target_field_guard": "passed",
        "target_or_vote_data_accessed": False,
        "generation": "forbidden; score() only",
    }


def full_config(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    inspection: dict[str, Any],
) -> dict[str, Any]:
    from corrected_sgta.run_cecd_factorial_v1 import (
        environment_fingerprint,
        full_model_artifact_fingerprint,
        python_source_tree_fingerprint,
    )

    scorer_source = Path(__file__).with_name("run_cecd_factorial_v1.py")
    v1_source = Path(__file__).with_name("run_target_blind_tristate_margin_v1.py")
    candidate = {
        "version": VERSION,
        "created_at": utc_now(),
        "command": sys.argv,
        "model_family": args.model_family,
        "model_dir": str(args.model_dir.resolve()),
        "model_artifact": full_model_artifact_fingerprint(args.model_dir),
        "external_huatuo_runtime": (
            python_source_tree_fingerprint(args.huatuo_root)
            if args.model_family == "huatuo"
            else None
        ),
        "device": args.device,
        "max_visual_tokens": args.max_visual_tokens,
        "input": inspection,
        "row_order_fingerprint": canonical_hash([str(row["qid"]) for row in rows]),
        "safe_row_fingerprint": canonical_hash(rows),
        "prompt_conversion": {"from": INPUT_SUFFIX, "to": SCORING_SUFFIX},
        "image_root": str(args.image_root.resolve()),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "source_sha256": {
            "runner": sha256_file(Path(__file__)),
            "v1_helpers": sha256_file(v1_source),
            "scorer": sha256_file(scorer_source),
        },
        "environment": environment_fingerprint(),
        "score_contract": (
            "HuatuoScorer/HuluScorer.score only; FP32 final-hidden @ Yes/No/Maybe lm-head; "
            "standard frozen DICOM renderer; no generation and no target/vote access"
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
    expected = canonical_hash(
        {key: value for key, value in existing.items() if key not in {"created_at", "command", "fingerprint"}}
    )
    if existing.get("fingerprint") != expected:
        raise ValueError("stored config fingerprint is internally invalid")
    return existing


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
        and row.get("dicom_sha256") == image_hash
        and set(row.get("tristate_logits_fp32", {})) == {"Yes", "No", "Maybe"}
    )


@contextmanager
def canonical_gpu_lock(path: Path, wait: bool) -> Iterator[None]:
    canonical = DEFAULT_GPU_LOCK.resolve()
    if path.resolve() != canonical:
        raise ValueError(f"GPU lock drift: VinDr runner requires {canonical}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        operation = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as error:
            raise RuntimeError(
                f"canonical GPU lock is busy: {path}; use --wait-for-gpu-lock to queue"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(args: argparse.Namespace, rows: list[dict[str, Any]], inspection: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = freeze_or_resume(
        full_config(args, rows, inspection), args.output_dir / "config.json", args.resume
    )
    fingerprint = str(config["fingerprint"])
    inventory = inspection["dicom_inventory"]
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, dict[str, Any], Path, str, str]] = []
    for index, row in enumerate(rows):
        qid = str(row["qid"])
        row_hash = canonical_hash(row)
        image_hash = str(inventory[str(row["img_name"])]["sha256"])
        path = shard_root / shard_name(index, qid)
        if path.exists() and not valid_shard(path, fingerprint, qid, row_hash, image_hash):
            raise ValueError(f"invalid or drifted existing shard: {path}")
        if not path.exists():
            pending.append((index, row, path, row_hash, image_hash))

    with canonical_gpu_lock(args.gpu_lock, args.wait_for_gpu_lock):
        scorer = None if not pending else build_scorer(args)
        for completed, (index, row, path, row_hash, image_hash) in enumerate(pending, 1):
            dicom_path = resolve_dicom(args.image_root, str(row["img_name"]))
            image = load_dicom(dicom_path)
            try:
                prompt = scoring_prompt(str(row["question"]))
                scores = public_scores(scorer.score(image, prompt))
            finally:
                image.close()
            record = {
                "status": "complete",
                "version": VERSION,
                "config_fingerprint": fingerprint,
                "index": index,
                "qid": str(row["qid"]),
                "pair_id": str(row["pair_id"]),
                "arm": str(row["arm"]),
                "finding": str(row["finding"]),
                "img_name": str(row["img_name"]),
                "dicom_sha256": image_hash,
                "input_row_sha256": row_hash,
                "input_question_sha256": hashlib.sha256(str(row["question"]).encode()).hexdigest(),
                "scoring_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_suffix_conversion": {"from": INPUT_SUFFIX, "to": SCORING_SUFFIX},
                **scores,
                "generation_called": False,
                "target_or_vote_data_accessed": False,
            }
            reject_target_fields(record)
            atomic_json(path, record)
            print(f"[{completed}/{len(pending)}] {row['qid']}", flush=True)

    packed: list[dict[str, Any]] = []
    shard_hashes: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        path = shard_root / shard_name(index, str(row["qid"]))
        row_hash = canonical_hash(row)
        image_hash = str(inventory[str(row["img_name"])]["sha256"])
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
        "target_or_vote_data_accessed": False,
    }
    reject_target_fields(summary)
    atomic_json(args.output_dir / "summary.json", summary)


def run_self_tests() -> None:
    reject_target_fields({"selection_uses_target_vote": False, "nested": [{"score": 1.0}]})
    for forbidden in (
        {"answer": "Yes"},
        {"nested": [{"reader_votes": [1, 0, 1]}]},
        {"selection_uses_target_vote": True},
    ):
        try:
            reject_target_fields(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target guard accepted: {forbidden}")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "image.dicom").write_bytes(b"not-a-real-dicom")
        assert resolve_dicom(root, "image.dicom") == (root / "image.dicom").resolve()
        try:
            resolve_dicom(root, "../escape.dicom")
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError("DICOM traversal guard failed")
        try:
            with canonical_gpu_lock(root / "wrong-gpu.lock", wait=False):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("non-canonical GPU lock was accepted")
    fake = {
        "logits": {"supported": 3.0, "refuted": 1.0, "undetermined": 2.0},
        "polarity": 2.0,
        "commitment": 1.0,
        "tristate_entropy": 0.5,
    }
    public = public_scores(fake)
    assert math.isclose(public["polarity_yes_minus_no"], 2.0)
    assert math.isclose(public["commitment_max_yes_no_minus_maybe"], 1.0)
    print(json.dumps({"status": "passed", "tests": 9, "gpu_used": False}, indent=2))


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
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--wait-for-gpu-lock", action="store_true")
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
    rows = load_rows(args.input)
    inspection = preflight(rows, args.input, args.image_root, args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "preflight.json", inspection)
    if args.preflight_only:
        summary = {
            key: inspection[key]
            for key in (
                "status",
                "version",
                "rows",
                "pairs",
                "arms",
                "findings",
                "unique_dicoms",
                "renderer",
                "one_real_dicom_render",
                "recursive_target_field_guard",
                "target_or_vote_data_accessed",
                "generation",
            )
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    run(args, rows, inspection)


if __name__ == "__main__":
    main()
