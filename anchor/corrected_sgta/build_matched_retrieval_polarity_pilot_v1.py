"""Build a target-blind 32-pair pilot from the frozen 108-pair canary.

Selection is cohort/finding membership followed only by SHA256 order.  TF-IDF
cosine is never an ordering or replacement criterion: after hash selection, a
predeclared minimum quality gate is audited and fails closed.  Both JSONL and
the JSON-list schema consumed by ``run_native_oe_vqa`` are emitted.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

from anchor.medeval.run_native_oe_vqa import load_rows, qid
from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    load_target_blind_manifest,
    preflight_inputs as target_blind_preflight_inputs,
)


SOURCE_DIR = Path("corrected_runs/matched_retrieval_polarity_canary_v1")
SOURCE_ROWS = SOURCE_DIR / "canary.jsonl"
SOURCE_PAIRS = SOURCE_DIR / "matched_pairs.jsonl"
SOURCE_RESULT = SOURCE_DIR / "result.json"
OUT_DIR = Path("corrected_runs/matched_retrieval_polarity_pilot_v1")
PILOT_JSON = OUT_DIR / "pilot.json"
PILOT_JSONL = OUT_DIR / "pilot.jsonl"
TARGET_BLIND_JSON = OUT_DIR / "target_blind_pilot.json"
RESULT_OUT = OUT_DIR / "result.json"
FULL_CONFIRMATION_OUT = OUT_DIR / "full108_confirmation.json"
IMAGE_ROOT = Path("data/medheval/images")
PROTOCOL = "matched-retrieval-polarity-pilot-v1"
SEED = 20260810
FINDINGS = ("pleural_effusion", "cardiomegaly", "pneumothorax", "lung_opacity")
PAIRS_PER_FINDING = 8
ARMS = ("present", "absent", "neutral", "random_deletion")
MIN_TFIDF_COSINE = 0.10
MAX_LENGTH_GAP = 0.10
BLINDED_ANSWER = "__TARGET_BLINDED__"

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    source_rows = read_jsonl(SOURCE_ROWS)
    source_pairs = read_jsonl(SOURCE_PAIRS)
    source_result = json.loads(SOURCE_RESULT.read_text())
    pairs_by_id = {row["pair_id"]: row for row in source_pairs}
    arms_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        arms_by_pair.setdefault(row["pair_id"], {})[row["arm"]] = row

    selected_pairs = []
    selection_audit = {}
    quality_failures = []
    for finding in FINDINGS:
        eligible = [
            row for row in source_pairs
            if row["cohort"] == "cxr_vishal" and row["finding"] == finding
        ]
        ordered = sorted(
            eligible,
            key=lambda row: stable_hash(f"{PROTOCOL}:hash-selection:{finding}:{row['pair_id']}"),
        )
        chosen = ordered[:PAIRS_PER_FINDING]
        selection_audit[finding] = {
            "eligible_cxr_pairs": len(eligible),
            "required": PAIRS_PER_FINDING,
            "selected_pair_ids_in_hash_order": [row["pair_id"] for row in chosen],
            "selection_uses_cosine": False,
        }
        if len(chosen) != PAIRS_PER_FINDING:
            raise RuntimeError(
                f"{finding}: strict pilot needs {PAIRS_PER_FINDING} pairs, found {len(chosen)}"
            )
        for row in chosen:
            if row["tfidf_cosine"] < MIN_TFIDF_COSINE or row["present_absent_length_gap"] > MAX_LENGTH_GAP:
                quality_failures.append({
                    "pair_id": row["pair_id"], "finding": finding,
                    "tfidf_cosine": row["tfidf_cosine"],
                    "length_gap": row["present_absent_length_gap"],
                })
        selected_pairs.extend(chosen)

    # Do not substitute another pair if a hash-selected item misses quality.
    if quality_failures:
        raise RuntimeError(
            "predeclared pilot quality gate failed after target/cosine-blind hash selection: "
            + json.dumps(quality_failures, sort_keys=True)
        )

    runner_rows = []
    missing_images = []
    decode_failures = []
    for pair in selected_pairs:
        pair_id = pair["pair_id"]
        arm_rows = arms_by_pair.get(pair_id, {})
        if set(arm_rows) != set(ARMS):
            raise RuntimeError(f"{pair_id}: source does not have exactly the four frozen arms")
        for arm in ARMS:
            source = arm_rows[arm]
            image_path = IMAGE_ROOT / source["image"]
            if not image_path.is_file():
                missing_images.append(str(image_path))
            else:
                try:
                    with Image.open(image_path) as image:
                        image.convert("RGB").load()
                except Exception as exc:  # exact runner-relevant decode check
                    decode_failures.append({"path": str(image_path), "error": f"{type(exc).__name__}: {exc}"})
            runner_rows.append({
                "id": f"{pair_id}:{arm}",
                "qid": f"{pair_id}:{arm}",
                "img_name": source["image"],
                "question": source["question"],
                "answer": BLINDED_ANSWER,
                "task": "target_blind_causal_generation",
                "prompt_contract": PROTOCOL,
                "dataset": "cxr_vishal",
                "question_type": "binary_target_blinded",
                "pair_id": pair_id,
                "arm": arm,
                "finding": pair["finding"],
                "source_qid": source["source_id"],
                "selection_uses_target_label": False,
                "tfidf_cosine_audit_only": pair["tfidf_cosine"],
                "present_absent_length_gap_audit_only": pair["present_absent_length_gap"],
            })

    if missing_images or decode_failures:
        raise RuntimeError(json.dumps({
            "missing_images": missing_images,
            "decode_failures": decode_failures,
        }, sort_keys=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_JSON.write_text(json.dumps(runner_rows, indent=2, sort_keys=True) + "\n")
    PILOT_JSONL.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in runner_rows))
    target_blind_rows = [
        {key: value for key, value in row.items() if key != "answer"}
        for row in runner_rows
    ]
    TARGET_BLIND_JSON.write_text(
        json.dumps(target_blind_rows, indent=2, sort_keys=True) + "\n"
    )

    # Exercise the exact runner loader without importing any model adapter.
    loaded_by_runner = load_rows(PILOT_JSON, limit=0)
    observed_qids = [qid(row) for row in loaded_by_runner]
    required_runner_fields = {"qid", "img_name", "question", "answer"}
    schema_failures = [
        row.get("qid", f"index-{index}")
        for index, row in enumerate(loaded_by_runner)
        if not required_runner_fields <= set(row)
    ]
    if len(observed_qids) != len(set(observed_qids)) or schema_failures:
        raise RuntimeError("runner manifest schema or qid uniqueness failed")
    target_blind_loaded = load_target_blind_manifest(TARGET_BLIND_JSON, limit=0)
    target_blind_preflight = target_blind_preflight_inputs(
        target_blind_loaded, IMAGE_ROOT
    )
    if [qid(row) for row in target_blind_loaded] != observed_qids:
        raise RuntimeError("target-blind manifest changed qid order")
    if any(
        target_blind_loaded[index]["question"] != loaded_by_runner[index]["question"]
        or target_blind_loaded[index]["img_name"] != loaded_by_runner[index]["img_name"]
        for index in range(len(target_blind_loaded))
    ):
        raise RuntimeError("target-blind manifest changed image or prompt content")

    selected_ids = {row["pair_id"] for row in selected_pairs}
    source_ids = {row["pair_id"] for row in source_pairs}
    full_confirmation = {
        "status": "confirmed_frozen_for_later_not_modified",
        "source_protocol": source_result["protocol"],
        "source_pair_count": len(source_pairs),
        "source_arm_count": len(source_rows),
        "source_pair_ids_unique": len(source_ids) == len(source_pairs),
        "source_every_pair_four_arms": all(len(arms_by_pair.get(pair_id, {})) == 4 for pair_id in source_ids),
        "pilot_pair_count": len(selected_ids),
        "remaining_for_later_pair_count": len(source_ids - selected_ids),
        "source_artifacts_unchanged": {
            "canary_jsonl": str(SOURCE_ROWS), "canary_sha256": file_hash(SOURCE_ROWS),
            "pairs_jsonl": str(SOURCE_PAIRS), "pairs_sha256": file_hash(SOURCE_PAIRS),
            "result": str(SOURCE_RESULT), "result_sha256": file_hash(SOURCE_RESULT),
        },
        "note": "This confirmation references the complete 108-pair artifact; it neither copies nor edits it.",
    }
    FULL_CONFIRMATION_OUT.write_text(json.dumps(full_confirmation, indent=2, sort_keys=True) + "\n")

    similarities = [row["tfidf_cosine"] for row in selected_pairs]
    length_gaps = [row["present_absent_length_gap"] for row in selected_pairs]
    result = {
        "status": "completed_target_blind_cpu_manifest_only",
        "protocol": PROTOCOL,
        "selection": {
            "cohort": "cxr_vishal_only_for_one_runner_image_root",
            "findings": list(FINDINGS),
            "pairs_per_finding": PAIRS_PER_FINDING,
            "selection_rule": "within cohort+finding, ascending SHA256(protocol,finding,pair_id); first 8",
            "selection_uses_target_label": False,
            "selection_uses_model_output": False,
            "selection_uses_cosine": False,
            "quality_is_post_selection_fail_closed_not_ranking": True,
            "audit": selection_audit,
        },
        "predeclared_quality_gate": {
            "minimum_tfidf_cosine": MIN_TFIDF_COSINE,
            "maximum_present_absent_length_gap": MAX_LENGTH_GAP,
            "failed_hash_selected_pairs": quality_failures,
            "passed": not quality_failures,
        },
        "counts": {
            "pairs": len(selected_pairs),
            "arms": len(runner_rows),
            "pairs_by_finding": dict(Counter(row["finding"] for row in selected_pairs)),
            "arms_by_name": dict(Counter(row["arm"] for row in runner_rows)),
            "distinct_source_images": len({row["img_name"] for row in runner_rows}),
        },
        "matching_quality": {
            "tfidf_cosine_min": min(similarities),
            "tfidf_cosine_median": float(np.median(similarities)),
            "tfidf_cosine_mean": float(np.mean(similarities)),
            "tfidf_cosine_max": max(similarities),
            "length_gap_max": max(length_gaps),
            "length_gap_median": float(np.median(length_gaps)),
        },
        "runner_validation": {
            "runner": "anchor.medeval.run_native_oe_vqa",
            "runner_json_list_loaded_n": len(loaded_by_runner),
            "unique_qids": len(observed_qids) == len(set(observed_qids)),
            "required_fields": sorted(required_runner_fields),
            "schema_failures": schema_failures,
            "image_root": str(IMAGE_ROOT.resolve()),
            "all_image_paths_exist": not missing_images,
            "all_images_pil_rgb_decode": not decode_failures,
            "target_placeholder": BLINDED_ANSWER,
            "warning": "Generation is runnable; standard GT efficacy evaluation is forbidden because answer is intentionally blinded.",
        },
        "target_blind_runner_validation": {
            "runner": "anchor.corrected_sgta.run_target_blind_canary_v1",
            "manifest_loaded_n": len(target_blind_loaded),
            "qid_order_identical_to_legacy_pilot": True,
            "image_and_question_content_identical_to_legacy_pilot": True,
            "preflight": target_blind_preflight,
            "forbidden_answer_target_reference_fields": 0,
            "preflight_command": (
                "PYTHONPATH=. .venv-full/bin/python -m "
                "anchor.corrected_sgta.run_target_blind_canary_v1 --model huatuo "
                f"--manifest {TARGET_BLIND_JSON} --image-root {IMAGE_ROOT} "
                f"--output-dir {OUT_DIR / 'huatuo_generation'} --limit 0 "
                "--max-new-tokens 128 --preflight-only"
            ),
        },
        "artifacts": {
            "runner_json": str(PILOT_JSON), "runner_json_sha256": file_hash(PILOT_JSON),
            "audit_jsonl": str(PILOT_JSONL), "audit_jsonl_sha256": file_hash(PILOT_JSONL),
            "target_blind_runner_json": str(TARGET_BLIND_JSON),
            "target_blind_runner_json_sha256": file_hash(TARGET_BLIND_JSON),
            "full108_confirmation": str(FULL_CONFIRMATION_OUT),
            "full108_confirmation_sha256": file_hash(FULL_CONFIRMATION_OUT),
        },
        "provenance": {
            "script": str(Path(__file__)), "script_sha256": file_hash(Path(__file__)),
            "source_canary": str(SOURCE_ROWS), "source_canary_sha256": file_hash(SOURCE_ROWS),
            "source_pairs": str(SOURCE_PAIRS), "source_pairs_sha256": file_hash(SOURCE_PAIRS),
            "source_result": str(SOURCE_RESULT), "source_result_sha256": file_hash(SOURCE_RESULT),
            "seed": SEED,
            "rebuild_command": "PYTHONPATH=. .venv-full/bin/python anchor/corrected_sgta/build_matched_retrieval_polarity_pilot_v1.py",
        },
    }
    RESULT_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
