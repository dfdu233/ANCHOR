"""Fail-closed audit for pre-runner RULE answer caches.

Legacy answers can be useful local evidence, but they are not eligible for
fingerprinted runner reuse unless every protocol-critical field is attested.
This module never mutates or upgrades a legacy cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from corrected_sgta.rule_mitigation_backend import (
    CONV_MODE,
    render_rule_prompt,
    stable_json,
)

AUDIT_PROTOCOL = "rule-legacy-cache-audit-v1"
VERIFIED_REUSE = "verified_reuse"
LEGACY_REPRODUCTION = "legacy_local_reproduction"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _attested(
    manifest: dict[str, Any] | None,
    dotted_path: tuple[str, ...],
) -> Any:
    value: Any = manifest
    for key in dotted_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def audit_legacy_rule_cache(
    *,
    dataset: str,
    expected_rows: list[dict[str, Any]],
    answers_path: Path,
    metrics_path: Path,
    manifest_path: Path | None,
    expected_model_tree_sha256: str,
    expected_generation: dict[str, Any],
    expected_image_identity_sha256: str,
) -> dict[str, Any]:
    """Classify a cache without ever promoting incomplete provenance.

    The answer body can establish qid/prompt/image-path agreement. Exact image
    bytes, model weights, conversation mode, generation configuration, and the
    run fingerprint must come from a manifest generated at inference time.
    """
    answers = _load_jsonl(answers_path)
    metrics = _load_optional_json(metrics_path)
    manifest = _load_optional_json(manifest_path)

    expected_qids = [str(row["question_id"]) for row in expected_rows]
    answer_qids = [str(row.get("question_id")) for row in answers]
    expected_prompts = [
        render_rule_prompt(dataset, row) for row in expected_rows
    ]
    answer_prompts = [row.get("prompt") for row in answers]
    expected_images = [str(row.get("image")) for row in expected_rows]
    answer_images = [
        str(row.get("image", row.get("image_id"))) for row in answers
    ]

    manifest_generation = _attested(manifest, ("generation",))
    manifest_conv_mode = _attested(manifest, ("conv_mode",))
    manifest_model_sha = _attested(manifest, ("model", "tree_sha256"))
    manifest_image_sha = _attested(
        manifest, ("images", "identity_sha256")
    )
    manifest_fingerprint = _attested(manifest, ("fingerprint",))
    manifest_answers_sha = _attested(
        manifest, ("artifacts", "answers", "sha256")
    )

    checks = {
        "questions_qids_order": answer_qids == expected_qids,
        "questions_prompts_exact": answer_prompts == expected_prompts,
        "image_paths_exact": answer_images == expected_images,
        "image_content_identity_attested": (
            manifest_image_sha == expected_image_identity_sha256
        ),
        "model_tree_identity_attested": (
            manifest_model_sha == expected_model_tree_sha256
        ),
        "conv_mode_attested": manifest_conv_mode == CONV_MODE,
        "generation_attested": manifest_generation == expected_generation,
        "max_new_tokens_attested": (
            isinstance(manifest_generation, dict)
            and manifest_generation.get("max_new_tokens")
            == expected_generation.get("max_new_tokens")
        ),
        "run_fingerprint_attested": (
            isinstance(manifest_fingerprint, str)
            and bool(manifest_fingerprint)
        ),
        "answer_artifact_hash_attested": (
            manifest_answers_sha == _sha256_file(answers_path)
        ),
        "metrics_count_matches": (
            isinstance(metrics, dict)
            and metrics.get("n") == len(expected_rows)
        ),
    }
    verified = all(checks.values())
    missing_or_mismatched = [
        key for key, passed in checks.items() if not passed
    ]
    return {
        "audit_protocol": AUDIT_PROTOCOL,
        "classification": VERIFIED_REUSE if verified else LEGACY_REPRODUCTION,
        "verified_reuse": verified,
        "eligible_for_runner_completion": verified,
        "dataset": dataset,
        "sample_count": len(answers),
        "answers": {
            "path": str(answers_path),
            "sha256": _sha256_file(answers_path),
        },
        "metrics": {
            "path": str(metrics_path),
            "sha256": _sha256_file(metrics_path),
        },
        "manifest": (
            {"path": str(manifest_path), "present": True}
            if manifest is not None
            else {
                "path": str(manifest_path) if manifest_path else None,
                "present": False,
            }
        ),
        "checks": checks,
        "missing_or_mismatched": missing_or_mismatched,
        "ordered_answer_record_sha256": hashlib.sha256(
            stable_json(
                [
                    {
                        "question_id": qid,
                        "prompt": prompt,
                        "image": image,
                    }
                    for qid, prompt, image in zip(
                        answer_qids, answer_prompts, answer_images
                    )
                ]
            ).encode()
        ).hexdigest(),
        "reuse_policy": (
            "May be imported as a verified external cache."
            if verified
            else (
                "Local historical evidence only; must not be represented as a "
                "completed job under the fingerprinted runner."
            )
        ),
    }
