"""Fail-closed acceptance of long native OE generation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifact_registry import latest_by_artifact
from .hashing import sha256_file
from .store import atomic_write_json


VERSION = "native-oe-full-acceptance-v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accept_model(
    *, model: str, output_dir: Path, expected_qids: list[str], manifest_sha256: str,
    registry_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    answers_path = output_dir / "answers.jsonl"
    qualification_path = output_dir / "qualification.json"
    config_path = output_dir / "generation_config.json"
    evaluation_path = output_dir / "evaluation_lexical_auxiliary.json"
    for path in (answers_path, qualification_path, config_path, evaluation_path):
        if not path.is_file():
            raise RuntimeError(f"{model}: required artifact is absent: {path}")

    answers = _jsonl(answers_path)
    qids = [str(row.get("question_id", "")) for row in answers]
    if qids != expected_qids or len(set(qids)) != len(qids):
        raise RuntimeError(f"{model}: answer qids are not the exact manifest sequence")
    if any(not str(row.get("text", "")).strip() for row in answers):
        raise RuntimeError(f"{model}: empty answer detected")
    if any(row.get("model_id") != model for row in answers):
        raise RuntimeError(f"{model}: model identity mismatch in answers")

    qualification = json.loads(qualification_path.read_text())
    required_qualification = {
        "passed": True,
        "status": "passed",
        "artifact_status": "admissible",
        "expected_count": len(expected_qids),
        "received_count": len(expected_qids),
        "exact_qid_alignment": True,
        "max_new_tokens": 256,
    }
    for key, expected in required_qualification.items():
        if qualification.get(key) != expected:
            raise RuntimeError(f"{model}: qualification field {key} did not pass")
    if float(qualification.get("cap_hit_rate", 1.0)) > 0.05:
        raise RuntimeError(f"{model}: token cap-hit rate exceeds 5%")
    if float(qualification.get("terminal_completeness_rate", 0.0)) < 0.95:
        raise RuntimeError(f"{model}: required response-form completeness is below 95%")

    config = json.loads(config_path.read_text())
    generation = config.get("generation", {})
    if (
        config.get("model") != model
        or config.get("manifest_sha256") != manifest_sha256
        or config.get("max_new_tokens") != 256
        or generation.get("do_sample") is not False
        or generation.get("num_beams") != 1
    ):
        raise RuntimeError(f"{model}: frozen greedy256 generation contract mismatch")

    answer_hash = sha256_file(answers_path)
    event = registry_events.get(str(answers_path.resolve()))
    if (
        not event
        or event.get("status") != "admissible"
        or event.get("artifact_sha256") != answer_hash
        or event.get("qualification_sha256") != sha256_file(qualification_path)
    ):
        raise RuntimeError(f"{model}: current artifact lacks matching admissible registry evidence")

    evaluation = json.loads(evaluation_path.read_text())
    if evaluation.get("answer_sha256") != [answer_hash]:
        raise RuntimeError(f"{model}: lexical auxiliary evaluation is stale")
    return {
        "model": model,
        "answers": str(answers_path.resolve()),
        "answers_sha256": answer_hash,
        "n": len(answers),
        "qualification_sha256": sha256_file(qualification_path),
        "generation_config_sha256": sha256_file(config_path),
        "lexical_auxiliary_sha256": sha256_file(evaluation_path),
        "cap_hit_rate": qualification["cap_hit_rate"],
        "terminal_completeness_rate": qualification["terminal_completeness_rate"],
        "median_prediction_tokens": qualification["median_prediction_tokens"],
        "clinical_claim_efficacy_status": "pending_separate_claim_evaluation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model-output", nargs=2, action="append", metavar=("MODEL", "DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected_qids = [str(row["qid"]) for row in manifest]
    events = latest_by_artifact(args.registry)
    rows = [
        accept_model(
            model=model,
            output_dir=Path(directory),
            expected_qids=expected_qids,
            manifest_sha256=sha256_file(args.manifest),
            registry_events=events,
        )
        for model, directory in args.model_output
    ]
    result = {
        "protocol_version": VERSION,
        "passed": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registry": str(args.registry.resolve()),
        "models": rows,
        "interpretation": (
            "This accepts generation/provenance/response form only. It does not certify "
            "clinical correctness or hallucination mitigation."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
