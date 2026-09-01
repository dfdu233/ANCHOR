"""Add a target-blind plain/no-retrieval arm to the frozen 128-row pilot v1.

The v1 manifest is read-only and retained byte-for-byte as the first 128 rows.
One plain arm per pair is appended in first-pair-occurrence order.  It changes
only the retrieved-report body to ``[none]``; image, instruction, and source
question are inherited exactly.  No target or model output is read.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from anchor.corrected_sgta.run_target_blind_canary_v1 import (
    load_target_blind_manifest,
    preflight_inputs,
)


ROOT = Path("corrected_runs/matched_retrieval_polarity_pilot_v1")
V1 = ROOT / "target_blind_pilot.json"
V1_RESULT = ROOT / "result.json"
V2 = ROOT / "target_blind_pilot_v2.json"
V2_RESULT = ROOT / "result_v2.json"
IMAGE_ROOT = Path("data/medheval/images")
PROTOCOL = "matched-retrieval-polarity-pilot-v2-plain-symmetry"
PLAIN_ARM = "plain"
EXPECTED_V1_ROWS = 128
EXPECTED_PAIRS = 32


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_prompt(prompt: str) -> str:
    marker = "Retrieved report:\n"
    question_marker = "\nQuestion:\n"
    if prompt.count(marker) != 1 or prompt.count(question_marker) != 1:
        raise ValueError("frozen prompt does not have one retrieval and question boundary")
    prefix, remainder = prompt.split(marker, 1)
    _, source_question = remainder.rsplit(question_marker, 1)
    return f"{prefix}{marker}[none]{question_marker}{source_question}"


def main() -> None:
    v1_hash_before = file_hash(V1)
    v1_result_hash_before = file_hash(V1_RESULT)
    v1_rows = load_target_blind_manifest(V1, limit=0)
    if len(v1_rows) != EXPECTED_V1_ROWS:
        raise RuntimeError(f"expected frozen v1 n={EXPECTED_V1_ROWS}, observed {len(v1_rows)}")

    pair_order = []
    arms_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in v1_rows:
        pair_id = str(row["pair_id"])
        if pair_id not in arms_by_pair:
            pair_order.append(pair_id)
            arms_by_pair[pair_id] = []
        arms_by_pair[pair_id].append(row)
    if len(pair_order) != EXPECTED_PAIRS or any(len(rows) != 4 for rows in arms_by_pair.values()):
        raise RuntimeError("frozen v1 is not exactly 32 pairs x 4 arms")

    plain_rows = []
    for pair_id in pair_order:
        source_rows = arms_by_pair[pair_id]
        images = {str(row["img_name"]) for row in source_rows}
        findings = {str(row["finding"]) for row in source_rows}
        source_qids = {str(row["source_qid"]) for row in source_rows}
        prompts = {str(row["question"]) for row in source_rows}
        if len(images) != 1 or len(findings) != 1 or len(source_qids) != 1:
            raise RuntimeError(f"pair identity drift in {pair_id}")
        source_questions = {
            prompt.rsplit("\nQuestion:\n", 1)[1] for prompt in prompts
        }
        instruction_prefixes = {
            prompt.split("Retrieved report:\n", 1)[0] for prompt in prompts
        }
        if len(source_questions) != 1 or len(instruction_prefixes) != 1:
            raise RuntimeError(f"instruction or source-question drift in {pair_id}")
        representative = source_rows[0]
        plain_rows.append({
            **{
                key: value for key, value in representative.items()
                if key not in {
                    "id", "qid", "arm", "question",
                    "tfidf_cosine_audit_only",
                    "present_absent_length_gap_audit_only",
                }
            },
            "id": f"{pair_id}:{PLAIN_ARM}",
            "qid": f"{pair_id}:{PLAIN_ARM}",
            "arm": PLAIN_ARM,
            "question": plain_prompt(representative["question"]),
            "plain_control_role": "no_retrieval_symmetry_origin",
            "selection_uses_target_label": False,
        })

    v2_rows = [*v1_rows, *plain_rows]
    V2.write_text(json.dumps(v2_rows, indent=2, sort_keys=True) + "\n")
    loaded = load_target_blind_manifest(V2, limit=0)
    preflight = preflight_inputs(loaded, IMAGE_ROOT)
    if loaded[:EXPECTED_V1_ROWS] != v1_rows:
        raise RuntimeError("v2 did not preserve v1 as an exact ordered prefix")
    if len(loaded) != EXPECTED_V1_ROWS + EXPECTED_PAIRS:
        raise RuntimeError("v2 row count is not 160")

    v1_hash_after = file_hash(V1)
    v1_result_hash_after = file_hash(V1_RESULT)
    if (v1_hash_before, v1_result_hash_before) != (v1_hash_after, v1_result_hash_after):
        raise RuntimeError("frozen v1 artifact changed during v2 build")

    result = {
        "status": "completed_target_blind_plain_symmetry_control_cpu_only",
        "protocol": PROTOCOL,
        "causal_design": {
            "estimand_gate": "m_present + m_absent approximately equals 2 * m_plain",
            "plain_definition": "same image, instruction, and source question; Retrieved report body is [none]",
            "target_or_model_output_read": False,
            "v1_frozen_as_exact_ordered_prefix": True,
            "v1_rows": EXPECTED_V1_ROWS,
            "plain_rows_appended": EXPECTED_PAIRS,
        },
        "counts": {
            "rows": len(loaded),
            "pairs": len(pair_order),
            "arms": dict(Counter(row["arm"] for row in loaded)),
            "findings": dict(Counter(row["finding"] for row in plain_rows)),
            "unique_qids": len({row["qid"] for row in loaded}),
        },
        "dedicated_runner_preflight": preflight,
        "preflight_command": (
            "PYTHONPATH=. .venv-full/bin/python -m "
            "anchor.corrected_sgta.run_target_blind_canary_v1 --model huatuo "
            f"--manifest {V2} --image-root {IMAGE_ROOT} "
            f"--output-dir {ROOT / 'huatuo_generation_v2'} --limit 0 "
            "--max-new-tokens 128 --preflight-only"
        ),
        "artifacts": {
            "target_blind_pilot_v2": str(V2),
            "target_blind_pilot_v2_sha256": file_hash(V2),
            "frozen_v1": str(V1),
            "frozen_v1_sha256_before": v1_hash_before,
            "frozen_v1_sha256_after": v1_hash_after,
            "frozen_v1_result_sha256_before": v1_result_hash_before,
            "frozen_v1_result_sha256_after": v1_result_hash_after,
        },
        "provenance": {
            "script": str(Path(__file__)),
            "script_sha256": file_hash(Path(__file__)),
            "image_root": str(IMAGE_ROOT.resolve()),
            "rebuild_command": (
                "PYTHONPATH=. .venv-full/bin/python "
                "anchor/corrected_sgta/build_matched_retrieval_polarity_pilot_v2.py"
            ),
        },
    }
    V2_RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
