#!/usr/bin/env python3
"""Persistently advance the frozen physician-OE pipeline as real returns arrive.

This monitor never creates clinical annotations or attestations.  It waits for
explicit human-return files, validates them fail-closed, and advances only the
already pre-registered blinded workflow.  Invalid or incomplete returns remain
visible in the heartbeat and are never converted into a successful job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file
from anchor.medeval.analyze_physician_oe_multiarm import (
    VERSION as ANALYSIS_VERSION,
)
from anchor.medeval.analyze_physician_oe_multiarm_v2 import (
    EXPECTED_GATE_NAMES as T3_EXPECTED_GATE_NAMES,
    EXPECTED_GATE_SPEC as T3_EXPECTED_GATE_SPEC,
    VERSION as T3_ANALYSIS_VERSION,
)
from anchor.medeval.store import atomic_write_json
from anchor.medeval.validate_physician_oe_review import (
    VERSION as VALIDATION_VERSION,
    load_jsonl,
    validate_completed,
)


VERSION = "anchor-physician-oe-clinical-pipeline-monitor-v1"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_BASE = (
    ROOT
    / "corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1"
)
DEFAULT_INBOX = (
    Path("/home/dbw/datasets/public/vqa_rad_hf/physician_review_returns")
    / "vqa_rad_t2_multiarm_v1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def freeze_copy(source: Path, directory: Path, label: str) -> Path:
    digest = sha256_file(source)
    suffix = "".join(source.suffixes) or ".dat"
    target = directory / f"{label}.{digest[:16]}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"frozen target hash mismatch: {target}")
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy hash mismatch: {source}")
    os.replace(temporary, target)
    return target


def validate_return(template: Path, completed: Path, output: Path) -> dict[str, Any]:
    result = validate_completed(load_jsonl(template), load_jsonl(completed))
    result.update(
        {
            "template": str(template.resolve()),
            "template_sha256": sha256_file(template),
            "completed": str(completed.resolve()),
            "completed_sha256": sha256_file(completed),
        }
    )
    if result.get("protocol_version") != VALIDATION_VERSION:
        raise RuntimeError("unexpected physician-review validation version")
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior != result:
            raise RuntimeError(f"validation output collision: {output}")
    else:
        atomic_write_json(output, result)
    return result


def load_attestation(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "adjudicator_id",
        "attest_model_blinded",
        "attest_no_private_mapping",
    }
    if set(row) != allowed:
        raise ValueError(f"attestation keys must be exactly {sorted(allowed)}")
    if not isinstance(row["adjudicator_id"], str) or not row["adjudicator_id"].strip():
        raise ValueError("adjudicator_id must be nonempty")
    if row["attest_model_blinded"] is not True:
        raise ValueError("attest_model_blinded must be true")
    if row["attest_no_private_mapping"] is not True:
        raise ValueError("attest_no_private_mapping must be true")
    return row


def run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )


def paths(base: Path, inbox: Path, output: Path) -> dict[str, Path]:
    return {
        "master": base / "review.template.jsonl",
        "mapping": base / "review.private_mapping.jsonl",
        "prereg": base / "clinical_analysis_prereg_v1.json",
        "template_a": base / "deliveries_v1/reviewer_A.blinded.jsonl",
        "template_b": base / "deliveries_v1/reviewer_B.blinded.jsonl",
        "return_a": inbox / "reviewer_A.completed.jsonl",
        "return_b": inbox / "reviewer_B.completed.jsonl",
        "clarification": inbox / "clarification_log.frozen.md",
        "adjudication_return": inbox / "adjudication.completed.jsonl",
        "attestation": inbox / "adjudication.attestation.json",
        "frozen": output / "frozen",
        "adjudication_template": output / "adjudication.blinded.jsonl",
        "preparation": output / "adjudication.preparation.json",
        "consensus": output / "consensus.clean.jsonl",
        "provenance": output / "consensus.provenance.json",
        "analysis": output / "clinical_analysis.json",
    }


def validate_analysis_artifact(p: dict[str, Path]) -> dict[str, Any]:
    """Recompute the frozen analysis' local provenance before declaring complete."""

    analysis = json.loads(p["analysis"].read_text(encoding="utf-8"))
    if analysis.get("protocol_version") not in {ANALYSIS_VERSION, T3_ANALYSIS_VERSION}:
        raise RuntimeError("physician OE analysis protocol mismatch")
    prereg = json.loads(p["prereg"].read_text(encoding="utf-8"))
    analysis_module = str(
        prereg.get("analysis_module", "anchor.medeval.analyze_physician_oe_multiarm")
    )
    if analysis_module not in {
        "anchor.medeval.analyze_physician_oe_multiarm",
        "anchor.medeval.analyze_physician_oe_multiarm_v2",
    }:
        raise RuntimeError("unapproved physician OE analysis module")
    expected_version = (
        T3_ANALYSIS_VERSION
        if analysis_module.endswith("_v2")
        else ANALYSIS_VERSION
    )
    if analysis.get("protocol_version") != expected_version:
        raise RuntimeError("physician OE analysis protocol mismatch")
    baseline = str(prereg.get("baseline", ""))
    iterations = int(prereg.get("bootstrap_iterations", 0))
    seed = int(prereg.get("bootstrap_seed", -1))
    prereg_provenance = prereg.get("provenance", {})
    frozen_sources = {
        "prepare_adjudication_source_sha256": ROOT
        / "anchor/medeval/prepare_physician_oe_adjudication.py",
        "finalize_consensus_source_sha256": ROOT
        / "anchor/medeval/finalize_physician_oe_consensus.py",
        "analysis_source_sha256": ROOT
        / "anchor/medeval/analyze_physician_oe_multiarm.py",
    }
    if analysis_module.endswith("_v2"):
        frozen_sources["analysis_wrapper_source_sha256"] = (
            ROOT / "anchor/medeval/analyze_physician_oe_multiarm_v2.py"
        )
    if (
        prereg.get("protocol_version") != "anchor-physician-oe-multiarm-prereg-v1"
        or prereg.get("frozen_before_physician_labels") is not True
        or prereg.get("clinical_labels_inspected") is not False
        or not baseline
        or iterations <= 0
        or seed < 0
        or prereg_provenance.get("review_template_sha256")
        != sha256_file(p["master"])
        or prereg_provenance.get("private_mapping_sha256")
        != sha256_file(p["mapping"])
        or any(
            prereg_provenance.get(key) != sha256_file(path)
            for key, path in frozen_sources.items()
        )
    ):
        raise RuntimeError("physician OE frozen preregistration/hash mismatch")
    if (
        analysis.get("baseline") != baseline
        or analysis.get("bootstrap_iterations") != iterations
        or analysis.get("seed") != seed
    ):
        raise RuntimeError("physician OE analysis departed from frozen statistics")
    methods = analysis.get("methods")
    contrasts = analysis.get("contrasts")
    aggregates = analysis.get("aggregates")
    promoted = analysis.get("promoted_methods")
    if (
        not isinstance(methods, list)
        or len(methods) != len(set(methods))
        or baseline not in methods
        or not isinstance(contrasts, dict)
        or set(contrasts) != set(methods) - {baseline}
        or not isinstance(aggregates, dict)
        or set(aggregates) != set(methods)
        or not isinstance(promoted, list)
        or not set(promoted).issubset(set(contrasts))
        or set(methods)
        != {baseline, *map(str, prereg.get("candidate_methods", []))}
    ):
        raise RuntimeError("physician OE analysis method/contrast closure mismatch")
    if any(row.get("versus") != baseline for row in contrasts.values()):
        raise RuntimeError("physician OE contrasts do not use the frozen baseline")
    if analysis_module.endswith("_v2"):
        if (
            prereg.get("machine_gate_spec") != T3_EXPECTED_GATE_SPEC
            or analysis.get("machine_gate_spec") != T3_EXPECTED_GATE_SPEC
            or analysis.get("evidence_stage") != "T3"
            or any(
                set(row.get("promotion_gates", {})) != T3_EXPECTED_GATE_NAMES
                or row.get("t3_promotion_authorized")
                is not all(row.get("promotion_gates", {}).values())
                for row in contrasts.values()
            )
        ):
            raise RuntimeError("T3 machine gate specification or decision drifted")
    expected_provenance = {
        "template": str(p["master"].resolve()),
        "template_sha256": sha256_file(p["master"]),
        "consensus": str(p["consensus"].resolve()),
        "consensus_sha256": sha256_file(p["consensus"]),
        "consensus_provenance": str(p["provenance"].resolve()),
        "consensus_provenance_sha256": sha256_file(p["provenance"]),
        "mapping": str(p["mapping"].resolve()),
        "mapping_sha256": sha256_file(p["mapping"]),
    }
    if analysis_module.endswith("_v2"):
        expected_provenance.update(
            {
                "prereg": str(p["prereg"].resolve()),
                "prereg_sha256": sha256_file(p["prereg"]),
            }
        )
    if analysis.get("provenance") != expected_provenance:
        raise RuntimeError("physician OE analysis provenance/hash mismatch")
    return analysis


def write_inbox_instructions(inbox: Path) -> None:
    target = inbox / "RETURN_FILES.md"
    if target.exists():
        return
    atomic_text(
        target,
        """# Physician OE return inbox

The monitor never creates clinical labels. Place the following human-completed
files here using these exact names:

1. `reviewer_A.completed.jsonl`
2. `reviewer_B.completed.jsonl`
3. `clarification_log.frozen.md` (must not contain the template `- Pending.`)

After the monitor validates and freezes both reviews, it creates a still-blinded
`adjudication.blinded.jsonl` in the analysis output directory. A third blinded
clinician returns:

4. `adjudication.completed.jsonl`
5. `adjudication.attestation.json` with exactly:

```json
{
  "adjudicator_id": "nonempty-blinded-clinician-id",
  "attest_model_blinded": true,
  "attest_no_private_mapping": true
}
```

Do not place a private method mapping or model identity in this inbox.
Copy each return under a temporary filename and rename it to the exact name
only after the copy completes. The monitor also requires unchanged bytes across
two polls before accepting any human input.
""",
    )


def human_input_signatures(p: dict[str, Path]) -> dict[str, dict[str, Any]]:
    signatures = {}
    for key in (
        "return_a",
        "return_b",
        "clarification",
        "adjudication_return",
        "attestation",
    ):
        path = p[key]
        if path.is_file():
            signatures[key] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return signatures


def advance(p: dict[str, Path]) -> dict[str, Any]:
    if p["analysis"].exists():
        analysis = validate_analysis_artifact(p)
        return {
            "stage": "complete",
            "analysis": str(p["analysis"]),
            "analysis_sha256": sha256_file(p["analysis"]),
            "promoted_methods": analysis["promoted_methods"],
        }

    if p["consensus"].exists() or p["provenance"].exists():
        if not p["consensus"].exists() or not p["provenance"].exists():
            raise RuntimeError("partial consensus/provenance write requires audit")
        prereg = json.loads(p["prereg"].read_text(encoding="utf-8"))
        analysis_module = str(
            prereg.get(
                "analysis_module", "anchor.medeval.analyze_physician_oe_multiarm"
            )
        )
        command = [
                sys.executable,
                "-m",
                analysis_module,
                "--template",
                str(p["master"]),
                "--consensus",
                str(p["consensus"]),
                "--consensus-provenance",
                str(p["provenance"]),
                "--mapping",
                str(p["mapping"]),
                "--baseline",
                str(prereg["baseline"]),
                "--bootstrap-iterations",
                str(int(prereg["bootstrap_iterations"])),
                "--seed",
                str(int(prereg["bootstrap_seed"])),
                "--output",
                str(p["analysis"]),
        ]
        if analysis_module.endswith("_v2"):
            command.extend(["--prereg", str(p["prereg"])])
        run_command(command)
        analysis = validate_analysis_artifact(p)
        return {
            "stage": "complete",
            "analysis": str(p["analysis"]),
            "analysis_sha256": sha256_file(p["analysis"]),
            "promoted_methods": analysis["promoted_methods"],
        }

    if p["preparation"].exists() or p["adjudication_template"].exists():
        if not p["preparation"].exists() or not p["adjudication_template"].exists():
            raise RuntimeError("partial adjudication preparation requires audit")
        missing = [
            str(p[key])
            for key in ("adjudication_return", "attestation")
            if not p[key].is_file()
        ]
        if missing:
            return {"stage": "waiting_for_blinded_adjudication", "missing": missing}
        attestation = load_attestation(p["attestation"])
        frozen_adjudication = freeze_copy(
            p["adjudication_return"], p["frozen"], "adjudication.completed"
        )
        frozen_attestation = freeze_copy(
            p["attestation"], p["frozen"], "adjudication.attestation"
        )
        run_command(
            [
                sys.executable,
                "-m",
                "anchor.medeval.finalize_physician_oe_consensus",
                "--master-template",
                str(p["master"]),
                "--adjudication-template",
                str(p["adjudication_template"]),
                "--completed-adjudication",
                str(frozen_adjudication),
                "--preparation-manifest",
                str(p["preparation"]),
                "--adjudicator-id",
                attestation["adjudicator_id"],
                "--attest-model-blinded",
                "--attest-no-private-mapping",
                "--output-consensus",
                str(p["consensus"]),
                "--output-provenance",
                str(p["provenance"]),
            ]
        )
        return {
            "stage": "consensus_frozen",
            "frozen_attestation_sha256": sha256_file(frozen_attestation),
            "next": "pre_registered_analysis",
        }

    required = ("return_a", "return_b", "clarification")
    missing = [str(p[key]) for key in required if not p[key].is_file()]
    if missing:
        return {"stage": "waiting_for_independent_reviews", "missing": missing}
    clarification = p["clarification"].read_text(encoding="utf-8")
    if not clarification.strip() or "- Pending." in clarification:
        raise ValueError("clarification log is not frozen")
    frozen_a = freeze_copy(p["return_a"], p["frozen"], "reviewer_A.completed")
    frozen_b = freeze_copy(p["return_b"], p["frozen"], "reviewer_B.completed")
    frozen_clarification = freeze_copy(
        p["clarification"], p["frozen"], "clarification_log.frozen"
    )
    validation_a = p["frozen"] / f"reviewer_A.{sha256_file(frozen_a)[:16]}.validation.json"
    validation_b = p["frozen"] / f"reviewer_B.{sha256_file(frozen_b)[:16]}.validation.json"
    validate_return(p["template_a"], frozen_a, validation_a)
    validate_return(p["template_b"], frozen_b, validation_b)
    run_command(
        [
            sys.executable,
            "-m",
            "anchor.medeval.prepare_physician_oe_adjudication",
            "--master-template",
            str(p["master"]),
            "--reviewer-a-template",
            str(p["template_a"]),
            "--reviewer-a-completed",
            str(frozen_a),
            "--reviewer-a-validation",
            str(validation_a),
            "--reviewer-b-template",
            str(p["template_b"]),
            "--reviewer-b-completed",
            str(frozen_b),
            "--reviewer-b-validation",
            str(validation_b),
            "--clarification-log",
            str(frozen_clarification),
            "--output-template",
            str(p["adjudication_template"]),
            "--output-manifest",
            str(p["preparation"]),
        ]
    )
    return {
        "stage": "adjudication_prepared",
        "adjudication_template": str(p["adjudication_template"]),
        "adjudication_template_sha256": sha256_file(p["adjudication_template"]),
        "next": "waiting_for_blinded_adjudication",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    output = args.output or args.base / "clinical_returns_v1"
    heartbeat = args.heartbeat or output / "monitor.heartbeat.json"
    args.inbox.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    write_inbox_instructions(args.inbox)
    resolved = paths(args.base, args.inbox, output)
    for key in ("master", "mapping", "prereg", "template_a", "template_b"):
        if not resolved[key].is_file():
            raise FileNotFoundError(resolved[key])

    previous_signatures: dict[str, dict[str, Any]] = {}
    while True:
        try:
            signatures = human_input_signatures(resolved)
            if signatures and signatures != previous_signatures:
                state = {
                    "stage": "waiting_for_stable_human_inputs",
                    "required_unchanged_polls": 2,
                    "human_input_signatures": signatures,
                }
            else:
                state = advance(resolved)
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "inbox": str(args.inbox.resolve()),
                "output": str(output.resolve()),
                "clinical_labels_synthesized": False,
                "private_mapping_joined_before_consensus": False,
                **state,
            }
            previous_signatures = signatures
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            payload = {
                "version": VERSION,
                "time": utc_now(),
                "pid": os.getpid(),
                "inbox": str(args.inbox.resolve()),
                "output": str(output.resolve()),
                "stage": "input_or_transition_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "clinical_labels_synthesized": False,
                "private_mapping_joined_before_consensus": False,
            }
        atomic_write_json(heartbeat, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if payload.get("stage") == "complete" or args.once:
            return
        if payload.get("stage") == "consensus_frozen":
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
