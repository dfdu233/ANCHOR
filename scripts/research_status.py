#!/usr/bin/env python3
"""Concise reconnect status for the detached ANCHOR experiment pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/dbw/ANCHOR")


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def alive(pid: object) -> bool:
    try:
        value = int(pid)
        state = Path(f"/proc/{value}/stat").read_text().split()[2]
        return state != "Z"
    except (ValueError, OSError, IndexError, TypeError):
        return False


def detached_job_rows(root: Path = ROOT) -> tuple[list[dict[str, Any]], int]:
    """Return manifest jobs with PID-derived state and count hidden stale history."""

    manifest_path = root / "configs/research_active_jobs.json"
    active_names = set(json.loads(manifest_path.read_text()).get("active_jobs", []))
    visible: list[dict[str, Any]] = []
    historical_stale = 0
    for path in sorted((root / "corrected_runs/detached_jobs").glob("*.json")):
        row = json.loads(path.read_text())
        if "name" not in row or "status" not in row:
            continue
        raw_status = str(row.get("status", "unknown"))
        # A `running` state represents the workload child, not merely a live
        # supervisor.  During `starting`, the supervisor may legitimately
        # exist before child_pid has been published.
        process = (
            row.get("child_pid")
            if raw_status == "running"
            else row.get("child_pid") or row.get("pid")
        )
        live = alive(process)
        effective_status = (
            "stale"
            if raw_status in {"starting", "running"} and not live
            else raw_status
        )
        name = str(row.get("name", path.stem))
        if name in active_names:
            visible.append(
                {
                    "name": name,
                    "raw_status": raw_status,
                    "status": effective_status,
                    "pid": process,
                    "live": live,
                }
            )
        elif effective_status == "stale":
            historical_stale += 1
    return visible, historical_stale


def print_detached_jobs(root: Path = ROOT) -> None:
    rows, historical_stale = detached_job_rows(root)
    print("Active detached jobs")
    for row in rows:
        marker = "alive" if row["live"] else "not-alive"
        print(f"  {row['name']:44s} {row['status']:8s} {marker}")
    if historical_stale:
        print(f"  ({historical_stale} historical stale state files hidden)")


def reader_boundary_progress(root: Path = ROOT) -> list[tuple[str, str]]:
    """Report the frozen reader-boundary decision, never absence-as-pending."""

    decision_path = (
        root / "corrected_runs/paper/iclr_oral_completion_audit_v1/audit.json"
    )
    decision: dict[str, Any] = {}
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text())
    rejected = bool(
        decision.get("version") == "iclr-oral-completion-audit-v1"
        and decision.get("current_authoritative_verdict")
        == "REJECT_CURRENT_READER_BOUNDARY_AND_PIVOT"
        and decision.get("killed_branches", {}).get("reader_early_erasure")
        == "data-refuted at frozen Huatuo development gate"
    )
    if rejected:
        return [
            (
                "VinDr Huatuo dev residual screen",
                "terminal/rejected: frozen development gate failed",
            ),
            (
                "VinDr Hulu dev residual screen",
                "terminal/rejected: post-hoc rescue prohibited",
            ),
            (
                "VinDr huatuo confirmation",
                "terminal/rejected: confirmation not authorized",
            ),
            (
                "VinDr hulu confirmation",
                "terminal/rejected: confirmation not authorized",
            ),
            (
                "VinDr two-model boundary synthesis",
                "terminal/rejected: reader-boundary branch closed",
            ),
        ]

    manifest_path = root / "configs/research_active_jobs.json"
    active_names = (
        set(json.loads(manifest_path.read_text()).get("active_jobs", []))
        if manifest_path.is_file()
        else set()
    )
    live_rows, _ = detached_job_rows(root) if manifest_path.is_file() else ([], 0)
    job_states = {row["name"]: row["status"] for row in live_rows}

    def state_for(
        *, complete: bool, job_names: tuple[str, ...], progress: str | None = None
    ) -> str:
        if complete:
            return "complete"
        active = [name for name in job_names if name in active_names]
        if not active:
            return "inactive/not-authorized"
        states = [job_states.get(name, "active/missing-state") for name in active]
        state = ",".join(states)
        return f"{state}: {progress}" if progress is not None else state

    huatuo_dev = (
        root
        / "corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3/reader_residual_dev_unanimity_v1.json"
    )
    hulu_dev = (
        root
        / "corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1/reader_residual_dev_screen_v1.json"
    )
    confirmation_jobs = (
        "vindr-v2-confirmation-collections-v1",
        "vindr-v2-confirmation-collections-v2",
        "vindr-v2-confirmation-collections-v3",
    )
    outputs = []
    outputs.append(
        (
            "VinDr Huatuo dev residual screen",
            state_for(
                complete=huatuo_dev.is_file(),
                job_names=("vindr-v2-huatuo-reader-residual-unanimity-v1",),
            ),
        )
    )
    outputs.append(
        (
            "VinDr Hulu dev residual screen",
            state_for(
                complete=hulu_dev.is_file(),
                job_names=(
                    "vindr-v2-hulu-dev-and-screens-v1",
                    "vindr-v2-hulu-dev-and-screens-v2",
                    "vindr-v2-hulu-dev-and-screens-v3",
                ),
            ),
        )
    )
    for model in ("huatuo", "hulu"):
        metadata = (
            root
            / f"corrected_runs/vindr_v2/hidden_confirmation_{model}_all_findings_v1/metadata.jsonl"
        )
        count = line_count(metadata)
        outputs.append(
            (
                f"VinDr {model} confirmation",
                state_for(
                    complete=count >= 1920,
                    job_names=confirmation_jobs,
                    progress=f"{count}/1920",
                ),
            )
        )
    summary = (
        root
        / "corrected_runs/vindr_v2/reader_residual_confirmation_v1/two_model_summary_v1.json"
    )
    outputs.append(
        (
            "VinDr two-model boundary synthesis",
            state_for(
                complete=summary.is_file(),
                job_names=("vindr-v2-confirmation-analysis-v1",),
            ),
        )
    )
    return outputs


def latest_existing(pattern: str, suffix: str) -> Path | None:
    candidates = [path / suffix for path in ROOT.glob(pattern) if (path / suffix).exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main() -> None:
    print_detached_jobs()
    print("\nProgress")
    paths = [
        (
            "LLaVA MIMIC greedy",
            ROOT / "corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl",
            694,
        ),
        (
            "Hulu three-state screen",
            ROOT / "corrected_runs/missing_third_state/mimic_report_triplets_v1/hulu_scores_v1/raw.jsonl",
            138,
        ),
        (
            "Hulu report qualification",
            ROOT / "corrected_runs/unified_eval/smoke/hulu_mimic_report_v2/predictions.jsonl",
            32,
        ),
        (
            "Hulu MIMIC report full",
            ROOT / "corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl",
            694,
        ),
        (
            "Hulu true-mismatch audit v4",
            ROOT / "corrected_runs/unified_eval/sanity/hulu_mimic_report_dependency_v4/generation.raw.jsonl",
            192,
        ),
        (
            "Canonical LLaVA VQA-RAD OE full",
            ROOT / "corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_v1/answers.jsonl",
            200,
        ),
        (
            "Native Hulu VQA-RAD OE full",
            ROOT / "corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_v1/answers.jsonl",
            200,
        ),
        (
            "Native Huatuo VQA-RAD OE full",
            ROOT / "corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v1/answers.jsonl",
            200,
        ),
    ]
    for name, path, total in paths:
        count = line_count(path)
        print(f"  {name:44s} {count:4d}/{total}")
    for name, status in reader_boundary_progress():
        print(f"  {name:44s} {status}")
    render_root = ROOT / "corrected_runs/vindr_v2/dicom_render_huatuo_pilot_v1"
    render_config = render_root / "config.json"
    if render_config.exists():
        config = json.loads(render_config.read_text())
        expected = int(config.get("selected_claims", 0))
        completed = len(list((render_root / "shards").glob("*.json")))
        print(f"  {'VinDr Huatuo paired DICOM render':44s} {completed:4d}/{expected}")
        print(
            f"  {'VinDr DICOM render formal analysis':44s} "
            f"{'complete' if (render_root / 'analysis_v1.json').exists() else 'waiting/pending'}"
        )
    second_identity = (
        ROOT
        / "corrected_runs/unified_eval/sanity/second_identity_conformance_v1/conformance.json"
    )
    if second_identity.exists():
        payload = json.loads(second_identity.read_text())
        if not payload.get("passed"):
            print(f"  {'SECOND VQA-RAD OE':44s} blocked: identity + recursion")
    smoke_state = ROOT / "corrected_runs/unified_eval/smoke/mitigation_matrix_v1/queue_state.jsonl"
    if smoke_state.exists():
        rows = [json.loads(line) for line in smoke_state.read_text().splitlines() if line.strip()]
        latest = {row["method"]: row["status"] for row in rows}
        print(f"  {'Legacy report mitigation smoke':44s} {latest}")
    # Pin the live protocol version. Selecting the latest *existing* queue can
    # silently fall back to a completed historical run while the current full
    # run has not started yet, which mixes incompatible generation contracts.
    vqa_smoke = (
        ROOT
        / "corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v4/queue_state.jsonl"
    )
    if vqa_smoke.exists():
        rows = [json.loads(line) for line in vqa_smoke.read_text().splitlines() if line.strip()]
        latest = {row["method"]: row["status"] for row in rows}
        print(f"  {'VQA-RAD OE mitigation smoke':44s} {latest}")
    vqa_full = (
        ROOT
        / "corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v4/queue_state.jsonl"
    )
    if vqa_full.exists():
        rows = [json.loads(line) for line in vqa_full.read_text().splitlines() if line.strip()]
        completed = {}
        for row in rows:
            completed[row["method"]] = completed.get(row["method"], 0) + int(
                row["status"] in {"done", "skipped"}
            )
        print(f"  {'VQA-RAD OE full completed chunks':44s} {completed}")
        evaluation = vqa_full.parent / "evaluation" / "summary.json"
        if evaluation.exists():
            payload = json.loads(evaluation.read_text())
            print(
                f"  {'VQA-RAD structural evaluations':44s} "
                f"{payload.get('completed_methods', [])}"
            )
            print(
                f"  {'VQA-RAD scientific validity':44s} "
                f"{payload.get('common_plumbing_valid')} "
                f"{payload.get('validity_reasons', [])}"
            )
    else:
        conformance = (
            ROOT
            / "corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v4/greedy_backend_conformance.json"
        )
        if conformance.exists():
            payload = json.loads(conformance.read_text())
            if not payload.get("passed"):
                print(
                    f"  {'VQA-RAD OE full (current v4)':44s} "
                    "blocked by backend identity gate"
                )
            else:
                print(f"  {'VQA-RAD OE full (current v4)':44s} pending")
        else:
            print(f"  {'VQA-RAD OE full (current v4)':44s} pending smoke qualification")
    hulu_v4 = ROOT / "corrected_runs/unified_eval/sanity/hulu_mimic_report_dependency_v4/summary.json"
    if hulu_v4.exists():
        payload = json.loads(hulu_v4.read_text())
        print(
            f"  {'Hulu true-mismatch dependency audit':44s} "
            f"{payload.get('admissible_for_report_generation_claim')}"
        )
    port_diagnostic = ROOT / "corrected_runs/unified_eval/sanity/llava_mitigation_port_diagnostic_v1/summary.json"
    if port_diagnostic.exists():
        payload = json.loads(port_diagnostic.read_text())
        print(f"  {'Keyword-stopping failure confirmed':44s} {payload.get('port_failure_confirmed')}")
    full_state = ROOT / "corrected_runs/unified_eval/full/mitigation_matrix_v1/queue_state.jsonl"
    if full_state.exists():
        rows = [json.loads(line) for line in full_state.read_text().splitlines() if line.strip()]
        latest = {}
        for row in rows:
            latest[row["method"]] = latest.get(row["method"], 0) + int(
                row["status"] in {"done", "skipped"}
            )
        print(f"  {'Full mitigation completed chunks':44s} {latest}")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    print("\nGPU processes")
    print(result.stdout.strip() or "  none")
    heartbeat = ROOT / "corrected_runs/detached_jobs/watchdog-heartbeat.json"
    print("\nRecovery watchdog")
    if not heartbeat.exists():
        print("  not started")
    else:
        row = json.loads(heartbeat.read_text())
        stamp = datetime.fromisoformat(row["time"])
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        state = "healthy" if age <= 90 and alive(row.get("watchdog_pid")) else "stale"
        print(f"  {state}; heartbeat age {age:.0f}s; PID {row.get('watchdog_pid')}")
        print(f"  {row.get('decisions', {})}")


if __name__ == "__main__":
    main()
