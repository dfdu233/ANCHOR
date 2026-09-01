#!/usr/bin/env python3
"""Canary for sensor-frame perception versus patient-frame verbalization.

The model is asked to localize two known findings in *display coordinates*.
Those coordinates are then deterministically transported through the standard
radiological display transform (screen-left = patient-right).  This tests a
strict mechanism: whether laterality errors arise after correct pixel-frame
localization because the response uses the wrong reference frame.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corrected_sgta.run_huatuo_binding_conservation_probe_v1 import (
    VERSION as BINDING_VERSION,
    append_jsonl,
    atomic_json,
    build_cases,
    generate_oe,
    normalize_finding,
    parse_binding,
)
from corrected_sgta.run_huatuo_vindr_commitment_probe import dicom_to_pil, import_huatuo


VERSION = "huatuo-frame-transport-probe-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def screen_prompt(patient_left: str, patient_right: str) -> str:
    return (
        f"This displayed chest X-ray contains {normalize_finding(patient_left)} and "
        f"{normalize_finding(patient_right)}. Ignore patient-anatomical left and right. "
        "Using only screen coordinates, state which finding appears on the left side "
        "of the displayed image and which appears on the right side. Answer in one short sentence."
    )


def analyze(rows: list[dict[str, Any]], direct_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    parseable = [row for row in ok if row["screen_parse"] != "unparsed"]
    screen_accuracy = float(np.mean([row["screen_parse"] == "correct" for row in parseable])) if parseable else float("nan")
    # Coordinate transport is exact: correct screen coordinates become correct
    # patient coordinates under the known left-right involution.
    transport_accuracy = screen_accuracy
    matched_direct = [
        direct_rows[row["case_key"]] for row in parseable
        if row["case_key"] in direct_rows and direct_rows[row["case_key"]].get("oe_parse") != "unparsed"
    ]
    direct_accuracy = float(np.mean([row["oe_parse"] == "correct" for row in matched_direct])) if matched_direct else float("nan")
    gap = transport_accuracy - direct_accuracy if matched_direct else float("nan")
    gate = bool(len(parseable) >= 8 and screen_accuracy >= 0.80 and gap >= 0.20)
    return {
        "version": VERSION,
        "status": "GO_FRAME_MISMATCH_MECHANISM" if gate else "NO_GO_FRAME_MISMATCH_MECHANISM",
        "n_ok": len(ok),
        "n_parseable_screen": len(parseable),
        "screen_parse_rate": len(parseable) / len(ok) if ok else 0.0,
        "screen_localization_accuracy": screen_accuracy,
        "transported_patient_accuracy": transport_accuracy,
        "matched_direct_patient_n": len(matched_direct),
        "direct_patient_accuracy": direct_accuracy,
        "transport_minus_direct_accuracy": gap,
        "gate_passed": gate,
        "gate": "screen accuracy >=80%, transport-direct >=20pp, at least 8 parseable",
        "mathematical_object": (
            "laterality is a coordinate in a reference frame; the DICOM display map is an involution "
            "that swaps screen-left/right into patient-right/left"
        ),
        "boundary": (
            "a pass establishes a Huatuo frame-mismatch mechanism, not a general VLM mitigation result; "
            "Hulu, prompt paraphrases, orientation controls, and natural reports remain required"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("/workspace/vinbigdata/train.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("/workspace/vinbigdata/train"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--huatuo-root", type=Path, default=Path("/home/dbw/HuatuoGPT-Vision"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/dbw/models/HuatuoGPT-Vision-7B"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    direct_rows = {
        row["case_key"]: row
        for row in (
            json.loads(line) for line in (args.binding_run / "raw.jsonl").read_text().splitlines() if line.strip()
        )
    }
    cases = build_cases(args.csv, args.image_root, args.limit, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config = {
        "version": VERSION,
        "created_at": now(),
        "model": "HuatuoGPT-Vision-7B",
        "binding_source_version": BINDING_VERSION,
        "binding_run": str(args.binding_run.resolve()),
        "seed": args.seed,
        "limit": args.limit,
        "cases": cases,
        "intervention": "decode display-frame location, then apply known screen-left/patient-right involution",
        "claim_conservation": "no finding is added or deleted; only the reference frame changes",
    }
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume")
        old = json.loads(config_path.read_text())
        for key in ("version", "model", "binding_source_version", "binding_run", "seed", "limit", "cases", "intervention"):
            if old[key] != config[key]:
                raise RuntimeError(f"resume config drift: {key}")
    else:
        atomic_json(config_path, config)

    raw_path = args.output_dir / "raw.jsonl"
    completed: set[str] = set()
    if raw_path.exists() and args.resume:
        completed = {json.loads(line)["case_key"] for line in raw_path.read_text().splitlines() if line.strip()}

    HuatuoChatbot = import_huatuo(args.huatuo_root)
    bot = HuatuoChatbot(str(args.model_dir), device="cuda:0")
    bot.debug = False
    for case in cases:
        if case["case_key"] in completed:
            continue
        row: dict[str, Any] = {**case, "version": VERSION, "status": "error"}
        try:
            image = dicom_to_pil(Path(case["image"]))
            prompt = screen_prompt(case["left_finding"], case["right_finding"])
            answer = generate_oe(bot, image, prompt, args.max_new_tokens)
            # Patient-right lies on screen-left; patient-left lies on screen-right.
            screen_parse = parse_binding(answer, case["right_finding"], case["left_finding"])
            row.update(
                {
                    "status": "ok",
                    "screen_prompt": prompt,
                    "screen_answer": answer,
                    "screen_parse": screen_parse,
                    "transported_patient_parse": screen_parse,
                    "completed_at": now(),
                }
            )
        except Exception as error:
            row.update({"error": repr(error), "traceback": traceback.format_exc(), "completed_at": now()})
        append_jsonl(raw_path, row)
        completed.add(case["case_key"])
        print(f"[{len(completed)}/{len(cases)}] {case['case_key']} status={row['status']} screen={row.get('screen_parse')}", flush=True)

    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    atomic_json(args.output_dir / "analysis.json", analyze(rows, direct_rows))


if __name__ == "__main__":
    main()
