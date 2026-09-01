#!/usr/bin/env python3
"""Validate four VinDr-listing admission returns without advancing science.

This persistent monitor is deliberately terminal at a write-once handoff for
genuine human adjudication.  It does not open the sealed mapping, decide
equivalence, create an admission receipt, run a model, or authorize GPU use.
Human inputs must use the eight exact role filenames and remain byte-stable
over two polls before validation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file
from anchor.medeval.store import atomic_write_json
from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import ROLES
from corrected_sgta.prepare_vindr_cecd_listing_adjudication_handoff_v1 import (
    prepare_handoff,
)
from corrected_sgta.validate_vindr_cecd_listing_admission_returns_v1 import (
    validate_all,
)
from corrected_sgta.verify_vindr_cecd_listing_admission_pack_v1 import verify


VERSION = "vindr-cecd-listing-return-monitor-v2-human-handoff"
ROOT = Path("/home/dbw/ANCHOR")
DEFAULT_PACK = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "cecd_listing_admission_pack_v1"
)
DEFAULT_INBOX = Path(
    "/home/dbw/datasets/physionet/vindr-cxr/1.0.0/"
    "cecd_listing_admission_returns_v1"
)
DEFAULT_HEARTBEAT = (
    ROOT
    / "corrected_runs/vindr_v2/cecd_listing_admission_returns_v1/"
    "monitor.heartbeat.json"
)
DEFAULT_HANDOFF = (
    ROOT
    / "corrected_runs/vindr_v2/cecd_listing_admission_returns_v1/"
    "human_adjudication_handoff_v1"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def return_paths(inbox: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    completed = {
        role: inbox / f"{role}.completed.csv"
        for role in ROLES
    }
    attestations = {
        role: inbox / f"{role}.attestation.json"
        for role in ROLES
    }
    return completed, attestations


def input_signatures(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.is_file()
    }


def unexpected_files(inbox: Path, allowed_names: set[str]) -> list[str]:
    """Reject aliases while permitting hidden/in-progress atomic-copy files."""

    return sorted(
        path.name
        for path in inbox.iterdir()
        if path.is_file()
        and path.name not in allowed_names
        and not path.name.startswith(".")
        and not path.name.endswith((".tmp", ".partial"))
    )


def advance(
    *,
    pack: Path,
    completed: dict[str, Path],
    attestations: dict[str, Path],
    handoff_dir: Path | None = None,
) -> dict[str, Any]:
    paths = [*completed.values(), *attestations.values()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {
            "stage": "waiting_for_four_independent_returns",
            "missing": missing,
            "returns_present": len(paths) - len(missing),
            "returns_required": len(paths),
        }
    result = validate_all(
        pack_dir=pack,
        completed=completed,
        attestations=attestations,
    )
    state = {
        "stage": "four_independent_returns_structurally_valid",
        "validated_roles": [
            {"role": row["role"], "rows": row["rows"]}
            for row in result["roles"]
        ],
        "admission_decision_computed": False,
    }
    if handoff_dir is not None:
        handoff = prepare_handoff(
            pack_dir=pack,
            completed=completed,
            attestations=attestations,
            output_dir=handoff_dir,
        )
        state.update(
            {
                "stage": "ready_for_human_adjudication",
                "adjudication_handoff": str((handoff_dir / "handoff.json").resolve()),
                "adjudication_handoff_fingerprint": handoff["fingerprint"],
                "adjudication_still_required": True,
                "admission_receipt_created": False,
            }
        )
    return state


def heartbeat_payload(
    *, pack: Path, inbox: Path, state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "time": utc_now(),
        "pid": os.getpid(),
        "pack": str(pack.resolve()),
        "pack_manifest_sha256": sha256_file(pack / "manifest.json"),
        "inbox": str(inbox.resolve()),
        "clinical_or_prompt_labels_synthesized": False,
        "attestations_synthesized": False,
        "sealed_mapping_opened": False,
        "returns_merged": False,
        "adjudication_performed": False,
        "admission_decision_computed": False,
        "model_scoring_authorized": False,
        "gpu_authorized": False,
        "model_or_gpu_launched": False,
        **state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--heartbeat", type=Path, default=DEFAULT_HEARTBEAT)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError("interval must be at least one second")
    args.inbox.mkdir(parents=True, exist_ok=True)
    integrity = verify(args.pack)
    if integrity.get("passed") is not True:
        raise RuntimeError("listing admission pack integrity failed")
    completed, attestations = return_paths(args.inbox)
    all_paths = [*completed.values(), *attestations.values()]
    allowed_names = {path.name for path in all_paths}
    previous_signatures: dict[str, dict[str, Any]] = {}

    while True:
        try:
            aliases = unexpected_files(args.inbox, allowed_names)
            signatures = input_signatures(all_paths)
            if aliases:
                state: dict[str, Any] = {
                    "stage": "unexpected_inbox_files",
                    "unexpected_files": aliases,
                    "expected_filenames": sorted(allowed_names),
                }
            elif signatures and signatures != previous_signatures:
                state = {
                    "stage": "waiting_for_stable_human_inputs",
                    "required_unchanged_polls": 2,
                    "human_input_signatures": signatures,
                }
            else:
                state = advance(
                    pack=args.pack,
                    completed=completed,
                    attestations=attestations,
                    handoff_dir=args.handoff_dir,
                )
                state["human_input_signatures"] = signatures
            payload = heartbeat_payload(
                pack=args.pack,
                inbox=args.inbox,
                state=state,
            )
            previous_signatures = signatures
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            payload = heartbeat_payload(
                pack=args.pack,
                inbox=args.inbox,
                state={
                    "stage": "return_validation_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        atomic_write_json(args.heartbeat, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.once or payload["stage"] == "ready_for_human_adjudication":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
