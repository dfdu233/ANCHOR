#!/usr/bin/env python3
"""Fail-closed two-model authorizer for Evidence Addressability Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "evidence-addressability-raw-stage2-joint-authorizer-v1"
MODEL_PROTOCOL = "evidence-addressability-raw-increment-gate-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute_model_gate(payload: dict) -> bool:
    confirmation = payload["confirmation"]
    bootstrap = confirmation["bootstrap"]
    permutation = confirmation["conditional_randomization"]
    return bool(
        confirmation["per_reader_log_loss_relative_improvement"] >= 0.05
        and bootstrap["log_loss_delta_ci95"][0] > 0
        and confirmation["reader_support_brier_relative_improvement"] >= 0.05
        and bootstrap["brier_delta_ci95"][0] > 0
        and confirmation["positive_nll_findings"] >= 5
        and permutation["plus_one_p_value"] <= 0.05
    )


def validate_attached_artifacts(result_path: Path, payload: dict) -> None:
    predictions = Path(payload["predictions"])
    marker = Path(payload["opened_marker"])
    if (
        not predictions.is_file()
        or sha256(predictions) != payload.get("predictions_sha256")
        or not marker.is_file()
        or sha256(marker) != payload.get("opened_marker_sha256")
    ):
        raise ValueError(f"missing or hash-mismatched attached artifact: {result_path}")
    opened = json.loads(marker.read_text(encoding="utf-8"))
    contract = opened.get("contract", {})
    canonical = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if (
        opened.get("status") != "opened_locked"
        or opened.get("contract_sha256") != canonical
        or contract.get("output") != str(result_path.resolve())
        or contract.get("predictions") != str(predictions.resolve())
        or contract.get("selection_receipt_sha256") != payload.get("selection_receipt_sha256")
        or contract.get("analysis_contract_sha256") != payload.get("analysis_contract_sha256")
        or contract.get("holdout_lock_sha256") != payload.get("holdout_lock_sha256")
        or contract.get("exposure_audit_sha256") != payload.get("exposure_audit_sha256")
    ):
        raise ValueError(f"invalid one-shot marker contract: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--hulu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite joint decision: {args.output}")
    payloads = {
        "huatuo": json.loads(args.huatuo.read_text(encoding="utf-8")),
        "hulu": json.loads(args.hulu.read_text(encoding="utf-8")),
    }
    for name, payload in payloads.items():
        if (
            payload.get("protocol") != MODEL_PROTOCOL
            or payload.get("mode") != "fresh_confirmation_once"
            or payload.get("status") != "complete"
            or payload.get("model") != name
        ):
            raise ValueError(f"invalid {name} confirmation result")
        validate_attached_artifacts(
            args.huatuo if name == "huatuo" else args.hulu, payload
        )
    equality_fields = (
        "findings",
        "development_record_keys_sha256",
        "development_labels_sha256",
        "confirmation_record_keys_sha256",
        "confirmation_labels_sha256",
        "confirmation_unique_images",
        "analysis_contract_sha256",
        "holdout_lock_sha256",
        "exposure_audit_sha256",
    )
    mismatched = [
        field
        for field in equality_fields
        if payloads["huatuo"].get(field) != payloads["hulu"].get(field)
    ]
    if mismatched:
        raise ValueError(f"two-model data identity mismatch: {mismatched}")
    model_pass = {}
    for name, payload in payloads.items():
        recomputed = recompute_model_gate(payload)
        if bool(payload["model_gate"]["model_pass"]) != recomputed:
            raise ValueError(f"stored and recomputed {name} gate decisions disagree")
        model_pass[name] = recomputed
    both = all(model_pass.values())
    result = {
        "protocol": PROTOCOL,
        "authorizer_code_sha256": sha256(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "inputs": {
            "huatuo": {"path": str(args.huatuo.resolve()), "sha256": sha256(args.huatuo)},
            "hulu": {"path": str(args.hulu.resolve()), "sha256": sha256(args.hulu)},
        },
        "identity": {
            field: payloads["huatuo"][field] for field in equality_fields
        },
        "model_pass": model_pass,
        "both_models_pass": both,
        "decision": (
            "GO_TO_LOCALIZATION_AND_CAUSALITY"
            if both
            else "CLOSE_GLOBAL_SUMMARY_INTERNAL_DECODING_ROUTE"
        ),
        "claim_boundary": (
            "A GO authorizes localization and causal tests only. It does not establish "
            "mitigation or ICLR readiness. A NO-GO closes only global mean/std raw and "
            "post-projector summary probes, not arbitrary spatial patch-token methods."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
