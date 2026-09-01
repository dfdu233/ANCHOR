#!/usr/bin/env python3
"""Join frozen T3 outputs to a preselected private physician-review manifest.

The generation side remains label-redacted.  This module is the first allowed
reference join and refuses to run unless the outcome-blind operational audit
has authorized a physician pack.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json


VERSION = "llava-mitigation-t3-physician-inputs-v1"
REFERENCE_KEYS = {"answer", "gt_ans", "reference", "references"}


def _load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object answer at {path}:{number}")
            rows.append(row)
    return rows


def _qid(row: dict[str, Any]) -> str:
    for key in ("question_id", "qid", "id", "sample_id"):
        if key in row:
            return str(row[key])
    raise ValueError("answer row has no question identifier")


def _contains_reference_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(REFERENCE_KEYS & set(value)) or any(
            _contains_reference_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_reference_key(child) for child in value)
    return False


def _write_jsonl_bound(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"write-once selected answer collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def prepare(
    *,
    run_root: Path,
    audit_path: Path,
    execution_contract_path: Path,
    selection_prereg_path: Path,
    selected_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = json.loads(execution_contract_path.read_text(encoding="utf-8"))
    prereg = json.loads(selection_prereg_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected manifest must be a nonempty JSON list")

    if (
        audit.get("protocol_version") != "llava-mitigation-t3-generation-audit-v1"
        or audit.get("all_operational_gates_passed") is not True
        or audit.get("physician_pack_authorized") is not True
        or audit.get("clinical_efficacy_authorized") is not False
    ):
        raise RuntimeError("generation audit did not authorize the physician pack")
    if audit.get("execution_contract_sha256") != sha256_file(execution_contract_path):
        raise RuntimeError("generation audit is not bound to the execution contract")
    if audit.get("run_root") != str(run_root.resolve()):
        raise RuntimeError("generation audit is bound to another run root")

    prereg_without_fingerprint = dict(prereg)
    observed_fingerprint = prereg_without_fingerprint.pop("fingerprint", None)
    if observed_fingerprint != sha256_json(prereg_without_fingerprint):
        raise RuntimeError("selection preregistration fingerprint mismatch")
    if prereg.get("model_outputs_read") is not False:
        raise RuntimeError("selection was not frozen outcome-blind")
    if (
        prereg.get("execution_contract_sha256")
        != sha256_file(execution_contract_path)
        or list(map(str, prereg.get("methods", [])))
        != list(map(str, contract.get("methods", [])))
    ):
        raise RuntimeError("selection preregistration is not bound to the method contract")
    if prereg.get("selected_manifest_sha256") != sha256_file(selected_manifest_path):
        raise RuntimeError("selected manifest differs from the frozen preregistration")

    selected_qids = [str(row.get("qid", row.get("id", ""))) for row in selected]
    if (
        not all(selected_qids)
        or len(selected_qids) != len(set(selected_qids))
        or selected_qids != list(map(str, prereg.get("selected_qids", [])))
        or len(selected_qids) != int(prereg.get("selected_groups", -1))
    ):
        raise RuntimeError("selected qid order/count differs from the preregistration")
    image_ids = [str(row.get("image_sha256", "")) for row in selected]
    if not all(image_ids) or len(image_ids) != len(set(image_ids)):
        raise RuntimeError("physician selection is not image-disjoint")
    if any(not str(row.get("answer", "")).strip() for row in selected):
        raise RuntimeError("private selected manifest lacks benchmark references")

    methods = list(map(str, contract.get("methods", [])))
    records = audit.get("method_records")
    if not isinstance(records, list) or {str(row.get("method")) for row in records} != set(methods):
        raise RuntimeError("generation audit method closure differs from execution contract")
    by_method = {str(row["method"]): row for row in records}
    outputs: dict[str, Any] = {}
    selected_set = set(selected_qids)
    for method in methods:
        record = by_method[method]
        if record.get("eligible") is not True:
            raise RuntimeError(f"method is not operationally eligible: {method}")
        paths = [Path(value) for value in record.get("answer_paths", [])]
        hashes = list(record.get("answer_sha256", []))
        if not paths or len(paths) != len(hashes):
            raise RuntimeError(f"missing audited answer files for {method}")
        for path, expected_hash in zip(paths, hashes):
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"audited answer source drifted for {method}: {path}")
            try:
                path.resolve().relative_to(run_root.resolve())
            except ValueError as error:
                raise RuntimeError(f"answer source escaped run root: {path}") from error
        rows = _load_jsonl(paths)
        if any(_contains_reference_key(row) for row in rows):
            raise RuntimeError(f"reference field leaked into raw output for {method}")
        row_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            qid = _qid(row)
            if qid in row_map:
                raise RuntimeError(f"duplicate raw qid for {method}: {qid}")
            row_map[qid] = row
        if not selected_set.issubset(row_map):
            missing = sorted(selected_set - set(row_map))
            raise RuntimeError(f"selected qids absent for {method}: {missing[:5]}")
        subset = [row_map[qid] for qid in selected_qids]
        output = output_dir / f"{method}.answers.jsonl"
        _write_jsonl_bound(output, subset)
        outputs[method] = {
            "source_paths": [str(path.resolve()) for path in paths],
            "source_sha256": hashes,
            "selected_output": str(output.resolve()),
            "selected_output_sha256": sha256_file(output),
            "selected_rows": len(subset),
            "reference_fields_absent": True,
        }

    result = {
        "protocol_version": VERSION,
        "first_authorized_reference_join": True,
        "generation_remained_label_redacted": True,
        "selection_frozen_before_outputs": True,
        "clinical_labels_inspected": False,
        "clinical_efficacy_authorized": False,
        "run_root": str(run_root.resolve()),
        "generation_audit": str(audit_path.resolve()),
        "generation_audit_sha256": sha256_file(audit_path),
        "execution_contract": str(execution_contract_path.resolve()),
        "execution_contract_sha256": sha256_file(execution_contract_path),
        "selection_prereg": str(selection_prereg_path.resolve()),
        "selection_prereg_sha256": sha256_file(selection_prereg_path),
        "selected_manifest": str(selected_manifest_path.resolve()),
        "selected_manifest_sha256": sha256_file(selected_manifest_path),
        "selected_groups": len(selected),
        "unique_images": len(set(image_ids)),
        "methods": methods,
        "planned_model_assignments": len(selected) * len(methods),
        "outputs": outputs,
    }
    result["fingerprint"] = sha256_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generation-audit", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--selection-prereg", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        run_root=args.run_root.resolve(),
        audit_path=args.generation_audit.resolve(),
        execution_contract_path=args.execution_contract.resolve(),
        selection_prereg_path=args.selection_prereg.resolve(),
        selected_manifest_path=args.selected_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    if args.provenance.exists():
        prior = json.loads(args.provenance.read_text(encoding="utf-8"))
        if prior != result:
            raise FileExistsError(f"write-once provenance collision: {args.provenance}")
    else:
        atomic_write_json(args.provenance, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
