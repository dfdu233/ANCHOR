#!/usr/bin/env python3
"""Freeze four valid listing returns into a human-adjudication handoff.

This transition copies human inputs byte-for-byte and creates blank-final
adjudication templates.  It never opens the sealed transform mapping, resolves
a disagreement, decides equivalence, or creates a scientific admission receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from corrected_sgta.build_vindr_cecd_listing_admission_pack_v1 import (
    CLINICAL_DECISION_FIELDS,
    PROMPT_DECISION_FIELDS,
    ROLES,
)
from corrected_sgta.prepare_vindr_reader_manifest import sha256_file
from corrected_sgta.validate_vindr_cecd_listing_admission_returns_v1 import (
    validate_all,
)
from corrected_sgta.verify_vindr_cecd_listing_admission_pack_v1 import verify


VERSION = "vindr-cecd-listing-human-adjudication-handoff-v1"
CLINICAL_FINAL_FIELDS = tuple(
    f"adjudicated_{field}" for field in CLINICAL_DECISION_FIELDS if field != "comments"
) + ("adjudication_rationale",)
PROMPT_FINAL_FIELDS = tuple(
    f"adjudicated_{field}" for field in PROMPT_DECISION_FIELDS if field != "comments"
) + ("adjudication_rationale",)


class HandoffError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise HandoffError(f"required regular file is absent: {resolved}")
    name = str(resolved.relative_to(relative_to.resolve())) if relative_to else str(resolved)
    return {"path": name, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise HandoffError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _adjudication_template(
    *, left_path: Path, right_path: Path, decision_fields: Sequence[str], id_field: str
) -> tuple[list[str], list[dict[str, str]]]:
    left_header, left_rows = _read_csv(left_path)
    right_header, right_rows = _read_csv(right_path)
    if left_header != right_header or len(left_rows) != len(right_rows):
        raise HandoffError("paired return sheets are not structurally aligned")
    immutable = [field for field in left_header if field not in decision_fields]
    final_fields = CLINICAL_FINAL_FIELDS if id_field == "pair_id" else PROMPT_FINAL_FIELDS
    fields = (
        immutable
        + [f"review_A__{field}" for field in decision_fields]
        + [f"review_B__{field}" for field in decision_fields]
        + list(final_fields)
    )
    output: list[dict[str, str]] = []
    for left, right in zip(left_rows, right_rows):
        if left[id_field] != right[id_field] or any(left[field] != right[field] for field in immutable):
            raise HandoffError("paired return row identities or immutable fields differ")
        row = {field: left[field] for field in immutable}
        row.update({f"review_A__{field}": left[field] for field in decision_fields})
        row.update({f"review_B__{field}": right[field] for field in decision_fields})
        row.update({field: "" for field in final_fields})
        output.append(row)
    return fields, output


def _expected_source_records(
    completed: Mapping[str, Path], attestations: Mapping[str, Path]
) -> dict[str, Any]:
    return {
        role: {
            "completed": file_record(completed[role]),
            "attestation": file_record(attestations[role]),
        }
        for role in ROLES
    }


def _validate_existing(
    output_dir: Path, *, pack_dir: Path, source_records: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = output_dir / "handoff.json"
    if not manifest_path.is_file():
        raise HandoffError("partial write-once handoff directory requires audit")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if (
        payload.get("schema_version") != VERSION
        or payload.get("status") != "ready_for_human_adjudication"
        or payload.get("source_returns") != source_records
        or payload.get("pack_manifest", {}).get("sha256") != sha256_file(pack_dir / "manifest.json")
        or fingerprint != canonical_sha256(body)
        or payload.get("admission_receipt_created") is not False
    ):
        raise HandoffError("existing handoff does not match current validated returns")
    inventory = payload.get("handoff_inventory")
    if not isinstance(inventory, list):
        raise HandoffError("handoff inventory is missing")
    expected_paths = {str(row.get("path")) for row in inventory} | {"handoff.json"}
    actual_paths = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise HandoffError("write-once handoff file closure drift")
    for record in inventory:
        path = output_dir / str(record["path"])
        if file_record(path, relative_to=output_dir) != record:
            raise HandoffError(f"handoff inventory hash drift: {path}")
    return payload


def prepare_handoff(
    *,
    pack_dir: Path,
    completed: Mapping[str, Path],
    attestations: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    integrity = verify(pack_dir)
    if integrity.get("passed") is not True:
        raise HandoffError("listing admission pack integrity failed")
    validation = validate_all(
        pack_dir=pack_dir, completed=dict(completed), attestations=dict(attestations)
    )
    source_records = _expected_source_records(completed, attestations)
    if output_dir.exists():
        return _validate_existing(
            output_dir, pack_dir=pack_dir, source_records=source_records
        )

    staging = output_dir.with_name(f".{output_dir.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise HandoffError(f"stale staging directory requires audit: {staging}")
    staging.mkdir(parents=True)
    try:
        frozen = staging / "frozen_returns"
        frozen.mkdir()
        for role in ROLES:
            shutil.copyfile(completed[role], frozen / f"{role}.completed.csv")
            shutil.copyfile(attestations[role], frozen / f"{role}.attestation.json")

        clinical_fields, clinical_rows = _adjudication_template(
            left_path=frozen / "clinical_reviewer_1.completed.csv",
            right_path=frozen / "clinical_reviewer_2.completed.csv",
            decision_fields=CLINICAL_DECISION_FIELDS,
            id_field="pair_id",
        )
        prompt_fields, prompt_rows = _adjudication_template(
            left_path=frozen / "clinical_template_reviewer.completed.csv",
            right_path=frozen / "language_reviewer.completed.csv",
            decision_fields=PROMPT_DECISION_FIELDS,
            id_field="item_id",
        )
        _write_csv(staging / "clinical_adjudication.template.csv", clinical_fields, clinical_rows)
        _write_csv(staging / "prompt_adjudication.template.csv", prompt_fields, prompt_rows)
        (staging / "adjudicator.attestation.template.json").write_text(
            json.dumps(
                {
                    "schema_version": "vindr-cecd-listing-scientific-admission-v1",
                    "handoff_fingerprint": "",
                    "human_admission_decision": "",
                    "adjudicators": [
                        {
                            "scope": "clinical",
                            "adjudicator_id": "",
                            "professional_role": "physician",
                            "independent_adjudication": True,
                            "blinded_to_model_outputs": True,
                            "completed_at_utc": "",
                        },
                        {
                            "scope": "prompt",
                            "adjudicator_id": "",
                            "professional_role": "",
                            "independent_adjudication": True,
                            "blinded_to_model_outputs": True,
                            "completed_at_utc": "",
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        instructions = (
            "# Human adjudication required\n\n"
            "The four role-isolated returns passed structural and attestation validation. "
            "No clinical or prompt equivalence decision has been made. A genuinely independent "
            "human adjudicator must fill every `adjudicated_*` field and a nonempty rationale, "
            "return completed files under new names, and complete the blank signed/time-zoned "
            "attestation with an explicit `admit` or `reject` decision. "
            "Do not edit review_A/review_B or immutable stimulus fields. This package is not an "
            "admission receipt and authorizes no model or GPU work.\n"
        )
        (staging / "ADJUDICATION_REQUIRED.md").write_text(instructions, encoding="utf-8")
        inventory = [
            file_record(path, relative_to=staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        payload = {
            "schema_version": VERSION,
            "status": "ready_for_human_adjudication",
            "pack_manifest": file_record(pack_dir / "manifest.json"),
            "source_returns": source_records,
            "validated_return_summary": validation,
            "handoff_inventory": inventory,
            "review_decisions_copied_not_synthesized": True,
            "clinical_equivalence_decided": False,
            "prompt_equivalence_decided": False,
            "adjudication_performed": False,
            "admission_receipt_created": False,
            "model_scoring_authorized": False,
            "gpu_authorized": False,
        }
        payload["fingerprint"] = canonical_sha256(payload)
        (staging / "handoff.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
        return payload
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _role_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role in output:
            raise HandoffError(f"expected unique role=/path: {value!r}")
        output[role] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--completed", action="append", default=[])
    parser.add_argument("--attestation", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_handoff(
        pack_dir=args.pack_dir,
        completed=_role_paths(args.completed),
        attestations=_role_paths(args.attestation),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
