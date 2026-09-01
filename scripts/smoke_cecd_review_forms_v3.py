#!/usr/bin/env python3
"""Real offline-browser smoke for all four CECD v3 reviewer forms.

Only synthetic nonclinical decisions are created inside a temporary directory.
Nothing is written into the frozen source pack or reviewer archives.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from anchor.corrected_sgta.build_cecd_reviewer_deliveries_v1 import (
    CLINICAL_FIELDS,
    LANGUAGE_FIELDS,
    ROLES,
    SOURCE_VERSION,
)
from anchor.medeval.hashing import sha256_file
from anchor.medeval.package_cecd_deliveries_v3 import PROFESSIONAL_ROLE, VERSION, v3_root
from anchor.medeval.store import atomic_write_json


SMOKE_VERSION = "cecd-v3-offline-browser-smoke-v1"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def browser_csv(header: list[str], rows: list[dict[str, str]]) -> bytes:
    def cell(value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    lines = [",".join(cell(field) for field in header)]
    lines.extend(",".join(cell(row.get(field, "")) for field in header) for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def synthetic_rows(role: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    kind = ROLES[role]["kind"]
    values = (
        {
            "support_state_same_supported_refuted_undetermined": "yes",
            "lesion_visibility": "unchanged",
            "clinically_interchangeable": "yes",
            "unable_to_judge": "no",
            "comments": "Synthetic browser smoke only; not a clinical review.",
        }
        if kind == "clinical"
        else {
            "same_clinical_proposition": "yes",
            "same_speech_act": "yes",
            "same_certainty_demand": "yes",
            "same_answer_space": "yes",
            "comments": "Synthetic browser smoke only; not a human language review.",
        }
    )
    return [{**row, **values} for row in rows]


def extract_shell(archive_path: Path, role: str, target: Path, pack: Path) -> Path:
    root = v3_root(role)
    target_root = target / root
    target_root.mkdir(parents=True)
    required = (
        "INSTRUCTIONS.md",
        "REVIEW_FORM.html",
        ROLES[role]["sheet"],
        "REVIEW_SCHEMA.json",
        "IMAGE_SHA256SUMS",
        "DELIVERY_MANIFEST.json",
    )
    wanted = {f"{root}/{filename}": filename for filename in required}
    # Stream only the small metadata prefix. The deterministic archive orders
    # all six shell files before the multi-gigabyte image members.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            filename = wanted.pop(member.name, None)
            if filename is not None:
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(f"cannot read reviewer member: {role}/{filename}")
                (target_root / filename).write_bytes(handle.read())
            if not wanted:
                break
    if wanted:
        raise RuntimeError(f"missing reviewer members for {role}: {sorted(wanted)}")
    if ROLES[role]["kind"] == "clinical":
        os.symlink(pack / "images", target_root / "images", target_is_directory=True)
    return target_root


def smoke_role(browser: Any, archive: Path, role: str, pack: Path, temporary: Path) -> dict[str, Any]:
    root = extract_shell(archive, role, temporary / role, pack)
    header, seed = read_csv(root / ROLES[role]["sheet"])
    completed = synthetic_rows(role, seed)
    completed_path = temporary / f"{role}.synthetic.completed.csv"
    completed_bytes = browser_csv(header, completed)
    completed_path.write_bytes(completed_bytes)
    tampered = [dict(row) for row in completed]
    immutable = "finding" if ROLES[role]["kind"] == "clinical" else "wording_A"
    tampered[0][immutable] += " TAMPERED"
    tampered_path = temporary / f"{role}.tampered.csv"
    tampered_path.write_bytes(browser_csv(header, tampered))

    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    console_errors: list[str] = []
    page_errors: list[str] = []
    remote_requests: list[str] = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "request",
        lambda request: remote_requests.append(request.url)
        if request.url.split(":", 1)[0] not in {"file", "data", "blob"}
        else None,
    )
    page.goto((root / "REVIEW_FORM.html").as_uri(), wait_until="load")
    page.locator("#progress").wait_for()
    images_loaded = True
    if ROLES[role]["kind"] == "clinical":
        images_loaded = bool(
            page.eval_on_selector_all(
                "img.image",
                "els => els.length === 2 && els.every(x => x.complete && x.naturalWidth > 0)",
            )
        )

    page.set_input_files("#import", tampered_path)
    page.wait_for_function("document.querySelector('#message').textContent.includes('Import rejected')")
    tamper_rejected = "immutable" in page.locator("#message").inner_text()
    page.set_input_files("#import", completed_path)
    page.wait_for_function("document.querySelector('#message').textContent.includes('Imported')")
    reviewer_id = f"SYNTHETIC_BROWSER_SMOKE_{role}"
    page.locator("#reviewer-id").fill(reviewer_id)
    for selector in ("#attest-qualified", "#attest-independent", "#attest-blinded"):
        page.locator(selector).check()
    page.reload(wait_until="load")
    autosave_passed = page.locator("#progress").inner_text() == f"{len(seed)}/{len(seed)} complete"
    reviewer_id_persisted = page.locator("#reviewer-id").input_value() == reviewer_id
    for selector in ("#attest-qualified", "#attest-independent", "#attest-blinded"):
        page.locator(selector).check()
    page.locator("#validate").click()
    validation_passed = page.locator("#message").inner_text().startswith("Browser checks pass")

    with page.expect_download() as csv_download:
        page.locator("#export").click()
    csv_path = Path(csv_download.value.path())
    csv_exact = csv_path.read_bytes() == completed_bytes
    with page.expect_download() as attestation_download:
        page.locator("#export-attestation").click()
    attestation_path = Path(attestation_download.value.path())
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation_exact = (
        set(attestation) == {"protocol_id", "review_role", "reviewer"}
        and attestation["protocol_id"] == SOURCE_VERSION
        and attestation["review_role"] == role
        and attestation["reviewer"]["reviewer_id"] == reviewer_id
        and attestation["reviewer"]["professional_role"] == PROFESSIONAL_ROLE[role]
        and attestation["reviewer"]["independent_review"] is True
        and attestation["reviewer"]["blinded_to_sealed_mapping"] is True
    )
    context.close()
    checks = {
        "images_loaded": images_loaded,
        "tamper_rejected": tamper_rejected,
        "autosave_passed": autosave_passed,
        "reviewer_id_persisted": reviewer_id_persisted,
        "validation_passed": validation_passed,
        "csv_exact_roundtrip": csv_exact,
        "attestation_exact": attestation_exact,
        "no_remote_requests": not remote_requests,
        "no_console_errors": not console_errors,
        "no_page_errors": not page_errors,
    }
    return {
        "role": role,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "rows": len(seed),
        "checks": checks,
        "remote_requests": remote_requests,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "passed": all(checks.values()),
        "synthetic_decisions_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.delivery / "delivery_index.json").read_text(encoding="utf-8"))
    by_role = {row["role"]: row for row in index["archives"]}
    with tempfile.TemporaryDirectory(prefix="cecd-v3-browser-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--allow-file-access-from-files"],
            )
            roles = [
                smoke_role(
                    browser,
                    args.delivery / by_role[role]["archive"],
                    role,
                    args.pack,
                    temporary,
                )
                for role in ROLES
            ]
            browser.close()
    result = {
        "version": SMOKE_VERSION,
        "delivery_version": VERSION,
        "delivery_index_sha256": sha256_file(args.delivery / "delivery_index.json"),
        "verification_sha256": sha256_file(args.delivery / "verification.json"),
        "roles": roles,
        "passed": all(row["passed"] for row in roles),
        "clinical_or_language_labels_created": False,
        "attestations_created_for_human_use": False,
        "synthetic_artifacts_retained": False,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
