#!/usr/bin/env python3
"""Offline Chromium acceptance for a blinded adjudicator archive.

All filled values are explicitly synthetic UI fixtures in a disposable
directory. They are never written to the clinical-return inbox.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import REVIEW_FIELDS
from anchor.corrected_sgta.validate_specificity_ratchet_adjudication_v1 import FINAL_FIELDS


VERSION = "specificity-ratchet-adjudication-browser-smoke-v1"
MUTABLE = {
    *[f"final_{field}" for field in FINAL_FIELDS],
    "adjudicator_id",
    "disagreement_reason",
    "adjudication_rationale",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_regular_files(archive_path: Path, output: Path) -> Path:
    roots: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or not parts or member.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            roots.add(parts[0])
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            destination = output.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
    if len(roots) != 1:
        raise RuntimeError(f"expected one archive root, found {sorted(roots)}")
    return output / next(iter(roots))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def completed_rows(rows):
    output = json.loads(json.dumps(rows))
    values = {
        "edge_entailment_admitted": "yes",
        "parent_visual_support": "supported",
        "child_visual_support": "supported",
        "increment_observability": "observable_on_supplied_image",
        "logical_scope_preserved": "yes",
        "clinical_usefulness_if_backed_off": "unchanged",
        "clinically_harmful_if_wrong": "no",
    }
    for row in output:
        row["adjudicator_id"] = "SYNTHETIC-NONCLINICAL-ADJUDICATOR"
        for field, value in values.items():
            row[f"final_{field}"] = value
        disagreement = any(
            row[f"r1_{field}"] != row[f"r2_{field}"] for field in REVIEW_FIELDS
        )
        row["disagreement_reason"] = (
            "Synthetic disagreement resolution for browser testing."
            if disagreement
            else ""
        )
        row["adjudication_rationale"] = (
            "Synthetic browser smoke adjudication; not a clinician label."
        )
    return output


def immutable(rows, header):
    fields = [field for field in header if field not in MUTABLE]
    return [[row[field] for field in fields] for row in rows]


def run_smoke(archive_path: Path) -> dict:
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory(prefix="specificity-adjudication-ui-") as name:
        root = extract_regular_files(archive_path, Path(name))
        form = root / "ADJUDICATION_FORM.html"
        merged = root / "adjudication.with_reviews.csv"
        header, seed = read_csv(merged)
        completed = completed_rows(seed)
        completed_path = root / "synthetic.completed.csv"
        write_csv(completed_path, header, completed)
        tampered = json.loads(json.dumps(completed))
        tampered[0]["r1_rationale"] += " tampered"
        tampered_path = root / "synthetic.tampered.csv"
        write_csv(tampered_path, header, tampered)
        console_errors: list[str] = []
        page_errors: list[str] = []
        network_urls: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = browser.new_context(accept_downloads=True)
            context.set_offline(True)
            page = context.new_page()
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "request",
                lambda request: network_urls.append(request.url)
                if not request.url.startswith(("file:", "blob:"))
                else None,
            )
            page.goto(form.resolve().as_uri(), wait_until="load")
            page.wait_for_function(
                "document.getElementById('image').complete && "
                "document.getElementById('image').naturalWidth > 0"
            )
            groups = int(page.evaluate("groups().length"))
            edges = int(page.evaluate("rows.length"))
            page.locator("#adjudicator-id").fill("TEMP-NONCLINICAL-ADJUDICATOR")
            first = page.locator("section.edge").first
            first.locator("select").first.select_option(index=1)
            first.locator("textarea").last.fill("Temporary nonclinical UI smoke.")
            if page.evaluate("localStorage.getItem(storageKey).length") <= 0:
                raise RuntimeError("autosave did not write localStorage")
            page.reload(wait_until="load")
            if page.locator("#adjudicator-id").input_value() != "TEMP-NONCLINICAL-ADJUDICATOR":
                raise RuntimeError("adjudicator ID did not survive reload")
            page.locator("#import").set_input_files(str(tampered_path))
            page.wait_for_function(
                "document.getElementById('message').textContent.startsWith('Import rejected:')"
            )
            if "immutable copied review content changed" not in page.locator("#message").inner_text():
                raise RuntimeError("tampered reviewer rationale was not rejected")
            page.locator("#import").set_input_files(str(completed_path))
            page.wait_for_function(
                "document.getElementById('message').textContent.startsWith('Imported')"
            )
            page.locator("#validate").click()
            if not page.locator("#message").inner_text().startswith("Browser checks pass"):
                raise RuntimeError(page.locator("#message").inner_text())
            with page.expect_download() as info:
                page.locator("#export").click()
            exported_path = root / info.value.suggested_filename
            info.value.save_as(exported_path)
            exported_header, exported = read_csv(exported_path)
            if exported_header != header or exported != completed:
                raise RuntimeError("completed adjudication CSV did not round-trip")
            if immutable(exported, header) != immutable(seed, header):
                raise RuntimeError("export changed immutable reviewer content")
            page.locator("#attest-physician").check()
            page.locator("#attest-blinded").check()
            with page.expect_download() as info:
                page.locator("#export-attestation").click()
            attestation_path = root / info.value.suggested_filename
            info.value.save_as(attestation_path)
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            record = attestation.get("adjudicator", {})
            if (
                attestation.get("protocol_id") != "specificity-ratchet-physician-pack-v2"
                or record.get("adjudicator_id") != completed[0]["adjudicator_id"]
                or record.get("role") != "physician"
                or record.get("blinded_to_private_provenance") is not True
                or not record.get("completed_at_utc")
            ):
                raise RuntimeError("adjudicator attestation export is invalid")
            browser.close()
        result = {
            "version": VERSION,
            "time": datetime.now(timezone.utc).isoformat(),
            "archive": str(archive_path.resolve()),
            "archive_sha256": sha256_file(archive_path),
            "groups": groups,
            "edges": edges,
            "image_loaded": True,
            "autosave_reload_passed": True,
            "immutable_tamper_rejected": True,
            "completed_round_trip_exact": True,
            "attestation_export_verified": True,
            "synthetic_values_are_clinical_labels": False,
            "network_urls": network_urls,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "passed": not network_urls and not console_errors and not page_errors,
        }
        if not result["passed"]:
            raise RuntimeError(json.dumps(result, indent=2))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke(args.archive)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
