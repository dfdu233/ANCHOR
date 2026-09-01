#!/usr/bin/env python3
"""Offline Chromium acceptance test for a Specificity Ratchet v3 archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from anchor.corrected_sgta.merge_specificity_ratchet_reviews_v1 import REVIEW_FIELDS


VERSION = "specificity-ratchet-review-form-browser-smoke-v1"


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
            if (
                not member.isfile()
                or not parts
                or member.name.startswith("/")
                or ".." in parts
            ):
                raise RuntimeError(f"unsafe archive member: {member.name}")
            roots.add(parts[0])
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            destination = output.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read())
    if len(roots) != 1:
        raise RuntimeError(f"expected one archive root, found {sorted(roots)}")
    return output / next(iter(roots))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV header missing")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def immutable_projection(
    rows: list[dict[str, str]], header: list[str], review_fields: list[str]
) -> list[list[str]]:
    mutable = {"reviewer_id", "rationale", *review_fields}
    immutable = [field for field in header if field not in mutable]
    return [[row[field] for field in immutable] for row in rows]


def synthetic_completed_rows(
    rows: list[dict[str, str]], schema: dict[str, Any], reviewer_id: str
) -> list[dict[str, str]]:
    output = json.loads(json.dumps(rows))
    preferred = {
        "edge_entailment_admitted": "yes",
        "parent_visual_support": "supported",
        "child_visual_support": "undetermined",
        "increment_observability": "observable_on_supplied_image",
        "logical_scope_preserved": "yes",
        "reviewer_confidence": "high",
        "clinical_usefulness_if_backed_off": "unchanged",
        "clinically_harmful_if_wrong": "no",
    }
    for row in output:
        row["reviewer_id"] = reviewer_id
        for field, values in schema["fields"].items():
            value = preferred[field]
            if value not in values:
                raise ValueError(f"synthetic value absent from schema: {field}={value}")
            row[field] = value
        row["rationale"] = "Synthetic browser smoke annotation; not a clinician label."
    return output


def run_smoke(archive_path: Path) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("run with `uv run --with playwright`") from exc

    with tempfile.TemporaryDirectory(prefix="specificity-ratchet-ui-") as temporary:
        root = extract_regular_files(archive_path, Path(temporary))
        form = root / "REVIEW_FORM.html"
        csv_files = sorted(root.glob("annotations.reviewer_*.csv"))
        if not form.is_file() or len(csv_files) != 1:
            raise RuntimeError("archive must contain one form and one reviewer CSV")
        schema = json.loads((root / "annotation_schema.json").read_text(encoding="utf-8"))
        header, seed = read_csv(csv_files[0])
        role = "1" if "reviewer_1" in csv_files[0].name else "2"
        completed = synthetic_completed_rows(
            seed, schema, f"SYNTHETIC-NONCLINICAL-REVIEWER-{role}"
        )
        completed_path = root / "synthetic.completed.csv"
        write_csv(completed_path, header, completed)
        tampered = json.loads(json.dumps(seed))
        tampered[0]["question"] += " [tampered]"
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
            if groups != len({row["case_id"] for row in seed}) or edges != len(seed):
                raise RuntimeError("rendered group/edge counts differ from CSV")

            # Exercise visible controls and local persistence.
            page.locator("#reviewer-id").fill("TEMP-NONCLINICAL-REVIEWER")
            first_edge = page.locator("section.edge").first
            for index, field in enumerate(REVIEW_FIELDS):
                first_edge.locator("select").nth(index).select_option(
                    schema["fields"][field][0]
                )
            first_edge.locator("textarea").fill("Temporary nonclinical UI smoke.")
            storage_size = page.evaluate("localStorage.getItem(storageKey).length")
            if not isinstance(storage_size, int) or storage_size <= 0:
                raise RuntimeError("autosave did not write localStorage")
            page.reload(wait_until="load")
            if page.locator("#reviewer-id").input_value() != "TEMP-NONCLINICAL-REVIEWER":
                raise RuntimeError("reviewer ID did not survive reload")
            if first_edge.locator("textarea").input_value() != "Temporary nonclinical UI smoke.":
                raise RuntimeError("rationale did not survive reload")

            page.locator("#import").set_input_files(str(tampered_path))
            page.wait_for_function(
                "document.getElementById('message').textContent.startsWith('Import rejected:')"
            )
            if "immutable reviewer-visible content changed" not in page.locator(
                "#message"
            ).inner_text():
                raise RuntimeError("tampered question was not rejected")

            page.locator("#import").set_input_files(str(completed_path))
            page.wait_for_function(
                "document.getElementById('message').textContent.startsWith('Imported')"
            )
            page.locator("#validate").click()
            if not page.locator("#message").inner_text().startswith("Browser checks pass"):
                raise RuntimeError(page.locator("#message").inner_text())
            with page.expect_download() as download_info:
                page.locator("#export").click()
            download = download_info.value
            export_path = root / download.suggested_filename
            download.save_as(export_path)
            exported_header, exported = read_csv(export_path)
            if exported_header != header or exported != completed:
                raise RuntimeError("exported CSV differs from completed import")
            review_fields = list(schema["fields"])
            if immutable_projection(exported, header, review_fields) != immutable_projection(
                seed, header, review_fields
            ):
                raise RuntimeError("export changed immutable reviewer-visible content")
            page.locator("#attest-physician").check()
            page.locator("#attest-independent").check()
            page.locator("#attest-blinded").check()
            with page.expect_download() as attestation_info:
                page.locator("#export-attestation").click()
            attestation_download = attestation_info.value
            attestation_path = root / attestation_download.suggested_filename
            attestation_download.save_as(attestation_path)
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            reviewer = attestation.get("reviewer", {})
            if (
                attestation.get("protocol_id") != schema.get("protocol_id")
                or reviewer.get("reviewer_id") != completed[0]["reviewer_id"]
                or reviewer.get("role") != "physician"
                or reviewer.get("independent_review") is not True
                or reviewer.get("blinded_to_private_provenance") is not True
                or not reviewer.get("completed_at_utc")
            ):
                raise RuntimeError("reviewer attestation export is incomplete or mismatched")
            browser_version = browser.version
            browser.close()

        if console_errors or page_errors or network_urls:
            raise RuntimeError(
                "browser emitted errors/external requests: "
                f"console={console_errors}, page={page_errors}, network={network_urls}"
            )
        return {
            "version": VERSION,
            "time": datetime.now(timezone.utc).isoformat(),
            "archive": str(archive_path.resolve()),
            "archive_sha256": sha256_file(archive_path),
            "form_sha256": sha256_file(form),
            "browser": f"Chromium {browser_version}",
            "offline_context": True,
            "groups_rendered": groups,
            "edges_exported": edges,
            "image_loaded": True,
            "autosave_reload_passed": True,
            "tampered_import_rejected": True,
            "completed_import_passed": True,
            "browser_validation_passed": True,
            "export_roundtrip_exact": True,
            "explicit_reviewer_attestation_exported": True,
            "external_network_requests": network_urls,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "synthetic_annotations_are_not_clinician_labels": True,
            "passed": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke(args.archive)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
