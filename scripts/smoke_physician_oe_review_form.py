#!/usr/bin/env python3
"""Run a real, offline Chromium smoke test on a physician-OE delivery archive.

The script never writes into the frozen review source or delivery archive.  It
uses synthetic in-browser annotations solely to exercise import, validation,
and export, then discards the temporary extraction directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


VERSION = "anchor-physician-oe-review-form-browser-smoke-v1"


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
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            destination = output.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
    if len(roots) != 1:
        raise RuntimeError(f"expected one archive root, found {sorted(roots)}")
    return output / next(iter(roots))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def immutable_projection(rows: list[dict[str, Any]]) -> list[Any]:
    return [
        [
            row["bundle_id"],
            row["group_id"],
            row["review_order"],
            row["review_phase"],
            row["reviewer_slot"],
            row["image"],
            row["question"],
            row["benchmark_reference"],
            [
                [candidate["answer_id"], candidate["answer_text"]]
                for candidate in row["candidate_answers"]
            ],
        ]
        for row in rows
    ]


def synthetic_completed_rows(seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Round-trip through JSON so the browser fixture cannot mutate the seed.
    rows = json.loads(json.dumps(seed))
    for row in rows:
        row["reference_annotation"] = {
            "visual_observability": "unobservable",
            "benchmark_reference_correctness": "indeterminate",
            "required_answer_claims": [],
            "notes": "",
        }
        for candidate in row["candidate_answers"]:
            candidate["annotation"] = {
                "direct_answer_correctness": "correct",
                "direct_answer_state": "unobservable",
                "atomic_claims": [],
                "no_clinical_claims": True,
                "omitted_required_claim_ids": [],
                "overall_clinically_harmful": "no",
                "reviewer_confidence": 3,
                "rationale": "",
            }
    return rows


def run_smoke(archive_path: Path) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "playwright is required; run with `uv run --with playwright`"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="anchor-physician-oe-ui-") as temporary:
        root = extract_regular_files(archive_path, Path(temporary))
        form = root / "REVIEW_FORM.html"
        review_files = sorted(root.glob("reviewer_*.blinded.jsonl"))
        if not form.is_file() or len(review_files) != 1:
            raise RuntimeError("archive lacks exactly one form and blinded review JSONL")
        seed = load_jsonl(review_files[0])
        completed = synthetic_completed_rows(seed)
        completed_path = root / "synthetic.completed.jsonl"
        completed_path.write_text(
            "".join(json.dumps(row) + "\n" for row in completed), encoding="utf-8"
        )
        tampered = json.loads(json.dumps(seed))
        tampered[0]["question"] += " [tampered]"
        tampered_path = root / "synthetic.tampered.jsonl"
        tampered_path.write_text(
            "".join(json.dumps(row) + "\n" for row in tampered), encoding="utf-8"
        )

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
            if page.locator("#position").inner_text().split(" · ")[0] != f"1/{len(seed)}":
                raise RuntimeError("navigation count differs from frozen JSONL")
            if page.locator("section.answer").count() != len(seed[0]["candidate_answers"]):
                raise RuntimeError("candidate count differs from frozen JSONL")

            # Exercise form-generated objects and persistence before replacing
            # them with a synthetically complete import fixture.
            page.get_by_role("button", name="Add required claim").click()
            page.get_by_role("button", name="Add atomic claim").first.click()
            if page.locator("section.answer input[type=checkbox]").first.is_checked():
                raise RuntimeError("adding an atomic claim left no_clinical_claims checked")
            storage_size = page.evaluate("localStorage.getItem(storageKey).length")
            if not isinstance(storage_size, int) or storage_size <= 0:
                raise RuntimeError("autosave did not create localStorage state")
            page.reload(wait_until="load")
            if page.get_by_text("Required claim 1", exact=True).count() != 1:
                raise RuntimeError("required claim did not survive browser reload")
            if page.get_by_text("Atomic claim 1", exact=True).count() != 1:
                raise RuntimeError("atomic claim did not survive browser reload")

            page.locator("#import").set_input_files(str(tampered_path))
            page.wait_for_function(
                "document.getElementById('message').textContent.startsWith('Import rejected:')"
            )
            if "immutable reviewer-visible content changed" not in page.locator(
                "#message"
            ).inner_text():
                raise RuntimeError("tampered immutable content was not rejected")

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
            exported_path = root / download.suggested_filename
            download.save_as(exported_path)
            exported = load_jsonl(exported_path)
            if immutable_projection(exported) != immutable_projection(seed):
                raise RuntimeError("export changed immutable reviewer-visible content")
            if exported != completed:
                raise RuntimeError("exported rows differ from imported completed rows")
            browser_version = browser.version
            browser.close()

        if console_errors or page_errors or network_urls:
            raise RuntimeError(
                "browser emitted errors or external requests: "
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
            "groups_rendered": len(seed),
            "answer_units_exported": sum(
                len(row["candidate_answers"]) for row in exported
            ),
            "image_loaded": True,
            "autosave_reload_passed": True,
            "tampered_import_rejected": True,
            "completed_import_passed": True,
            "browser_validation_passed": True,
            "export_roundtrip_exact": True,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
