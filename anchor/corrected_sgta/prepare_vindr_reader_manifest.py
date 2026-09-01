#!/usr/bin/env python3
"""Build a balanced VinDr-CXR reader-vote manifest from the official CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from corrected_sgta.clinical_claims import VERSION as CLAIM_VERSION
from corrected_sgta.clinical_claims import normalize_term, reader_state


VERSION = "vindr-reader-vote-manifest-v4"
DEFAULT_BASE_URL = "https://physionet.org/files/vindr-cxr/1.0.0/train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode()).hexdigest()


def experiment_split(image_id: str, seed: int, dev_fraction: float) -> str:
    """Assign all claims from one image to the same deterministic split."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must lie strictly between zero and one")
    digest = stable_key(seed, "experiment-split", str(image_id))
    uniform = int(digest[:16], 16) / float(16**16)
    return "dev" if uniform < dev_fraction else "test"


def find_column(fieldnames: Iterable[str], candidates: set[str]) -> str:
    for name in fieldnames:
        if normalize_term(name) in candidates:
            return name
    raise ValueError(f"missing required column; expected one of {sorted(candidates)}")


def read_votes(
    path: Path,
) -> tuple[dict[str, dict[str, dict[str, int]]], list[str], str, str]:
    """Read the complete image × finding × pseudonymous-reader vote tensor.

    Keeping ``rad_ID`` is necessary for reader random effects and leave-reader-
    out sensitivity analyses.  An aggregate 0/3--3/3 count alone cannot
    distinguish image ambiguity from a systematically liberal or conservative
    reader.
    """

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("VinDr CSV has no header")
        image_col = find_column(reader.fieldnames, {"image_id", "imageid"})
        reader_col = find_column(reader.fieldnames, {"rad_id", "reader_id", "radiologist_id"})
        excluded = {image_col, reader_col}
        findings = [name for name in reader.fieldnames if name not in excluded]
        votes: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        seen_readers: dict[str, set[str]] = defaultdict(set)
        for line_number, row in enumerate(reader, start=2):
            image_id = str(row[image_col]).strip()
            rad_id = str(row[reader_col]).strip()
            if not image_id or not rad_id:
                raise ValueError(f"empty image/radiologist ID on CSV line {line_number}")
            if rad_id in seen_readers[image_id]:
                raise ValueError(f"duplicate reader {rad_id!r} for image {image_id!r}")
            seen_readers[image_id].add(rad_id)
            for finding in findings:
                raw = str(row[finding]).strip()
                if raw not in {"0", "1", "0.0", "1.0"}:
                    raise ValueError(
                        f"non-binary label {raw!r} for {finding!r} on line {line_number}"
                    )
                votes[image_id][finding][rad_id] = int(float(raw))
    counts = Counter(len(readers) for readers in seen_readers.values())
    if counts != {3: len(seen_readers)}:
        raise ValueError(f"expected exactly 3 independent readers per image, got {dict(counts)}")
    return votes, findings, image_col, reader_col


def load_ontology_findings(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("findings", payload)
    if not isinstance(source, dict) or not source:
        raise ValueError("ontology must be a non-empty object or contain 'findings'")
    findings = {normalize_term(str(value)) for value in source}
    if len(findings) != len(source):
        raise ValueError("ontology contains duplicate normalized finding names")
    return findings


def select_ontology_columns(
    source_findings: Iterable[str], ontology_findings: set[str]
) -> tuple[list[str], list[str]]:
    by_normalized: dict[str, str] = {}
    for source in source_findings:
        normalized = normalize_term(source)
        if normalized in by_normalized:
            raise ValueError(f"duplicate normalized VinDr finding column: {normalized}")
        by_normalized[normalized] = source
    missing = ontology_findings - set(by_normalized)
    if missing:
        raise ValueError(f"ontology findings absent from VinDr CSV: {sorted(missing)}")
    selected = [
        source for source in source_findings if normalize_term(source) in ontology_findings
    ]
    excluded = [source for source in source_findings if source not in selected]
    if not selected:
        raise ValueError("ontology selects no VinDr finding columns")
    return selected, excluded


def build_records(
    votes: dict[str, dict[str, dict[str, int]]],
    findings: list[str],
    base_url: str,
) -> tuple[list[dict[str, object]], dict[str, Counter[str]]]:
    records = []
    statistics: dict[str, Counter[str]] = {finding: Counter() for finding in findings}
    for image_id in sorted(votes):
        for finding in findings:
            values = votes[image_id][finding]
            reader_votes = [
                {"rad_id": rad_id, "vote": int(vote)}
                for rad_id, vote in sorted(values.items())
            ]
            positive = sum(int(item["vote"]) for item in reader_votes)
            count = len(reader_votes)
            vote_bin = f"{positive}/{count}"
            statistics[finding][vote_bin] += 1
            records.append(
                {
                    "dataset": "vindr-cxr-1.0.0",
                    "reference_source": "vindr_reader_votes",
                    "evidence_grade": "A",
                    "formal_reference": True,
                    "split": "train",
                    "image_id": image_id,
                    "finding": normalize_term(finding),
                    "finding_source_name": finding,
                    "positive_votes": positive,
                    "reader_count": count,
                    "reader_ids": [str(item["rad_id"]) for item in reader_votes],
                    "reader_votes": reader_votes,
                    "reader_support": positive / count,
                    "reader_state": reader_state(positive, count),
                    "dicom_relpath": f"train/{image_id}.dicom",
                    "dicom_url": f"{base_url.rstrip('/')}/{image_id}.dicom",
                }
            )
    return records, statistics


def reader_effect_summary(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Summarize reader coverage without estimating effects on the full data.

    This is a readiness audit.  Formal reader random effects are fit only on
    the locked development split after image selection.
    """

    reader_images: dict[str, set[str]] = defaultdict(set)
    per_finding: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for row in records:
        image_id = str(row["image_id"])
        finding = str(row["finding"])
        reader_votes = row.get("reader_votes")
        if not isinstance(reader_votes, list) or len(reader_votes) != int(
            row["reader_count"]
        ):
            raise ValueError("reader-level votes missing from formal VinDr record")
        seen: set[str] = set()
        for item in reader_votes:
            if not isinstance(item, dict):
                raise ValueError("reader_votes entries must be objects")
            rad_id = str(item.get("rad_id", ""))
            vote = int(item.get("vote", -1))
            if not rad_id or rad_id in seen or vote not in {0, 1}:
                raise ValueError("invalid or duplicate reader-level vote")
            seen.add(rad_id)
            reader_images[rad_id].add(image_id)
            per_finding[finding][rad_id][0] += vote
            per_finding[finding][rad_id][1] += 1
        if sum(int(item["vote"]) for item in reader_votes) != int(
            row["positive_votes"]
        ):
            raise ValueError("aggregate positive_votes disagree with reader_votes")
    image_counts = {reader: len(images) for reader, images in sorted(reader_images.items())}
    return {
        "reader_identity_preserved": True,
        "reader_id_semantics": "pseudonymous official VinDr rad_ID",
        "unique_readers": len(reader_images),
        "images_per_reader": image_counts,
        "minimum_images_per_reader": min(image_counts.values()) if image_counts else 0,
        "maximum_images_per_reader": max(image_counts.values()) if image_counts else 0,
        "per_finding_reader_counts": {
            finding: {
                reader: {"positive": values[0], "total": values[1]}
                for reader, values in sorted(readers.items())
            }
            for finding, readers in sorted(per_finding.items())
        },
        "formal_modeling_requirement": (
            "fit reader and finding effects on dev only; report leave-reader-out "
            "sensitivity where reader overlap permits"
        ),
    }


def balanced_subset(
    records: list[dict[str, object]],
    eligible: set[str],
    per_bin: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        finding = str(row["finding_source_name"])
        if finding in eligible:
            grouped[(finding, int(row["positive_votes"]))].append(row)
    chosen = []
    for finding in sorted(eligible):
        for positive in range(4):
            candidates = sorted(
                grouped[(finding, positive)],
                key=lambda row: stable_key(seed, finding, str(positive), str(row["image_id"])),
            )
            chosen.extend(candidates[:per_bin])
    return sorted(
        chosen,
        key=lambda row: (str(row["finding"]), int(row["positive_votes"]), str(row["image_id"])),
    )


def oe_listing_relevance(positive_votes: int, reader_count: int) -> str:
    """Freeze reporting obligation for a list-visible-abnormalities prompt."""

    if reader_count <= 0 or positive_votes not in range(reader_count + 1):
        raise ValueError("invalid reader votes for OE relevance")
    if positive_votes == reader_count:
        return "required"
    if positive_votes == 0:
        return "out_of_scope"
    return "optional"


def build_oe_listing_records(
    records: Iterable[dict[str, object]],
    selected_image_ids: set[str],
    eligible_findings: set[str],
    seed: int,
    dev_fraction: float,
) -> list[dict[str, object]]:
    """Expand selected images to a complete eligible-finding OE universe."""

    output = []
    seen: set[tuple[str, str]] = set()
    for source in records:
        image_id = str(source["image_id"])
        finding_source_name = str(source["finding_source_name"])
        if image_id not in selected_image_ids or finding_source_name not in eligible_findings:
            continue
        key = (image_id, str(source["finding"]))
        if key in seen:
            raise ValueError(f"duplicate OE claim row: {key}")
        seen.add(key)
        row = dict(source)
        row.update(
            {
                "experiment_split": experiment_split(
                    image_id, seed, dev_fraction
                ),
                "reference_contract_version": CLAIM_VERSION,
                "reference_observability": "image_grounded",
                "reference_relevance": oe_listing_relevance(
                    int(row["positive_votes"]), int(row["reader_count"])
                ),
                "oe_task_contract": "vindr-list-visible-abnormalities-v1",
            }
        )
        output.append(row)
    expected = len(selected_image_ids) * len(eligible_findings)
    if len(output) != expected:
        raise ValueError(
            f"incomplete OE universe: expected {expected} rows, found {len(output)}"
        )
    return sorted(
        output,
        key=lambda row: (str(row["image_id"]), str(row["finding"])),
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--min-per-bin", type=int, default=100)
    parser.add_argument("--samples-per-bin", type=int, default=100)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_per_bin <= 0 or args.samples_per_bin <= 0:
        raise ValueError("bin sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    votes, source_findings, image_col, reader_col = read_votes(args.labels_csv)
    if args.ontology is None:
        findings = source_findings
        excluded_source_findings: list[str] = []
    else:
        findings, excluded_source_findings = select_ontology_columns(
            source_findings, load_ontology_findings(args.ontology)
        )
    records, statistics = build_records(votes, findings, args.base_url)
    reader_audit = reader_effect_summary(records)
    required_bins = tuple(f"{value}/3" for value in range(4))
    eligible = {
        finding
        for finding, counts in statistics.items()
        if all(counts[name] >= args.min_per_bin for name in required_bins)
    }
    if not eligible:
        details = {
            finding: {name: counts[name] for name in required_bins}
            for finding, counts in statistics.items()
        }
        atomic_text(args.output_dir / "reader_vote_counts.json", json.dumps(details, indent=2) + "\n")
        raise RuntimeError(
            "no finding satisfies min-per-bin; counts were written for an auditable threshold decision"
        )
    per_bin = min(
        args.samples_per_bin,
        min(statistics[finding][name] for finding in eligible for name in required_bins),
    )
    selected = balanced_subset(records, eligible, per_bin, args.seed)
    for row in selected:
        row["experiment_split"] = experiment_split(
            str(row["image_id"]), args.seed, args.dev_fraction
        )
    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    atomic_text(args.output_dir / "reader_vote_manifest.jsonl", manifest_text)
    selected_image_ids = {str(row["image_id"]) for row in selected}
    oe_records = build_oe_listing_records(
        records,
        selected_image_ids,
        eligible,
        args.seed,
        args.dev_fraction,
    )
    atomic_text(
        args.output_dir / "oe_listing_reference.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in oe_records),
    )
    urls = sorted({str(row["dicom_url"]) for row in selected})
    atomic_text(args.output_dir / "image_urls.txt", "\n".join(urls) + "\n")
    summary = {
        "version": VERSION,
        "claim_contract_version": CLAIM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(args.labels_csv.resolve()),
        "source_csv_sha256": sha256_file(args.labels_csv),
        "source_columns": {"image": image_col, "reader": reader_col},
        "image_count": len(votes),
        "finding_count": len(findings),
        "source_finding_count": len(source_findings),
        "excluded_source_findings": excluded_source_findings,
        "ontology": str(args.ontology.resolve()) if args.ontology else None,
        "ontology_sha256": sha256_file(args.ontology) if args.ontology else None,
        "eligible_findings": sorted(eligible),
        "required_bins": list(required_bins),
        "minimum_per_bin": args.min_per_bin,
        "selected_per_bin": per_bin,
        "selected_claim_rows": len(selected),
        "selected_unique_images": len(urls),
        "oe_listing": {
            "task_contract": "vindr-list-visible-abnormalities-v1",
            "reference_manifest": "oe_listing_reference.jsonl",
            "complete_fixed_ontology": True,
            "claim_rows": len(oe_records),
            "unique_images": len(selected_image_ids),
            "findings_per_image": len(eligible),
            "reference_relevance": dict(
                Counter(str(row["reference_relevance"]) for row in oe_records)
            ),
        },
        "experiment_split": {
            "method": "image-level deterministic SHA256; no image crosses dev/test",
            "dev_fraction": args.dev_fraction,
            "claim_rows": dict(Counter(str(row["experiment_split"]) for row in selected)),
            "unique_images": dict(
                Counter(
                    experiment_split(image_id, args.seed, args.dev_fraction)
                    for image_id in {str(row["image_id"]) for row in selected}
                )
            ),
        },
        "reader_vote_counts": {
            finding: {name: statistics[finding][name] for name in required_bins}
            for finding in findings
        },
        "reader_effect_readiness": reader_audit,
        "selection": "stable SHA256 ordering within finding and reader-vote bin",
        "seed": args.seed,
        "command": " ".join(sys.argv),
    }
    summary["fingerprint"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()
    ).hexdigest()
    atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
