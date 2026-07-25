#!/usr/bin/env python3
"""Build a sealed source-confirmation split disjoint from a base manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corrected_sgta.build_rule_source_manifest import (
    BuildConfig,
    load_iu,
    load_slake,
    load_vqarad,
    sha256_bytes,
    sha256_file,
    source_stats,
    stable_digest,
    write_json_and_jsonl,
)

VERSION = "rule-source-confirm-manifest-v1"
DOMAINS = ("rule_iuxray", "slake_xray", "vqa_rad_train")


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return value


def independence_unit(row: dict[str, Any]) -> str:
    domain = str(row["source_domain"])
    if domain in {"rule_iuxray", "slake_xray"}:
        return str(row["source_id"]).split("/", 1)[0]
    return str(row["image_sha256"])


def select_confirm_rows(
    rows: list[dict[str, Any]],
    excluded_hashes: set[str],
    excluded_units: set[tuple[str, str]],
    images_per_domain: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if images_per_domain <= 0:
        raise ValueError("images_per_domain must be positive")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {} for domain in DOMAINS
    }
    hash_domains: dict[str, set[str]] = {}
    for row in rows:
        domain = str(row["source_domain"])
        image_hash = str(row["image_sha256"])
        hash_domains.setdefault(image_hash, set()).add(domain)
        if image_hash in excluded_hashes:
            continue
        if (domain, independence_unit(row)) in excluded_units:
            continue
        grouped[domain].setdefault(image_hash, []).append(row)
    collisions = {h: ds for h, ds in hash_domains.items() if len(ds) > 1}
    if collisions:
        raise RuntimeError(f"cross-domain RGB hash collision: {next(iter(collisions))}")
    selected: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        hashes = sorted(
            grouped[domain],
            key=lambda h: stable_digest(seed, "confirm-image", domain, h),
        )
        if len(hashes) < images_per_domain:
            raise RuntimeError(
                f"domain {domain} has only {len(hashes)} independent unused images"
            )
        chosen = []
        for image_hash in hashes[:images_per_domain]:
            candidates = sorted(
                grouped[domain][image_hash],
                key=lambda row: stable_digest(
                    seed,
                    "confirm-qa",
                    row["id"],
                    row["conversations"][0]["value"],
                ),
            )
            chosen.append(candidates[0])
        selected[domain] = sorted(chosen, key=lambda row: row["id"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--images-per-domain", type=int, default=48)
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    base = json.loads(args.base_manifest.read_text())
    if base.get("version") != "rule-source-manifest-v2":
        raise ValueError("unsupported base manifest")
    config = base["config"]
    build_config = BuildConfig(
        iu_json=Path(config["iu_json"]),
        iu_image_root=Path(config["iu_image_root"]),
        slake_root=Path(config["slake_root"]),
        vqarad_parquet=Path(config["vqarad_parquet"]),
        locked_test=Path(config["locked_test"]),
        locked_image_root=Path(config["locked_image_root"]),
        output_dir=args.output_dir,
        seed=args.seed,
        dev_fraction=0.2,
        max_images_per_domain=0,
        qas_per_image=1,
    )
    base_rows = []
    for split in ("train", "dev"):
        base_rows.extend(load_rows(Path(base["outputs"][split]["json"])))
    excluded_hashes = {str(row["image_sha256"]) for row in base_rows}
    excluded_units = {
        (str(row["source_domain"]), independence_unit(row)) for row in base_rows
    }
    all_rows = load_iu(build_config) + load_slake(build_config) + load_vqarad(build_config)
    selected = select_confirm_rows(
        all_rows, excluded_hashes, excluded_units, args.images_per_domain, args.seed
    )
    selected_hashes = {
        str(row["image_sha256"]) for rows in selected.values() for row in rows
    }
    if selected_hashes & excluded_hashes:
        raise RuntimeError("confirmation split overlaps base train/dev")
    outputs = {
        domain: write_json_and_jsonl(args.output_dir / f"confirm.{domain}", rows)
        for domain, rows in selected.items()
    }
    protocol = {
        "version": VERSION,
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "base_manifest_fingerprint": base["fingerprint"],
        "seed": args.seed,
        "images_per_domain": args.images_per_domain,
        "selection": "stable hash over unused independence units; one QA per image",
        "records": [
            {
                "domain": domain,
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "answer": row["conversations"][1]["value"],
            }
            for domain, rows in sorted(selected.items())
            for row in rows
        ],
    }
    fingerprint = sha256_bytes(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    )
    payload = {
        **protocol,
        "fingerprint": fingerprint,
        "base_excluded_images": len(excluded_hashes),
        "target_audit_inherited": base["locked_test"],
        "target_file_opened": False,
        "available_after_exclusion": {
            domain: len({
                row["image_sha256"] for row in all_rows
                if row["source_domain"] == domain
                and row["image_sha256"] not in excluded_hashes
                and (domain, independence_unit(row)) not in excluded_units
            })
            for domain in DOMAINS
        },
        "selected": {
            "total": source_stats([row for rows in selected.values() for row in rows]),
            "domains": {domain: source_stats(rows) for domain, rows in selected.items()},
        },
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "fingerprint": fingerprint, "selected": payload["selected"], "target_file_opened": False}, indent=2))


if __name__ == "__main__":
    main()
