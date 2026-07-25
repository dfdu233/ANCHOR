"""Build a strictly filtered, source-only robust Fourier style bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from corrected_sgta.anchor_dg import (
    VERSION, file_sha256, intensity_statistics, shifted_log_amplitude,
    stable_sha256, standardized_image,
)
from corrected_sgta.train_rule_source_group_adapter import parse_named_paths


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array or JSONL: {path}")
    return rows


def select_images(manifest: Path, root: Path, domain: str, maximum: int, seed: int) -> list[tuple[str, Path]]:
    unique: dict[str, tuple[str, Path]] = {}
    for row in load_rows(manifest):
        if row.get("anchor_cxr_accepted") is False:
            continue
        raw = str(row.get("image", "")).strip()
        if not raw:
            continue
        path = Path(raw)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing source image for {domain}: {path}")
        unique.setdefault(str(path), (str(row.get("source_id") or row.get("id") or path), path))
    selected = sorted(unique.values(), key=lambda item: stable_sha256({"domain": domain, "image": str(item[1]), "seed": seed}))
    return selected[:maximum] if maximum > 0 else selected


def validate_filter_report(path: Path, domains: set[str]) -> dict[str, Any]:
    report = json.loads(path.read_text())
    override = bool(report.get("unverified_source_override"))
    if not override:
        if not report.get("human_audit_complete"):
            raise RuntimeError("filter report is not backed by a complete human audit")
        if int(report.get("human_audit_n", 0)) < 100:
            raise RuntimeError("filter report has fewer than 100 human-audited images")
        if float(report.get("calibrated_precision", 0.0)) < 0.95:
            raise RuntimeError("filter precision is below 95%")
    if not domains.issubset(set(report.get("output_domains", []))):
        raise RuntimeError("filter report does not cover all source manifests")
    return report


def robust_log_center(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot build a center from no images")
    return np.median(np.stack(values), axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build robust source-only chest-radiograph style statistics; never pass target manifests.")
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--filter-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-domain", action="append", default=[])
    parser.add_argument("--max-images-per-source", type=int, default=64)
    parser.add_argument("--min-images-per-source", type=int, default=32)
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = parse_named_paths(args.source, "--source")
    roots = parse_named_paths(args.source_image_root, "--source-image-root")
    if set(manifests) != set(roots):
        raise ValueError("--source and --source-image-root names must match exactly")
    heldout = set(args.heldout_domain)
    if heldout.intersection(manifests):
        raise ValueError(f"held-out domain supplied to bank builder: {sorted(heldout.intersection(manifests))}")
    if len(manifests) < 2:
        raise ValueError("at least two source domains are required")
    filter_report = validate_filter_report(args.filter_report, set(manifests))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".meta.json").exists():
        raise FileExistsError(output)
    domains = tuple(sorted(manifests))
    arrays: dict[str, np.ndarray] = {"domains": np.asarray(domains, dtype=np.str_)}
    selected_meta: dict[str, list[dict[str, str]]] = {}
    for index, domain in enumerate(domains):
        selected = select_images(manifests[domain], roots[domain], domain, args.max_images_per_source, args.seed)
        if len(selected) < args.min_images_per_source:
            raise RuntimeError(f"source {domain!r} retains only {len(selected)} images; minimum is {args.min_images_per_source}")
        logs, means, stds = [], [], []
        selected_meta[domain] = []
        for image_id, path in selected:
            with Image.open(path) as handle:
                image = standardized_image(handle, args.view_size)
            logs.append(shifted_log_amplitude(image))
            mean, std = intensity_statistics(image)
            means.append(mean)
            stds.append(std)
            selected_meta[domain].append({"id": image_id, "image_sha256": file_sha256(path)})
        arrays[f"log_amplitude_{index}"] = robust_log_center(logs)
        arrays[f"rgb_mean_{index}"] = np.median(np.stack(means), axis=0).astype(np.float32)
        arrays[f"rgb_std_{index}"] = np.median(np.stack(stds), axis=0).astype(np.float32)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)
    metadata = {
        "version": VERSION, "scope": "strict_chest_radiograph_source_style_only",
        "center": "pixelwise_median_shifted_log1p_amplitude", "domains": list(domains),
        "heldout_domains": sorted(heldout), "view_size": args.view_size,
        "max_images_per_source": args.max_images_per_source, "min_images_per_source": args.min_images_per_source,
        "seed": args.seed, "filter_report": str(args.filter_report.resolve()),
        "filter_report_sha256": file_sha256(args.filter_report),
        "filter_version": filter_report.get("version"),
        "unverified_source_override": bool(filter_report.get("unverified_source_override")),
        "override_reason": filter_report.get("override_reason"),
        "manifest_sha256": {domain: file_sha256(manifests[domain]) for domain in domains},
        "selected_images": selected_meta, "builder_sha256": file_sha256(Path(__file__).resolve()),
        "core_sha256": file_sha256(Path(__file__).with_name("anchor_dg.py")), "npz_sha256": file_sha256(output),
        "target_data_accessed": False,
    }
    metadata["fingerprint"] = stable_sha256(metadata)
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "fingerprint": metadata["fingerprint"], "domains": list(domains), "counts": {key: len(value) for key, value in selected_meta.items()}}, indent=2))


if __name__ == "__main__":
    main()
