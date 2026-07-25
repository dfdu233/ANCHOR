"""BiomedCLIP filtering and auditable calibration for ANCHOR-DG sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from corrected_sgta.anchor_dg import file_sha256, stable_sha256
from corrected_sgta.train_rule_source_group_adapter import parse_named_paths

VERSION = "anchor-dg-cxr-filter-v2"
DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/BiomedCLIP")
PROMPTS = (
    "a frontal chest radiograph",
    "a lateral chest radiograph",
    "a skull or head radiograph",
    "a pelvic or hip radiograph",
    "an abdominal radiograph",
    "a CT scan of the brain",
    "an MRI scan of the brain",
    "an abdominal CT scan",
    "a skeletal extremity radiograph",
    "a retinal fundus photograph",
    "an ultrasound image",
)
FORBIDDEN_TARGET_TOKENS = ("mimic", "chexpert", "padchest", "harvard")


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array or JSONL: {path}")
    return rows


def resolve_unique_images(manifest: Path, root: Path, domain: str) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in load_rows(manifest):
        raw = str(row.get("image", "")).strip()
        if not raw:
            continue
        path = Path(raw)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        unique.setdefault(str(path), {"id": str(row.get("source_id") or row.get("id") or path), "domain": domain, "image": str(path)})
    return sorted(unique.values(), key=lambda item: stable_sha256({"domain": domain, "image": item["image"]}))


def load_biomedclip(root: Path):
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.factory import _MODEL_CONFIGS
    config = json.loads((root / "open_clip_config.json").read_text())
    name = "anchor_biomedclip_local"
    _MODEL_CONFIGS[name] = config["model_cfg"]
    model, _, preprocess = create_model_and_transforms(
        name, pretrained=str(root / "open_clip_pytorch_model.bin"),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    return model, preprocess, get_tokenizer(name), config


def validate_source_only_names(names: list[str]) -> None:
    text = " ".join(names).lower()
    if any(token in text for token in FORBIDDEN_TARGET_TOKENS):
        raise RuntimeError("locked/target dataset token found in filter inputs")


def select_precision_threshold(
    labelled: list[tuple[float, bool]], min_precision: float
) -> tuple[float, float, int]:
    candidates = []
    for threshold in sorted({margin for margin, _ in labelled}):
        selected = [label for margin, label in labelled if margin > threshold]
        if selected:
            precision = float(np.mean(selected))
            if precision >= min_precision:
                candidates.append((len(selected), precision, threshold))
    if not candidates:
        raise RuntimeError("no threshold reaches requested human-audit precision")
    count, precision, threshold = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return float(threshold), float(precision), int(count)


def human_label(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"cxr", "chest", "chest_xray", "yes", "1", "true"}:
        return True
    if normalized in {"non_cxr", "other", "no", "0", "false"}:
        return False
    return None


def score_command(args: argparse.Namespace) -> None:
    manifests = parse_named_paths(args.source, "--source")
    roots = parse_named_paths(args.source_image_root, "--source-image-root")
    if set(manifests) != set(roots):
        raise ValueError("source manifest/root names must match")
    validate_source_only_names([str(path) for path in manifests.values()] + list(manifests))
    images = [item for domain in sorted(manifests) for item in resolve_unique_images(manifests[domain], roots[domain], domain)]
    model, preprocess, tokenizer, config = load_biomedclip(args.model_root)
    device = torch.device(args.device)
    model = model.to(device).eval()
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer(list(PROMPTS), context_length=256).to(device), normalize=True)
    records = []
    for start in range(0, len(images), args.batch_size):
        batch = images[start:start + args.batch_size]
        tensors = []
        for item in batch:
            with Image.open(item["image"]) as handle:
                tensors.append(preprocess(handle.convert("RGB")))
        with torch.inference_mode():
            features = model.encode_image(torch.stack(tensors).to(device), normalize=True)
            scores = (features @ text_features.T).float().cpu().numpy()
        for item, values in zip(batch, scores):
            runner_up = float(np.max(values[1:]))
            records.append({
                **item, "image_sha256": file_sha256(Path(item["image"])),
                "scores": {prompt: float(value) for prompt, value in zip(PROMPTS, values)},
                "cxr_margin": float(values[0] - runner_up),
                "predicted_category": PROMPTS[int(np.argmax(values))],
            })
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    if args.scores.exists() or args.audit.exists():
        raise FileExistsError("score or audit output already exists")
    args.scores.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    domains = sorted({row["domain"] for row in records})
    quotas = {domain: args.audit_size // len(domains) for domain in domains}
    for domain in domains[: args.audit_size % len(domains)]:
        quotas[domain] += 1
    chosen = []
    for domain in domains:
        ordered = sorted(
            (row for row in records if row["domain"] == domain),
            key=lambda row: (row["cxr_margin"], stable_sha256(row["id"])),
        )
        positions = np.linspace(0, len(ordered) - 1, min(quotas[domain], len(ordered))).round().astype(int)
        chosen.extend(ordered[position] for position in positions)
    chosen.sort(key=lambda row: stable_sha256({"audit": row["id"]}))
    audit = [{"audit_id": f"A{index:03d}", "id": row["id"], "image": row["image"], "human_label": None} for index, row in enumerate(chosen)]
    args.audit.write_text(json.dumps({"version": VERSION, "blinded_to_model_scores": True, "items": audit}, indent=2) + "\n")
    metadata = {
        "version": VERSION, "scores": str(args.scores.resolve()), "audit": str(args.audit.resolve()),
        "source_sha256": {name: file_sha256(path) for name, path in manifests.items()},
        "n_images": len(records), "audit_n": len(audit), "prompts": list(PROMPTS),
        "model_config_sha256": file_sha256(args.model_root / "open_clip_config.json"),
        "model_weights_sha256": file_sha256(args.model_root / "open_clip_pytorch_model.bin"),
        "preprocess_config": config.get("preprocess_cfg"), "target_data_accessed": False,
    }
    args.scores.with_suffix(args.scores.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_images": len(records), "audit_n": len(audit), "scores": str(args.scores), "audit": str(args.audit)}, indent=2))


def calibrate_command(args: argparse.Namespace) -> None:
    records = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in records}
    audit = json.loads(args.audit.read_text())
    labelled = []
    for item in audit.get("items", []):
        label = human_label(item.get("human_label"))
        if label is None:
            continue
        if item["id"] not in by_id:
            raise RuntimeError(f"audit id absent from score file: {item['id']}")
        labelled.append((float(by_id[item["id"]]["cxr_margin"]), label))
    if len(labelled) < args.required_audit_n:
        raise RuntimeError(f"only {len(labelled)} human labels; {args.required_audit_n} required")
    point_threshold, precision, _ = select_precision_threshold(labelled, args.min_precision)
    rng = np.random.default_rng(args.seed)
    bootstrap_thresholds = []
    for _ in range(args.bootstrap_replicates):
        indexes = rng.integers(0, len(labelled), size=len(labelled))
        resample = [labelled[index] for index in indexes]
        try:
            sampled_threshold, _, _ = select_precision_threshold(resample, args.min_precision)
            bootstrap_thresholds.append(sampled_threshold)
        except RuntimeError:
            continue
    if not bootstrap_thresholds:
        raise RuntimeError("audit threshold is unstable in every bootstrap resample")
    threshold = float(max(point_threshold, np.quantile(bootstrap_thresholds, 0.95)))
    result = {
        "version": VERSION, "human_audit_complete": True, "human_audit_n": len(labelled),
        "calibrated_precision": precision, "min_precision": args.min_precision,
        "point_threshold": point_threshold, "threshold": threshold,
        "boundary_width": threshold - point_threshold,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_valid_fraction": len(bootstrap_thresholds) / args.bootstrap_replicates,
        "selection_rule": "cxr_margin > bootstrap_95pct_threshold",
        "scores_sha256": file_sha256(args.scores), "audit_sha256": file_sha256(args.audit),
        "target_data_accessed": False,
    }
    result["fingerprint"] = stable_sha256(result)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


def apply_command(args: argparse.Namespace) -> None:
    if args.assume_all_chest:
        if args.calibration is not None or args.scores is not None:
            raise ValueError("--assume-all-chest cannot be combined with scores/calibration")
        calibration = {
            "human_audit_complete": False, "human_audit_n": 0,
            "calibrated_precision": None, "threshold": None,
            "unverified_source_override": True,
            "override_reason": args.override_reason,
        }
        scores: dict[str, dict[str, Any]] = {}
        threshold = None
    else:
        if args.calibration is None or args.scores is None:
            raise ValueError("strict apply requires --scores and --calibration")
        calibration = json.loads(args.calibration.read_text())
        if not calibration.get("human_audit_complete") or int(calibration.get("human_audit_n", 0)) < 100 or float(calibration.get("calibrated_precision", 0)) < 0.95:
            raise RuntimeError("calibration does not satisfy the preregistered human-audit gate")
        scores = {row["image"]: row for row in (json.loads(line) for line in args.scores.read_text().splitlines() if line.strip())}
        threshold = float(calibration["threshold"])
    manifests = parse_named_paths(args.source, "--source")
    roots = parse_named_paths(args.source_image_root, "--source-image-root")
    trusted = set(args.trusted_source)
    if set(manifests) != set(roots) or not trusted.issubset(manifests):
        raise ValueError("source/root/trusted names are inconsistent")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, row_counts, output_paths = {}, {}, {}
    for domain in sorted(manifests):
        source_rows = load_rows(manifests[domain])
        accepted_paths = set()
        for row in source_rows:
            raw = str(row.get("image", "")).strip()
            if not raw:
                continue
            path = Path(raw)
            path = (path if path.is_absolute() else roots[domain] / path).resolve()
            score = scores.get(str(path))
            keep = args.assume_all_chest or domain in trusted or (score is not None and score["predicted_category"] == PROMPTS[0] and float(score["cxr_margin"]) > threshold)
            if keep:
                accepted_paths.add(str(path))
        if len(accepted_paths) < args.min_images_per_source:
            raise RuntimeError(f"source {domain!r} retains {len(accepted_paths)} unique images; minimum is {args.min_images_per_source}")
        accepted = []
        for row in source_rows:
            raw = str(row.get("image", "")).strip()
            if not raw:
                continue
            path = Path(raw)
            path = (path if path.is_absolute() else roots[domain] / path).resolve()
            if str(path) in accepted_paths:
                copied = dict(row)
                copied["image"] = str(path)
                copied["anchor_cxr_accepted"] = True
                copied["anchor_cxr_filter_version"] = VERSION
                copied["unverified_source_override"] = bool(args.assume_all_chest)
                accepted.append(copied)
        output = args.output_dir / f"{domain}.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(accepted, indent=2) + "\n")
        counts[domain] = len(accepted_paths)
        row_counts[domain] = len(accepted)
        output_paths[domain] = str(output.resolve())
    report = {
        **calibration, "version": VERSION, "output_domains": sorted(manifests),
        "trusted_sources": sorted(trusted), "counts": counts, "row_counts": row_counts,
        "output_manifests": output_paths,
        "scores_sha256": file_sha256(args.scores) if args.scores else None,
        "calibration_sha256": file_sha256(args.calibration) if args.calibration else None,
        "target_data_accessed": False,
    }
    report["fingerprint"] = stable_sha256(report)
    report_path = args.output_dir / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), "counts": counts, "row_counts": row_counts, "unverified_source_override": report.get("unverified_source_override", False), "fingerprint": report["fingerprint"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict source-only chest-radiograph filtering with a human-audited BiomedCLIP threshold.")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    score.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    score.add_argument("--scores", type=Path, required=True)
    score.add_argument("--audit", type=Path, required=True)
    score.add_argument("--audit-size", type=int, default=100)
    score.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--device", default="cuda")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--scores", type=Path, required=True)
    calibrate.add_argument("--audit", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--required-audit-n", type=int, default=100)
    calibrate.add_argument("--min-precision", type=float, default=0.95)
    calibrate.add_argument("--bootstrap-replicates", type=int, default=2000)
    calibrate.add_argument("--seed", type=int, default=42)
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    apply_parser.add_argument("--source-image-root", action="append", required=True, metavar="NAME=PATH")
    apply_parser.add_argument("--scores", type=Path)
    apply_parser.add_argument("--calibration", type=Path)
    apply_parser.add_argument("--trusted-source", action="append", default=[])
    apply_parser.add_argument("--assume-all-chest", action="store_true")
    apply_parser.add_argument("--override-reason", default="user-authorized exploratory run without human audit")
    apply_parser.add_argument("--output-dir", type=Path, required=True)
    apply_parser.add_argument("--min-images-per-source", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"score": score_command, "calibrate": calibrate_command, "apply": apply_command}[args.command](args)


if __name__ == "__main__":
    main()
