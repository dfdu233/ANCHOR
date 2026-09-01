#!/usr/bin/env python3
"""Source-separated dataset audit and experiment manifest bootstrap."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
MEDHEVAL_IMAGE_ROOT = REPO_ROOT / "data/medheval/images"
MMEDRAG_ROOT = REPO_ROOT / "data/mmedrag"
CHEXPERT_ROOT = REPO_ROOT / "data/chexpert_subset_report"
CHEXPERT_IMAGE_ROOT = CHEXPERT_ROOT / "processed-v1/images"
RULE_ROOT = REPO_ROOT / "data/rule/test"


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    dataset: str
    task: str
    path: Path
    image_root: Path | None = None
    absolute_images: bool = False
    note: str = ""


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"unsupported dataset shape: {path}")


def first_image_value(row: dict[str, Any]) -> str:
    value = (
        row.get("image")
        or row.get("img_name")
        or row.get("img_id")
        or row.get("image_path")
        or ""
    )
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def reference_value(row: dict[str, Any]) -> str:
    value = row.get("answer")
    if value is None:
        value = row.get("report")
    if value is None:
        value = row.get("gt_ans")
    return "" if value is None else str(value).strip()


def prompt_value(row: dict[str, Any], task: str) -> str:
    value = row.get("question")
    if value is None:
        value = row.get("prompt")
    if value is None and task == "report_generation":
        value = (
            "You are a professional radiologist. Generate a concise medical "
            "report for the image."
        )
    return "" if value is None else str(value).strip()


def candidate_roots(spec: DatasetSpec) -> list[Path]:
    roots = []
    if spec.image_root is not None:
        roots.append(spec.image_root)
    roots.extend(
        [
            MEDHEVAL_IMAGE_ROOT,
            MEDHEVAL_IMAGE_ROOT / "IU-Xray",
            MEDHEVAL_IMAGE_ROOT / "Slake",
            MEDHEVAL_IMAGE_ROOT / "VQA-RAD",
            CHEXPERT_IMAGE_ROOT,
        ]
    )
    for env_name in (
        "ANCHOR_HARVARD_IMAGE_ROOT",
        "ANCHOR_PMC_OA_IMAGE_ROOT",
        "ANCHOR_QUILT_IMAGE_ROOT",
        "ANCHOR_CHEXPERT_IMAGE_ROOT",
    ):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    seen: set[Path] = set()
    ordered = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered


def is_materialized_image(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(128)
    except OSError:
        return False
    return not head.startswith(b"version https://git-lfs.github.com/spec/")


def resolve_image(spec: DatasetSpec, row: dict[str, Any]) -> Path | None:
    raw = first_image_value(row)
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute() and path.is_file() and is_materialized_image(path):
        return path
    names = [raw]
    if path.is_absolute():
        names.append(path.name)
    for root in candidate_roots(spec):
        for name in names:
            candidate = root / name
            if candidate.is_file() and is_materialized_image(candidate):
                return candidate
    return None


def audit_spec(spec: DatasetSpec, max_missing: int) -> dict[str, Any]:
    rows = load_rows(spec.path)
    missing: list[str] = []
    available = 0
    prompts = 0
    refs = 0
    unique_images: set[str] = set()
    for row in rows:
        if prompt_value(row, spec.task):
            prompts += 1
        if reference_value(row):
            refs += 1
        image = resolve_image(spec, row)
        if image is None:
            if len(missing) < max_missing:
                missing.append(first_image_value(row))
        else:
            available += 1
            unique_images.add(str(image))
    return {
        **asdict(spec),
        "path": str(spec.path),
        "image_root": str(spec.image_root) if spec.image_root is not None else None,
        "total": len(rows),
        "available_images": available,
        "missing_images": len(rows) - available,
        "unique_available_images": len(unique_images),
        "rows_with_prompt": prompts,
        "rows_with_reference": refs,
        "first_missing_images": missing,
        "ready": bool(rows) and available == len(rows) and refs == len(rows),
    }


def dataset_specs() -> list[DatasetSpec]:
    specs: list[DatasetSpec] = [
        DatasetSpec("rule", "mimic", "vqa_binary", RULE_ROOT / "mimic_test.jsonl", MEDHEVAL_IMAGE_ROOT),
        DatasetSpec("rule", "iuxray", "vqa_binary", RULE_ROOT / "iuxray_test.jsonl", MEDHEVAL_IMAGE_ROOT / "IU-Xray"),
        DatasetSpec(
            "rule",
            "harvard",
            "vqa_binary",
            RULE_ROOT / "harvard_test.jsonl",
            Path(os.environ.get("ANCHOR_HARVARD_IMAGE_ROOT", "/root/autodl-tmp/source_data/FairVLMed/extracted/Test")),
        ),
        DatasetSpec(
            "chexpert_report",
            "chexpert_subset",
            "report_generation",
            CHEXPERT_ROOT / "processed-v1/anchor_report_manifest.json",
            CHEXPERT_IMAGE_ROOT,
            absolute_images=True,
            note="chexpert_subset_unverified",
        ),
    ]
    for task, suffix in (("vqa_binary", "vqa"), ("report_generation", "report")):
        for name in ("mimic", "iuxray", "harvard", "pmc-oa", "quilt-1m"):
            ext = "jsonl" if task == "vqa_binary" else "json"
            root = None
            if name == "mimic":
                root = MEDHEVAL_IMAGE_ROOT
            elif name == "iuxray":
                root = MEDHEVAL_IMAGE_ROOT / "IU-Xray"
            elif name == "harvard" and os.environ.get("ANCHOR_HARVARD_IMAGE_ROOT"):
                root = Path(os.environ["ANCHOR_HARVARD_IMAGE_ROOT"])
            specs.append(
                DatasetSpec(
                    "mmedrag",
                    name,
                    task,
                    MMEDRAG_ROOT / f"test/{suffix}/{name}_test.{ext}",
                    root,
                )
            )
    med = REPO_ROOT / "data/medheval/benchmark_data"
    specs.extend(
        [
            DatasetSpec("medheval", "visual_oe_mimic", "open_vqa", med / "Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "knowledge_oe_mimic", "open_vqa", med / "Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "context_mimic", "vqa_binary", med / "Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "knowledge_ce_mimic", "vqa_binary", med / "Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "cxr_vishal", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "mm_vishal", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "mimic_fine_grained_ce", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json", MEDHEVAL_IMAGE_ROOT),
            DatasetSpec("medheval", "iuxray_fine_grained_ce", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json", MEDHEVAL_IMAGE_ROOT / "IU-Xray"),
            DatasetSpec("medheval", "slake_ce", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json", MEDHEVAL_IMAGE_ROOT / "Slake"),
            DatasetSpec("medheval", "vqa_rad_ce", "vqa_binary", med / "Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json", MEDHEVAL_IMAGE_ROOT / "VQA-RAD"),
        ]
    )
    return [spec for spec in specs if spec.path.exists()]


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def methods_for(task: str) -> list[str]:
    baseline = ["greedy", "beam"] if task != "open_vqa" else ["greedy"]
    mitigation = ["DoLa", "VCD", "OPERA", "AVISC", "M3ID", "DAMRO", "PAI"]
    if task in {"open_vqa", "report_generation", "vqa_binary"}:
        return baseline + mitigation
    return baseline


def write_manifest(audits: list[dict[str, Any]], output: Path) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = git_commit()
    rows = []
    for audit in audits:
        if not audit["ready"]:
            rows.append(
                {
                    "source": audit["source"],
                    "dataset": audit["dataset"],
                    "task": audit["task"],
                    "method": "all",
                    "status": "blocked",
                    "created_at": now,
                    "git_commit": commit,
                    "output_dir": None,
                    "command": None,
                    "failure_reason": "dataset audit not ready",
                    "audit": audit,
                }
            )
            continue
        for method in methods_for(audit["task"]):
            rows.append(
                {
                    "source": audit["source"],
                    "dataset": audit["dataset"],
                    "task": audit["task"],
                    "method": method,
                    "status": "planned",
                    "created_at": now,
                    "git_commit": commit,
                    "output_dir": None,
                    "command": None,
                    "failure_reason": None,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "corrected_runs/high_efficiency")
    parser.add_argument("--max-missing", type=int, default=20)
    args = parser.parse_args()
    audits = [audit_spec(spec, args.max_missing) for spec in dataset_specs()]
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "dataset_audit.json"
    audit_path.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": audits}, indent=2))
    write_manifest(audits, args.out / "manifest.jsonl")
    print(json.dumps({"audit": str(audit_path), "manifest": str(args.out / "manifest.jsonl"), "datasets": len(audits), "ready": sum(1 for row in audits if row["ready"])}, indent=2))


if __name__ == "__main__":
    main()
