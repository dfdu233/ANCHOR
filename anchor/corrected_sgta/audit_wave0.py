#!/usr/bin/env python3
"""Wave-0 readiness audit: CE interfaces, local judge, and report metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ce-cache", type=Path, nargs="*", default=())
    parser.add_argument(
        "--qwen-snapshot",
        type=Path,
        default=Path("/root/autodl-tmp/hf_hub_cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("/root/autodl-tmp/MedHEval/code/evaluation/report_eval"),
    )
    return parser.parse_args()


def qwen_readiness(snapshot: Path) -> dict:
    index_path = snapshot / "model.safetensors.index.json"
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    shards = []
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing.extend(name for name in shards if not (snapshot / name).is_file())
    else:
        missing.append(index_path.name)
    return {
        "snapshot": str(snapshot),
        "n_expected_weight_shards": len(shards),
        "missing_files": sorted(set(missing)),
        "ready": not missing and bool(shards),
    }


def report_readiness(root: Path) -> dict:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("RaTEScore", "radgraph", "bert_score")
    }
    chexbert_candidates = list(root.rglob("*.ckpt")) + list(root.rglob("*.pth"))
    radgraph_candidates = [
        path for path in root.rglob("*")
        if path.is_file() and "radgraph" in path.name.lower() and path.suffix in {".pt", ".tar", ".gz"}
    ]
    entrypoints = [
        root / "run_all_metrics.py",
        root / "run_chair.py",
        root / "CXRMetric" / "run_extraction.py",
    ]
    return {
        "root": str(root),
        "python_modules": modules,
        "entrypoints": {str(path): path.is_file() for path in entrypoints},
        "chexbert_checkpoint_candidates": [str(path) for path in chexbert_candidates],
        "radgraph_checkpoint_candidates": [str(path) for path in radgraph_candidates],
        "ready": (
            all(modules.values())
            and bool(chexbert_candidates)
            and bool(radgraph_candidates)
            and all(path.is_file() for path in entrypoints)
        ),
    }


def ce_interface(cache: Path) -> dict:
    metadata_path = cache.with_suffix(cache.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"unsupported protocol in {cache}")
    records = list(iter_successes(cache, metadata["fingerprint"]))
    surface, sequence, sequence_gt, gt, disagreement = [], [], [], [], []
    decoded_text_available = 0
    decoded_parsed = 0
    for row in records:
        surface_prediction = int(np.argmax(np.asarray(row["style_logits"])[0]))
        surface.append(surface_prediction)
        gt.append(int(row["gt_index"]))
        values = row.get("style_sequence_nll")
        if values and values[0] is not None:
            sequence_prediction = int(np.argmin(np.asarray(values)[0]))
            sequence.append(sequence_prediction)
            sequence_gt.append(int(row["gt_index"]))
            disagreement.append(surface_prediction != sequence_prediction)
        decoded_texts = row.get("style_decoded_text") or []
        if decoded_texts and decoded_texts[0] is not None:
            decoded_text_available += 1
        if row.get("decoded_prediction") is not None:
            decoded_parsed += 1
    return {
        "cache": str(cache),
        "n": len(records),
        "surface_accuracy": float(np.mean(np.asarray(surface) == np.asarray(gt))) if records else None,
        "sequence_nll_n": len(sequence),
        "sequence_nll_accuracy": (
            float(np.mean(np.asarray(sequence) == np.asarray(sequence_gt))) if sequence else None
        ),
        "surface_sequence_disagreement": float(np.mean(disagreement)) if disagreement else None,
        "actual_decode_n": decoded_text_available,
        "actual_decode_parsed_n": decoded_parsed,
        "actual_decode_parse_coverage": (
            float(decoded_parsed / len(records)) if records else None
        ),
        "interface_locked": (
            bool(records)
            and len(sequence) == len(records)
            and decoded_text_available == len(records)
        ),
    }


def main() -> None:
    args = parse_args()
    qwen = qwen_readiness(args.qwen_snapshot)
    report = report_readiness(args.report_root)
    interfaces = [ce_interface(path) for path in args.ce_cache]
    result = {
        "validation_type": "wave0_readiness",
        "qwen_knowledge_judge": qwen,
        "report_clinical_metrics": report,
        "ce_interfaces": interfaces,
        "gates": {
            "knowledge_judge_ready": qwen["ready"],
            "report_metrics_ready": report["ready"],
            "ce_interface_locked": bool(interfaces) and all(item["interface_locked"] for item in interfaces),
        },
    }
    result["gates"]["passed"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
