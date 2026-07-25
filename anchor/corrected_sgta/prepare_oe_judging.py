#!/usr/bin/env python3
"""Export a blinded, reproducible OE candidate annotation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from corrected_sgta.cache import iter_successes
from corrected_sgta.clinical_judgments import candidate_key
from corrected_sgta.protocol_v2 import PROTOCOL_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("knowledge", "report"))
    parser.add_argument("--fields", nargs="+", default=("sampled", "style_sampled"))
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def template(task: str) -> dict:
    if task == "knowledge":
        return {
            "hallucination_score": None,
            "clinically_admissible": None,
            "rationale": "",
            "annotator_id": "",
            "rubric_version": "MedHEval-0-5-v1",
            "rubric": "MedHEval 0-5; higher means more severe hallucination; <=2 is admissible",
        }
    return {
        "clinical_entity_precision": None,
        "clinical_fact_recall": None,
        "critical_contradiction": None,
        "clinically_admissible": None,
        "radgraph_f1": None,
        "ratescore": None,
        "rationale": "",
        "annotator_id": "",
        "metric_bundle_version": "",
        "metric_manifest_sha256": "",
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + ".meta.json").read_text())
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unsupported cache protocol")
    records = list(iter_successes(args.cache, metadata["fingerprint"]))
    bundle_payload = (
        f"{metadata['fingerprint']}:{args.task}:{args.seed}:"
        + ",".join(args.fields)
    )
    annotation_bundle_id = hashlib.sha256(bundle_payload.encode()).hexdigest()[:24]
    items = []
    manifest = []
    seen = set()
    for row in records:
        for field in args.fields:
            for index, output in enumerate(row.get(field) or []):
                item_id = candidate_key(row["qid"], field, index, output, metadata["fingerprint"])
                if item_id in seen:
                    continue
                seen.add(item_id)
                items.append(
                    {
                        "item_id": item_id,
                        "cache_fingerprint": metadata["fingerprint"],
                        "annotation_bundle_id": annotation_bundle_id,
                        "question": row.get("question", ""),
                        "ground_truth": row.get("answer", ""),
                        "model_answer": output.get("text", ""),
                        **template(args.task),
                    }
                )
                manifest.append(
                    {
                        "item_id": item_id,
                        "cache_fingerprint": metadata["fingerprint"],
                        "annotation_bundle_id": annotation_bundle_id,
                        "qid": str(row["qid"]),
                        "field": field,
                        "candidate_index": index,
                        "style": output.get("style", "unknown"),
                        "domain_id": output.get("domain_id", "unknown"),
                        "candidate_id": output.get("candidate_id"),
                    }
                )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(items)).tolist()
    if args.max_items:
        order = order[: args.max_items]
    selected_ids = {items[index]["item_id"] for index in order}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(items[index]) + "\n" for index in order))
    args.manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest if row["item_id"] in selected_ids)
    )
    print(json.dumps({"n_blinded_items": len(order), "task": args.task}, indent=2))


if __name__ == "__main__":
    main()
