"""Extract target-blind canary rows from frozen full shared-RAG generations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANARY = ROOT / "corrected_runs/polarity_firewall_canary_v1"
DEFAULT_SOURCE = (
    ROOT / "corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected list: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict[str, Any]) -> str:
    return str(row.get("question_id") or row.get("qid") or row.get("id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, default=DEFAULT_CANARY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--models", nargs="+", default=["huatuo", "hulu"])
    args = parser.parse_args()

    audit: dict[str, Any] = {
        "protocol": "polarity-firewall-cached-arm-extraction-v1",
        "selection_uses_target_label": False,
        "models": {},
    }
    for model in args.models:
        model_audit: dict[str, Any] = {}
        for manifest_name, source_arm in (
            ("raw_rag", "rag"),
            ("no_context", "no_context"),
        ):
            manifest_path = args.canary_root / f"{manifest_name}.json"
            source_path = (
                args.source_root / model / "cxr_vishal" / source_arm / "answers.jsonl"
            )
            manifest = load_json(manifest_path)
            wanted = [key(row) for row in manifest]
            if len(wanted) != len(set(wanted)):
                raise ValueError(f"duplicate canary qid in {manifest_path}")
            source_rows = load_jsonl(source_path)
            source_by_qid = {key(row): row for row in source_rows}
            if len(source_by_qid) != len(source_rows):
                raise ValueError(f"duplicate source qid in {source_path}")
            missing = [qid for qid in wanted if qid not in source_by_qid]
            if missing:
                raise ValueError(f"missing {len(missing)} qids from {source_path}: {missing[:3]}")
            selected = [source_by_qid[qid] for qid in wanted]
            output = args.canary_root / "cached_answers" / model / manifest_name / "answers.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
            model_audit[manifest_name] = {
                "n": len(selected),
                "qid_order_exact": [key(row) for row in selected] == wanted,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256(manifest_path),
                "source": str(source_path.relative_to(ROOT)),
                "source_sha256": sha256(source_path),
                "output": str(output.relative_to(ROOT)),
                "output_sha256": sha256(output),
            }
        audit["models"][model] = model_audit
    output = args.canary_root / "cached_answers" / "extraction_audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
