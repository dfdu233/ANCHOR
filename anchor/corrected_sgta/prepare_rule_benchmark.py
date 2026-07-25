"""Prepare an auditable RULE-compatible benchmark without changing labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "rule-data-protocol-v1"
PAPER_REFERENCE = {
    "citation": "Xia et al., EMNLP 2024, Tables 1 and 2",
    "url": "https://aclanthology.org/2024.emnlp-main.62/",
    "datasets": {
        "iuxray": {"images": 589, "questions": 2573},
        "harvard": {"images": 713, "questions": 4285},
        "mimic": {"images": 700, "questions": 3470},
    },
    "llava_med_1_5": {
        "iuxray": {"accuracy": 75.47, "precision": 53.17, "recall": 80.49, "f1": 64.04},
        "harvard": {"accuracy": 63.03, "precision": 92.13, "recall": 61.46, "f1": 74.11},
        "mimic": {"accuracy": 75.79, "precision": 81.01, "recall": 79.38, "f1": 80.49},
    },
    "greedy": {
        "iuxray": {"accuracy": 76.88, "precision": 54.41, "recall": 82.53, "f1": 65.59},
        "harvard": {"accuracy": 78.32, "precision": 91.59, "recall": 82.38, "f1": 86.75},
        "mimic": {"accuracy": 82.54, "precision": 82.68, "recall": 81.73, "f1": 85.98},
    },
    "rule_accuracy": {"iuxray": 87.84, "harvard": 87.12, "mimic": 83.92},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-root", type=Path, default=Path("/root/autodl-tmp/RULE"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--pilot-size", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots = {"iuxray": Path("/root/autodl-tmp/MedHEval/images/IU-Xray")}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or dataset not in PAPER_REFERENCE["datasets"]:
            raise ValueError(f"expected DATASET=PATH, got {value!r}")
        roots[dataset] = Path(path)
    return roots


def rule_binary_label(text: object) -> str:
    """Mirror RULE/LLaVA POPE label coercion, including its empty-to-yes behavior."""
    sentence = "" if text is None else str(text).replace("\n", " ").strip()
    if "." in sentence:
        sentence = sentence.split(".", 1)[0]
    words = sentence.replace(",", "").split(" ")
    return "No" if any(word in {"No", "no", "not"} for word in words) else "Yes"


def stable_pilot(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    keyed = []
    for row in rows:
        digest = hashlib.sha256(f"RULE-pilot-v1:{row['qid']}".encode()).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed)[: min(size, len(keyed))]]


def git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    roots = parse_roots(args.image_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "rule_commit": git_commit(args.rule_root),
        "paper_reference": PAPER_REFERENCE,
        "datasets": {},
        "warning": (
            "Paper-reference numbers are transcribed, not reproduced. A run is paper-comparable "
            "only when every annotation and image is present and model/decoder match."
        ),
    }
    for dataset, expected in PAPER_REFERENCE["datasets"].items():
        source = args.rule_root / "data" / "test" / f"{dataset}_test.jsonl"
        raw_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        qids = [str(row["question_id"]) for row in raw_rows]
        if len(qids) != len(set(qids)):
            raise ValueError(f"{dataset}: duplicate question_id")
        noncanonical_answers = sum(
            str(row.get("answer", "")).strip().lower().rstrip(".") not in {"yes", "no"}
            for row in raw_rows
        )
        root = roots.get(dataset)
        available: list[dict[str, Any]] = []
        missing: list[str] = []
        converted: list[dict[str, Any]] = []
        for row in raw_rows:
            image = str(row["image"])
            present = bool(root and (root / image).is_file())
            item = {
                "qid": str(row["question_id"]),
                "question_id": row["question_id"],
                "img_name": image,
                "question": str(row["question"]).replace("<image>", "").strip(),
                "answer": rule_binary_label(row.get("answer")),
                "raw_answer": row.get("answer"),
                "question_type": "binary",
                "source": f"RULE/{dataset}",
                "report": row.get("report"),
                "image_available": present,
            }
            converted.append(item)
            (available if present else missing).append(item if present else image)
        dataset_dir = args.output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "questions.full.json").write_text(json.dumps(converted, indent=2))
        (dataset_dir / "questions.available.json").write_text(json.dumps(available, indent=2))
        (dataset_dir / f"questions.pilot{args.pilot_size}.json").write_text(json.dumps(stable_pilot(available, args.pilot_size), indent=2))
        unique_images = {row["image"] for row in raw_rows}
        labels = Counter(rule_binary_label(row.get("answer")).lower() for row in raw_rows)
        manifest["datasets"][dataset] = {
            "annotation": str(source),
            "annotation_sha256": sha256(source),
            "questions": len(raw_rows),
            "unique_image_paths": len(unique_images),
            "paper_reported_images": expected["images"],
            "paper_reported_questions": expected["questions"],
            "label_counts": dict(sorted(labels.items())),
            "noncanonical_raw_answers": noncanonical_answers,
            "label_normalization": "RULE eval_pope first-period, no/not convention",
            "image_root": str(root) if root else None,
            "available_questions": len(available),
            "missing_questions": len(missing),
            "missing_unique_images": len(set(missing)),
            "paper_comparable_data_complete": len(raw_rows) == expected["questions"] and not missing,
            "note": "Paper image counts can exceed unique JSON image paths when studies have multiple views.",
        }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
