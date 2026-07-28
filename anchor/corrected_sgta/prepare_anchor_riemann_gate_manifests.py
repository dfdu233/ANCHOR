#!/usr/bin/env python3
"""Prepare small normalized manifests for ANCHOR-Riemann gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from corrected_sgta.anchor_transport import stable_json_sha256
from corrected_sgta.run_anchor_riemann_gate import REPORT_PROMPT


VERSION = "anchor-riemann-gate-manifest-prep-v1"


def atomic_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return list(payload["records"])
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if rows:
        return rows
    raise ValueError(f"unsupported manifest schema: {path}")


def _exists(image: str, image_root: Path | None) -> bool:
    path = Path(image)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return path.is_file()


def convert_rule_ce(
    rows: list[dict[str, Any]], *, limit: int, image_root: Path | None
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        conversations = row.get("conversations")
        if isinstance(conversations, list) and len(conversations) >= 2:
            prompt = str(conversations[0].get("value", "")).replace("<image>", "").strip()
            answer = str(conversations[1].get("value", "")).strip()
        else:
            prompt = str(row.get("prompt", row.get("question", ""))).replace("<image>", "").strip()
            answer = str(row.get("answer", row.get("ground_truth", ""))).strip()
        image = row.get("image")
        if not prompt or not answer or not image or not _exists(str(image), image_root):
            continue
        base = str(row.get("id", row.get("question_id", image)))
        output.append(
            {
                "id": f"mimic-ce-{index:06d}-{stable_json_sha256(base)[:10]}",
                "source_id": base,
                "image": str(image),
                "prompt": prompt,
                "answer": answer,
                "domain": "mimic_rule_ce",
                "patient_id": str(base).split("/", 2)[1] if str(base).startswith("p") and "/" in str(base) else base,
            }
        )
        if limit and len(output) >= limit:
            break
    if not output:
        raise RuntimeError("no CE rows converted")
    return output


def convert_report(
    rows: list[dict[str, Any]], *, limit: int, domain: str, image_root: Path | None
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        image = row.get("image", row.get("image_path"))
        if isinstance(image, list):
            image = image[0] if image else None
        answer = row.get("answer", row.get("reference", row.get("report")))
        identifier = row.get("id", row.get("study_id", index))
        if not image or not answer or not _exists(str(image), image_root):
            continue
        output.append(
            {
                "id": f"{domain}-{index:06d}-{stable_json_sha256(str(identifier))[:10]}",
                "source_id": str(identifier),
                "image": str(image),
                "prompt": REPORT_PROMPT,
                "answer": str(answer).strip(),
                "domain": domain,
                "patient_id": str(row.get("subject_id", row.get("patient_id", identifier))),
            }
        )
        if limit and len(output) >= limit:
            break
    if not output:
        raise RuntimeError("no report rows converted")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-ce", type=Path, required=True)
    parser.add_argument("--mimic-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--ce-limit", type=int, default=256)
    parser.add_argument("--oe-limit", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ce_rows = convert_rule_ce(
        load_rows(args.mimic_ce), limit=args.ce_limit, image_root=args.image_root
    )
    oe_rows = convert_report(
        load_rows(args.mimic_report),
        limit=args.oe_limit,
        domain="mimic_report_oe",
        image_root=args.image_root,
    )
    atomic_json(args.output_dir / "mimic_ce.json", ce_rows)
    atomic_json(args.output_dir / "mimic_report_oe.json", oe_rows)
    meta = {
        "version": VERSION,
        "mimic_ce": str(args.mimic_ce.resolve()),
        "mimic_report": str(args.mimic_report.resolve()),
        "ce_rows": len(ce_rows),
        "oe_rows": len(oe_rows),
        "target_labels_used_for_generation_or_selection": False,
    }
    (args.output_dir / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
