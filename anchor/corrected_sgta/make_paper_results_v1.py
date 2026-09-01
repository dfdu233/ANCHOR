#!/usr/bin/env python3
"""Build paper-ready Baseline status/score figures with explicit N/A cells.

The source of truth is the unified coverage audit.  This script never infers
completion from an answer-file line count and never converts pending cells to
zero.  Score cells are populated only when the audit points to a score artifact
and its primary metric is finite.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


VERSION = "paper-results-figure-v1"
DATASETS = ["cxr_vishal", "knowledge_mimic_ce", "slake_fine_grained", "vqa_rad_official_oe", "visual_mimic_oe", "iu_xray_report", "mimic_cxr_report"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def dotted(payload: Any, path: str | None) -> float | None:
    if not path:
        return None
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return finite(value)


def score_from_artifact(path: str | None) -> tuple[float | None, str | None]:
    if not path or not Path(path).is_file():
        return None, None
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    primary = payload.get("primary_metric")
    value = dotted(payload, primary)
    if value is not None:
        return value, str(primary)
    # Report summaries use clinical_metrics rather than a single dotted
    # primary_metric.  RadGraph complete is the paper-facing clinical score.
    groups = ((payload.get("clinical_metrics") or {}).get("groups") or {})
    if isinstance(groups, dict):
        for group in groups.values():
            estimate = dotted(group, "bootstrap_ci95.radgraph_complete.estimate")
            if estimate is None:
                estimate = dotted(group, "radgraph_complete")
            if estimate is not None:
                return estimate, "clinical_metrics.radgraph_complete"
    return None, str(primary) if primary else None


def load_rows(audit: Path) -> tuple[list[str], list[dict[str, Any]]]:
    data = json.loads(audit.read_text())
    rows = data.get("cells", [])
    keys = sorted({f"{r.get('track','?')}|{r.get('model','?')}|{r.get('method','?')}" for r in rows}, key=lambda x: (x.split("|")[0], x.split("|")[1], x.split("|")[2]))
    return keys, rows


def status_color(status: str) -> tuple[int, int, int]:
    return {"completed": (141, 211, 199), "N/A": (220, 220, 220), "pending": (255, 224, 150), "running_or_partial": (244, 160, 160), "generated-unscored": (244, 190, 160)}.get(status, (230, 230, 230))


def draw_grid(rows: list[str], cells: dict[tuple[str, str], dict[str, Any]], out: Path, mode: str) -> None:
    left, top, cell_w, cell_h = 370, 92, 170, 42
    width, height = left + cell_w * len(DATASETS) + 30, top + cell_h * len(rows) + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = "Baseline coverage audit" if mode == "status" else "Baseline primary score (N/A = not evaluated)"
    draw.text((20, 18), title, fill=(20, 20, 20), font=font(25, True))
    draw.text((20, 50), "Source: unified coverage_audit.json; pending/partial cells are not treated as zero.", fill=(80, 80, 80), font=font(14))
    for j, dataset in enumerate(DATASETS):
        x = left + j * cell_w + cell_w // 2
        draw.text((x, top - 13), dataset.replace("_", "\n"), fill=(20, 20, 20), font=font(12, True), anchor="mm", align="center")
    for i, key in enumerate(rows):
        y = top + i * cell_h
        track, model, method = key.split("|", 2)
        label = f"{track}/{model}/{method}"
        draw.text((left - 12, y + cell_h // 2), label, fill=(20, 20, 20), font=font(12), anchor="rm")
        for j, dataset in enumerate(DATASETS):
            x = left + j * cell_w
            cell = cells.get((key, dataset), {"status": "N/A"})
            status = cell.get("status", "N/A")
            if mode == "status":
                fill = status_color(status)
                label_value = {"completed": "OK", "N/A": "N/A", "pending": "PEND", "running_or_partial": "PART", "generated-unscored": "UNS"}.get(status, status[:5])
            else:
                value = cell.get("score")
                if value is not None and status == "completed":
                    # Scores are accuracies/F1-like values on [0,1].
                    shade = int(max(0, min(255, 245 - 170 * value)))
                    fill = (shade, shade + 8 if shade < 247 else 247, 255)
                    label_value = f"{100*value:.1f}"
                else:
                    # A completed generation with no paper-facing score is
                    # still not a numeric result (e.g. OE auxiliary output).
                    # Keep it visibly separate from pending/partial while
                    # using N/A in the score figure rather than fabricating a
                    # zero or displaying completion as a score.
                    fill = status_color("N/A") if status == "completed" else status_color(status)
                    label_value = "N/A" if status in ("N/A", "completed") else status[:5].upper()
            draw.rectangle((x + 1, y + 1, x + cell_w - 1, y + cell_h - 1), fill=fill, outline=(255, 255, 255))
            draw.text((x + cell_w // 2, y + cell_h // 2), label_value, fill=(20, 20, 20), font=font(12, True), anchor="mm")
    legend_y = top + cell_h * len(rows) + 20
    legend = [("OK", status_color("completed")), ("N/A", status_color("N/A")), ("PEND", status_color("pending")), ("PART", status_color("running_or_partial"))]
    for i, (label, color) in enumerate(legend):
        x = left + i * 120
        draw.rectangle((x, legend_y, x + 22, legend_y + 18), fill=color)
        draw.text((x + 28, legend_y + 9), label, fill=(20, 20, 20), font=font(12), anchor="lm")
    image.save(out, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_keys, raw = load_rows(args.audit)
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw:
        key = f"{row.get('track','?')}|{row.get('model','?')}|{row.get('method','?')}"
        score, metric = score_from_artifact(row.get("score_artifact"))
        cells[(key, str(row.get("dataset")))] = {**row, "score": score, "metric": metric}
    draw_grid(row_keys, cells, args.output_dir / "figure_baseline_coverage.png", "status")
    draw_grid(row_keys, cells, args.output_dir / "figure_baseline_primary_score.png", "score")
    # Full matrices are useful for machine inspection but too tall for a
    # readable paper page.  Emit two row-balanced panels for appendix/main
    # paper inclusion at full landscape width.
    for part, start in enumerate(range(0, len(row_keys), 24), 1):
        subset = row_keys[start : start + 24]
        draw_grid(subset, cells, args.output_dir / f"figure_baseline_coverage_part{part}.png", "status")
        draw_grid(subset, cells, args.output_dir / f"figure_baseline_primary_score_part{part}.png", "score")
    payload = {"version": VERSION, "audit": str(args.audit.resolve()), "datasets": DATASETS, "rows": row_keys, "cells": [{**v, "row": k[0], "dataset": k[1]} for k, v in sorted(cells.items())]}
    (args.output_dir / "paper_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output_dir.resolve()), "rows": len(row_keys), "cells": len(cells)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
