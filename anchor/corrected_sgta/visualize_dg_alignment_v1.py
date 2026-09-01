#!/usr/bin/env python3
"""Create a dependency-light visual audit of DG/style interventions.

This is deliberately post-hoc: it never trains a model and never changes an
answer.  It accepts either the ``style_phenomenon`` score records or the
FEDD-G ``raw_generations`` records and writes a self-contained HTML dashboard.
The dashboard separates image-change guards (PSNR/edge correlation) from
claim movement, which makes it possible to tell evidence destruction from
decoder/domain sensitivity.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import statistics
import io
from pathlib import Path
from typing import Any

from PIL import Image


VERSION = "dg-alignment-visual-audit-v1"


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        for key in ("records", "rows", "samples", "data"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"unsupported input container: {path}")


def number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def score_delta(score: dict[str, Any]) -> float | None:
    for key in ("yes_minus_no", "logit_delta", "margin", "score", "probability"):
        value = number(score.get(key))
        if value is not None:
            return value
    yes, no = number(score.get("yes_logit")), number(score.get("no_logit"))
    if yes is not None and no is not None:
        return yes - no
    return None


def variants(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize known DG output formats into variant rows."""
    output: list[dict[str, Any]] = []
    scores = record.get("scores")
    guards = record.get("style_guards") or {}
    if isinstance(scores, dict):
        for name, value in scores.items():
            value = value if isinstance(value, dict) else {"value": value}
            guard = guards.get(name) if isinstance(guards, dict) else {}
            guard = guard if isinstance(guard, dict) else {}
            output.append({
                "name": str(name),
                "delta": score_delta(value),
                "prediction": value.get("prediction"),
                "psnr": number(guard.get("psnr")),
                "edge": number(guard.get("edge_correlation")),
                "passed": guard.get("passed"),
                "text": record.get("text") if name == "original" else None,
            })
        return output
    candidates = record.get("candidates")
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            structure = meta.get("structure") if isinstance(meta, dict) else {}
            structure = structure if isinstance(structure, dict) else {}
            output.append({
                "name": str(item.get("style") or item.get("name") or "unknown"),
                "delta": number(item.get("yes_minus_no")) or number(item.get("margin")),
                "prediction": item.get("strict_prediction") or item.get("pope_prediction"),
                "psnr": number(structure.get("psnr")),
                "edge": number(structure.get("edge_correlation")),
                "passed": not item.get("skipped") and not item.get("error"),
                "text": item.get("text"),
                "nll": number(item.get("mean_token_nll")),
            })
        return output
    paired_meta = record.get("metadata") or {}
    if isinstance(paired_meta, dict) and paired_meta.get("variant"):
        # Output produced by run_dg_paired_validation_v1: native text is
        # carried in metadata because the native answer was intentionally not
        # regenerated.
        output.append({
            "name": "original",
            "delta": None,
            "prediction": None,
            "psnr": None,
            "edge": None,
            "passed": True,
            "text": paired_meta.get("native_text", ""),
        })
        output.append({
            "name": str(paired_meta.get("variant")),
            "delta": None,
            "prediction": None,
            "psnr": None,
            "edge": None,
            "passed": True,
            "text": record.get("text", ""),
            "nll": number(paired_meta.get("mean_token_nll")),
            "view_path": paired_meta.get("view_path"),
        })
        return output
    # Evidence-DG records have no explicit style score; retain a useful row
    # for source/quality inspection and token-trajectory visualization.
    evidence = record.get("generated_evidence") or {}
    final = evidence.get("layers", {}).get(str(evidence.get("final_layer"))) if isinstance(evidence, dict) else None
    output.append({
        "name": "original",
        "delta": score_delta(final or {}),
        "prediction": record.get("prediction") or record.get("pope_prediction"),
        "psnr": None,
        "edge": None,
        "passed": True,
        "text": record.get("text") or record.get("raw_text"),
        "trajectory": evidence.get("layers", {}) if isinstance(evidence, dict) else {},
    })
    return output


def image_data(path: Path) -> str | None:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((320, 320))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=72, optimize=True)
            raw = buffer.getvalue()
    except OSError:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def fmt(value: Any, digits: int = 3) -> str:
    value = number(value)
    return "—" if value is None else f"{value:.{digits}f}"


def resolve_image(record: dict[str, Any], root: Path | None) -> Path | None:
    value = record.get("image") or record.get("image_path") or record.get("img_name")
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    candidate = Path(str(value))
    if candidate.is_file():
        return candidate
    if root is not None and (root / candidate).is_file():
        return root / candidate
    return None


def resolve_variant_image(record: dict[str, Any], name: str, root: Path | None) -> Path | None:
    """Best-effort lookup for prepared DG views under ``root/name``."""
    meta = record.get("metadata") or {}
    if name != "original" and isinstance(meta, dict) and meta.get("view_path"):
        direct = Path(str(meta["view_path"]))
        if direct.is_file():
            return direct
    if root is None:
        return None
    value = record.get("image") or record.get("image_path") or record.get("img_name")
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    relative = Path(str(value))
    candidates = [root / name / relative, root / name / relative.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list((root / name).rglob(relative.name)) if (root / name).is_dir() else []
    return matches[0] if matches else None


def svg_scatter(points: list[tuple[float, float, str, str]], width: int = 760, height: int = 360) -> str:
    if not points:
        return '<div class="empty">No numeric style score and guard pair was found.</div>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax += 1.0
    if ymax == ymin:
        ymax += 1.0
    pad = 45
    def x(v: float) -> float:
        return pad + (v - xmin) / (xmax - xmin) * (width - 2 * pad)
    def y(v: float) -> float:
        return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)
    body = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="DG style drift versus image preservation">']
    body.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/>')
    body.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>')
    body.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle">image change (1-edge correlation)</text>')
    body.append(f'<text x="14" y="{height/2}" transform="rotate(-90 14 {height/2})" text-anchor="middle">claim delta from original</text>')
    for xv, yv, label, color in points:
        body.append(f'<circle cx="{x(xv):.1f}" cy="{y(yv):.1f}" r="5" fill="{color}" opacity=".78"><title>{html.escape(label)}</title></circle>')
    body.append('</svg>')
    return ''.join(body)


def svg_bars(summary: dict[str, dict[str, float]], width: int = 760, height: int = 300) -> str:
    names = list(summary)
    if not names:
        return '<div class="empty">No variant-level statistics.</div>'
    maxv = max(1.0, max((abs(summary[n].get("flip_rate", 0.0)) for n in names), default=1.0))
    barw = max(18.0, (width - 60) / max(1, len(names)) - 8)
    body = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="DG flip rate by variant">']
    for i, name in enumerate(names):
        x = 35 + i * (barw + 8)
        h = 210 * summary[name].get("flip_rate", 0.0) / maxv
        y = 235 - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barw:.1f}" height="{h:.1f}" fill="#377eb8"><title>{html.escape(name)} flip={summary[name].get("flip_rate", 0):.3f}</title></rect>')
        body.append(f'<text x="{x+barw/2:.1f}" y="252" transform="rotate(35 {x+barw/2:.1f} 252)" text-anchor="start">{html.escape(name[:18])}</text>')
    body.append('<text x="380" y="290" text-anchor="middle">prediction flip rate (relative to original)</text></svg>')
    return ''.join(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--variant-root", type=Path, help="optional root containing <variant>/<relative image>")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    records = read_records(args.input)
    if args.max_records > 0:
        records = records[: args.max_records]
    normalized = [(row, variants(row)) for row in records]
    by_name: dict[str, list[dict[str, Any]]] = {}
    points: list[tuple[float, float, str, str]] = []
    for row, items in normalized:
        base = next((x for x in items if x["name"] == "original"), None)
        base_delta = base.get("delta") if base else None
        base_pred = base.get("prediction") if base else None
        for item in items:
            name = item["name"]
            by_name.setdefault(name, []).append(item)
            if name == "original":
                continue
            if item.get("delta") is not None and base_delta is not None and item.get("edge") is not None:
                drift = item["delta"] - base_delta
                points.append((1.0 - item["edge"], drift, name, "#d73027" if item.get("prediction") != base_pred else "#4575b4"))
    summary: dict[str, dict[str, float]] = {}
    for name, items in by_name.items():
        if name == "original":
            continue
        flips = []
        psnr, edge, deltas = [], [], []
        for item in items:
            if item.get("prediction") is not None:
                # Compare against the record-level baseline later; this is
                # filled conservatively when no explicit prediction exists.
                flips.append(float(item.get("prediction") != "original"))
            if item.get("psnr") is not None:
                psnr.append(item["psnr"])
            if item.get("edge") is not None:
                edge.append(item["edge"])
            if item.get("delta") is not None:
                deltas.append(item["delta"])
        summary[name] = {
            "n": float(len(items)),
            "flip_rate": float(statistics.mean(flips)) if flips else 0.0,
            "mean_delta": float(statistics.mean(deltas)) if deltas else 0.0,
            "median_psnr": float(statistics.median(psnr)) if psnr else 0.0,
            "median_edge": float(statistics.median(edge)) if edge else 0.0,
        }
    # Correct prediction flips using the explicit original row.
    for row, items in normalized:
        base = next((x for x in items if x["name"] == "original"), None)
        for item in items:
            if item["name"] != "original" and item["name"] in summary and base:
                if item.get("prediction") is not None and base.get("prediction") is not None:
                    summary[item["name"]]["_flip_count"] = summary[item["name"]].get("_flip_count", 0.0) + float(item["prediction"] != base["prediction"])
                    summary[item["name"]]["_pred_n"] = summary[item["name"]].get("_pred_n", 0.0) + 1.0
    for value in summary.values():
        if value.get("_pred_n", 0.0):
            value["flip_rate"] = value["_flip_count"] / value["_pred_n"]
        value.pop("_flip_count", None)
        value.pop("_pred_n", None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": VERSION,
        "input": str(args.input.resolve()),
        "records": len(records),
        "variants": summary,
        "interpretation": {
            "high_image_change": "low edge correlation or low PSNR; a claim flip is not evidence of a useful DG correction",
            "low_image_change_high_flip": "decoder/domain sensitivity or low-margin boundary; inspect claim trajectory before mitigation",
            "no_style_score": "input contains image/quality evidence only; add claim logits to identify a DG direction",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    cards = []
    for index, (row, items) in enumerate(normalized):
        image = resolve_image(row, args.image_root)
        thumb = image_data(image) if image else None
        title = row.get("question_id") or row.get("id") or row.get("record_key") or index
        text = row.get("text") or row.get("raw_text") or row.get("baseline_text") or ""
        variant_thumbs = []
        for item in items:
            view = resolve_variant_image(row, item["name"], args.variant_root)
            data = image_data(view) if view else None
            if data:
                variant_thumbs.append(f'<div><small>{html.escape(item["name"])}</small><img src="{data}" alt="{html.escape(item["name"])}"/></div>')
        cards.append('<article class="card">' +
                     f'<h3>{html.escape(str(title))}</h3>' +
                     (f'<img src="{thumb}" alt="input image"/>' if thumb else '<div class="noimage">image unavailable</div>') +
                     (f'<div class="views">{"".join(variant_thumbs)}</div>' if variant_thumbs else '') +
                     '<table><tr><th>variant</th><th>claim delta</th><th>prediction</th><th>PSNR</th><th>edge</th><th>status</th></tr>' +
                     ''.join(f'<tr><td>{html.escape(str(x["name"]))}</td><td>{fmt(x.get("delta"), 4)}</td><td>{html.escape(str(x.get("prediction") or "—"))}</td><td>{fmt(x.get("psnr"), 2)}</td><td>{fmt(x.get("edge"), 3)}</td><td>{"skipped" if x.get("passed") is False else "used"}</td></tr>' for x in items) +
                     '</table>' +
                     (f'<details><summary>baseline/output text</summary><pre>{html.escape(str(text)[:1600])}</pre></details>' if text else '') +
                     '</article>')
    html_text = f'''<!doctype html><html><head><meta charset="utf-8"><title>DG alignment audit</title>
<style>body{{font:14px system-ui,sans-serif;background:#f6f7f9;color:#20242a;margin:24px}}h1,h2{{margin:0.4em 0}}.note{{background:#fff8dc;border-left:4px solid #d99b00;padding:10px;margin:10px 0 18px}}.panel{{background:white;padding:14px;margin:14px 0;border:1px solid #ddd;border-radius:8px;overflow:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}}.card{{background:white;padding:10px;border:1px solid #ddd;border-radius:8px}}.card img{{max-width:180px;max-height:180px;object-fit:contain;background:#111;display:block;margin-bottom:8px}}.views{{display:flex;gap:8px;overflow:auto;margin-bottom:8px}}.views img{{max-width:120px;max-height:120px}}.views small{{display:block;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #ddd;padding:4px;text-align:left}}pre{{white-space:pre-wrap;max-height:180px;overflow:auto}}.noimage,.empty{{color:#777;padding:18px;background:#f1f1f1}}svg{{width:100%;height:auto}}</style></head><body>
<h1>DG / style alignment visual audit</h1><div class="note">This dashboard is diagnostic, not a mitigation result. A style-induced answer change is only meaningful after separating image evidence destruction, low-margin boundary sensitivity, and language/domain prior effects.</div>
<div class="panel"><h2>Claim movement vs image change</h2>{svg_scatter(points)}</div>
<div class="panel"><h2>Prediction flips</h2>{svg_bars(summary)}</div>
<div class="panel"><h2>Variant summary</h2><pre>{html.escape(json.dumps(report, indent=2, ensure_ascii=False))}</pre></div>
<h2>Per-sample inspection</h2><div class="grid">{"".join(cards)}</div></body></html>'''
    (args.output_dir / "dashboard.html").write_text(html_text, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir.resolve()), "records": len(records), "variants": list(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
