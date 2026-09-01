#!/usr/bin/env python3
"""CPU audit of DG/style-transfer failure modes using existing raw outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_style_phenomenon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in rows for name in (row.get("scores") or {}) if name != "original"})
    out: dict[str, Any] = {"n": len(rows), "variants": {}}
    for name in names:
        flips, drifts, margins, psnr, edge = [], [], [], [], []
        for row in rows:
            scores, guards = row.get("scores") or {}, row.get("style_guards") or {}
            if name not in scores or "original" not in scores:
                continue
            native, styled = scores["original"], scores[name]
            if native.get("prediction") is not None and styled.get("prediction") is not None:
                flips.append(float(native.get("prediction") != styled.get("prediction")))
            if num(native.get("yes_minus_no")) is not None and num(styled.get("yes_minus_no")) is not None:
                drifts.append(abs(float(styled["yes_minus_no"]) - float(native["yes_minus_no"])))
                margins.append((abs(float(native["yes_minus_no"])), bool(native.get("prediction") != styled.get("prediction"))))
            guard = guards.get(name) or {}
            if num(guard.get("psnr")) is not None:
                psnr.append(float(guard["psnr"]))
            if num(guard.get("edge_correlation")) is not None:
                edge.append(float(guard["edge_correlation"]))
        flipped_margin = [m for m, f in margins if f]
        stable_margin = [m for m, f in margins if not f]
        out["variants"][name] = {
            "n": len(flips),
            "flip_rate": statistics.mean(flips) if flips else None,
            "mean_abs_claim_drift": statistics.mean(drifts) if drifts else None,
            "median_native_margin_flipped": statistics.median(flipped_margin) if flipped_margin else None,
            "median_native_margin_stable": statistics.median(stable_margin) if stable_margin else None,
            "median_psnr": statistics.median(psnr) if psnr else None,
            "median_edge": statistics.median(edge) if edge else None,
        }
    return out


def summarize_feddg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({str(item.get("style")) for row in rows for item in (row.get("candidates") or []) if item.get("style") != "original"})
    out: dict[str, Any] = {"n": len(rows), "variants": {}}
    for name in names:
        flips, psnr, edge, nll = [], [], [], []
        for row in rows:
            candidates = {str(item.get("style")): item for item in (row.get("candidates") or [])}
            base = candidates.get("original")
            item = candidates.get(name)
            if not item or not base or item.get("skipped") or item.get("error"):
                continue
            if base.get("strict_prediction") is not None and item.get("strict_prediction") is not None:
                flips.append(float(base["strict_prediction"] != item["strict_prediction"]))
            structure = (item.get("metadata") or {}).get("structure") or {}
            if num(structure.get("psnr")) is not None:
                psnr.append(float(structure["psnr"]))
            if num(structure.get("edge_correlation")) is not None:
                edge.append(float(structure["edge_correlation"]))
            if num(item.get("mean_token_nll")) is not None and num(base.get("mean_token_nll")) is not None:
                nll.append(float(item["mean_token_nll"]) - float(base["mean_token_nll"]))
        out["variants"][name] = {
            "n": len(flips),
            "flip_rate": statistics.mean(flips) if flips else None,
            "median_psnr": statistics.median(psnr) if psnr else None,
            "median_edge": statistics.median(edge) if edge else None,
            "mean_nll_delta": statistics.mean(nll) if nll else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    if any(isinstance(row.get("scores"), dict) for row in rows):
        result = {"protocol": "style_phenomenon", **summarize_style_phenomenon(rows)}
    else:
        result = {"protocol": "feddg_raw_generations", **summarize_feddg(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "protocol": result["protocol"], "n": result["n"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
