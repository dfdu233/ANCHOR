#!/usr/bin/env python3
"""Export fail-closed paper tables from the frozen baseline coverage ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file


VERSION = "baseline-paper-table-export-v3-strict-ce-task-aware-oe"
ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def put_ci(row: dict[str, Any], prefix: str, payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    row[prefix] = payload.get("estimate")
    row[f"{prefix}_ci95_lower"] = payload.get("ci95_lower")
    row[f"{prefix}_ci95_upper"] = payload.get("ci95_upper")


def parse_ce(row: dict[str, Any], score: dict[str, Any]) -> None:
    metrics = score["primary_multiclass"]
    official = score["official_benchmark_proxy"]
    row.update(
        accuracy=metrics["accuracy_invalid_as_error"],
        official_benchmark_accuracy=official["accuracy"],
        strict_overall_accuracy=metrics["accuracy_invalid_as_error"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        parse_rate=metrics["parse_rate"],
        confusion_json=json.dumps(metrics.get("confusion", {}), sort_keys=True),
    )
    put_ci(row, "accuracy", score.get("cluster_bootstrap_ci95", {}).get("accuracy_invalid_as_error"))
    put_ci(row, "official_benchmark_accuracy", official.get("cluster_bootstrap_ci95"))
    for answer_type, typed in metrics.get("by_answer_type", {}).items():
        for name in ("accuracy_invalid_as_error", "balanced_accuracy", "macro_f1", "parse_rate"):
            row[f"{answer_type}_{name}"] = typed.get(name)
    ci = score.get("cluster_bootstrap_ci95", {})
    for source, target in (
        ("accuracy_invalid_as_error", "strict_overall_accuracy"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("macro_f1", "macro_f1"),
        ("parse_rate", "parse_rate"),
    ):
        payload = ci.get(source)
        if payload:
            row[f"{target}_ci95_lower"] = payload.get("ci95_lower")
            row[f"{target}_ci95_upper"] = payload.get("ci95_upper")


def parse_oe(row: dict[str, Any], score: dict[str, Any]) -> None:
    absolute = score["absolute"]
    for name, payload in absolute["metrics"].items():
        put_ci(row, name, payload)
    diagnostics = absolute.get("output_diagnostics", {})
    row.update(
        mean_prediction_tokens=diagnostics.get("mean_prediction_tokens"),
        reference_coverage=diagnostics.get("reference_phrase_coverage_rate"),
        empty_rate=diagnostics.get("empty_rate"),
        cap_hit_rate=diagnostics.get("token_budget_hit_rate"),
    )


def parse_report(row: dict[str, Any], score: dict[str, Any]) -> None:
    text_groups = score["text_metrics"]["by_dataset_modality_method"]
    if len(text_groups) != 1:
        raise ValueError(f"expected one report text group, found {len(text_groups)}")
    text = next(iter(text_groups.values()))
    for name in (
        "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge_l", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1",
        "meteor", "token_f1",
    ):
        put_ci(row, name, text.get("bootstrap_ci95", {}).get(name))
    row.update(
        mean_prediction_words=text.get("mean_prediction_words"),
        normal_template_rate=text.get("normal_template_rate"),
        abnormal_finding_rate=text.get("abnormal_finding_rate"),
    )
    clinical = score.get("clinical_metrics") or {}
    groups = clinical.get("groups", {})
    if len(groups) > 1:
        raise ValueError(f"expected one report clinical group, found {len(groups)}")
    if groups:
        aggregate = next(iter(groups.values()))
        for name in (
            "radgraph_simple",
            "radgraph_partial",
            "radgraph_complete",
            "ratescore",
            "chexbert_example_f1_14",
            "chexbert_macro_f1_14",
            "chexbert_micro_f1_14",
        ):
            put_ci(row, name, aggregate.get("bootstrap_ci95", {}).get(name))


def write_visual_mimic_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "bleu_1", "bleu_2", "bleu_3", "bleu_4",
        "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "meteor",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        grouped.setdefault(str(item["model"]), []).append(item)
    ranks: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for model, items in grouped.items():
        for key in columns:
            values = sorted(
                {float(item[key]) for item in items if item.get(key) is not None},
                reverse=True,
            )
            ranks[(model, key)] = (
                values[0] if values else None,
                values[1] if len(values) > 1 else None,
            )
    def value(row: dict[str, Any], key: str) -> str:
        number = row.get(key)
        if number is None:
            return "N/A"
        numeric = float(number)
        rendered = f"{numeric:.4f}"
        best, second = ranks[(str(row["model"]), key)]
        if best is not None and numeric == best:
            return rf"\textbf{{{rendered}}}"
        if second is not None and numeric == second:
            return rf"\underline{{{rendered}}}"
        return rendered
    def escape(value: object) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{adjustbox}",
        r"\begin{document}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Task-aware Visual-MIMIC report-generation results.}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Model & Method & B-1 & B-2 & B-3 & B-4 & R-1 & R-2 & R-L & METEOR \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: (item["model"], item["method"])):
        cells = [escape(row["model"]), escape(row["method"]), *(value(row, key) for key in columns)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\end{table}", r"\end{document}",
    ])
    path.write_text("\n".join(lines) + "\n")


def task_for(dataset: str) -> str:
    if dataset in {"cxr_vishal", "knowledge_mimic_ce", "slake_fine_grained"}:
        return "CE"
    if dataset in {"vqa_rad_official_oe", "visual_mimic_oe"}:
        return "OE"
    return "report"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["track", "task", "dataset", "model", "method", "status", "expected", "actual", "reason"]
    extras = sorted({key for row in rows for key in row} - set(keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + extras)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage",
        type=Path,
        default=ROOT / "corrected_runs/paper_baselines_v1/full_matrix_v1/coverage_audit.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export a progress-only draft when the frozen matrix is incomplete.",
    )
    args = parser.parse_args()
    coverage = load(args.coverage)
    if not coverage.get("complete") and not args.allow_incomplete:
        raise ValueError(
            "coverage is incomplete; pass --allow-incomplete only for a DRAFT progress export"
        )
    rows: list[dict[str, Any]] = []
    parse_failures = []
    for cell in coverage["cells"]:
        row = {
            "track": cell["track"],
            "task": task_for(cell["dataset"]),
            "dataset": cell["dataset"],
            "model": cell["model"],
            "method": cell["method"],
            "status": cell["status"],
            "expected": cell["expected"],
            "actual": cell["actual"],
            "reason": cell.get("reason"),
            "score_artifact": cell.get("score_artifact"),
        }
        score_path = Path(cell["score_artifact"])
        if cell["status"] == "completed":
            try:
                score = load(score_path)
                if row["task"] == "CE":
                    parse_ce(row, score)
                elif row["task"] == "OE":
                    parse_oe(row, score)
                else:
                    parse_report(row, score)
                row["score_sha256"] = sha256_file(score_path)
            except Exception as error:
                parse_failures.append(
                    {"model": row["model"], "method": row["method"], "dataset": row["dataset"], "error": f"{type(error).__name__}: {error}"}
                )
        rows.append(row)
    if parse_failures:
        raise ValueError(f"completed cells with unreadable paper metrics: {parse_failures}")
    auxiliary_rows: list[dict[str, Any]] = []
    for cell in coverage.get("auxiliary_controls", []):
        task = task_for(cell["dataset"])
        row = {
            "track": "auxiliary_control",
            "task": task,
            "dataset": cell["dataset"],
            "model": cell["model"],
            "method": cell["control"],
            "status": cell["status"],
            "expected": cell["expected"],
            "actual": cell["actual"],
            "reason": cell.get("reason"),
            "score_artifact": cell.get("score_artifact"),
        }
        score_path = Path(cell["score_artifact"])
        if cell["status"] == "completed":
            if cell["control"] == "shared_rag_causal_comparisons_n200":
                comparison_paths = [Path(path) for path in cell["comparison_artifacts"]]
                row["comparison_artifacts_json"] = json.dumps(
                    [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in comparison_paths
                    ],
                    sort_keys=True,
                )
            else:
                score = load(score_path)
                if task == "CE":
                    parse_ce(row, score)
                elif task == "OE":
                    parse_oe(row, score)
                else:
                    parse_report(row, score)
                row["score_sha256"] = sha256_file(score_path)
        auxiliary_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_cells.csv", rows)
    for task in ("CE", "OE", "report"):
        write_csv(args.output_dir / f"{task.lower()}_table.csv", [row for row in rows if row["task"] == task])
    write_csv(args.output_dir / "na_reasons.csv", [row for row in rows if row["status"] == "N/A"])
    write_csv(args.output_dir / "auxiliary_controls.csv", auxiliary_rows)
    visual_report_rows: list[dict[str, Any]] = []
    visual_root = args.coverage.parent / "report_scores" / "visual_mimic"
    for score_path in sorted(visual_root.glob("*/*/summary.json")):
        row = {
            "track": "task_aware_supplement",
            "task": "report",
            "dataset": "visual_mimic_oe",
            "model": score_path.parent.parent.name,
            "method": score_path.parent.name,
            "status": "completed",
            "expected": 490,
            "actual": 490,
            "reason": None,
            "score_artifact": str(score_path),
            "score_sha256": sha256_file(score_path),
        }
        parse_report(row, load(score_path))
        visual_report_rows.append(row)
    write_csv(args.output_dir / "visual_mimic_report_table.csv", visual_report_rows)
    write_visual_mimic_latex(
        args.output_dir / "visual_mimic_report_table.tex", visual_report_rows
    )
    provenance = {
        "version": VERSION,
        "coverage": str(args.coverage.resolve()),
        "coverage_sha256": sha256_file(args.coverage),
        "exporter_sha256": sha256_file(Path(__file__)),
        "coverage_complete": coverage["complete"],
        "rows": len(rows),
        "completed_rows": sum(row["status"] == "completed" for row in rows),
        "na_rows": sum(row["status"] == "N/A" for row in rows),
        "auxiliary_rows": len(auxiliary_rows),
        "completed_auxiliary_rows": sum(row["status"] == "completed" for row in auxiliary_rows),
        "paper_ready": coverage["complete"] and all(row["status"] in {"completed", "N/A"} for row in rows),
        "draft": not coverage["complete"],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
