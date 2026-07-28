#!/usr/bin/env python3
"""Evidence pack for the Output-Side Distributional Hypothesis.

The script is deliberately analysis-only: it never calls a VLM and never uses
labels to select outputs.  It gathers already-completed results into a compact
claim-driven packet:

1. visual-side interventions mostly fail to produce usable causal leverage;
2. output-side interventions show the strongest positive evidence;
3. token-level generation trajectories reveal output-side distribution gaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime in minimal envs
    plt = None

VERSION = "output-side-dg-hypothesis-analysis-v1"
DEFAULT_OUT = Path("corrected_runs/output_side_dg_hypothesis_v1")

PATHS = {
    "source_margin": Path("results_reference/rule_mimic_source_margin/result.json"),
    "word_center_metrics": Path("/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/mmedrag_word_center_final_full/metrics.json"),
    "word_center_raw": Path("/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/mmedrag_word_center_final_full/predictions.raw.jsonl"),
    "oe_collapse": Path("corrected_runs/oe_sanity_audit_v1_existing/summary.json"),
    "oe_prompt_sanity": Path("corrected_runs/oe_sanity_audit_v1_generation8/summary.json"),
    "riemann_ce": Path("corrected_runs/final_anchor_riemann_gate_v1/ce128_t160.analysis.json"),
    "riemann_oe": Path("corrected_runs/final_anchor_riemann_gate_v1/oe64_t160.analysis.json"),
    "riemann_oe_raw": Path("corrected_runs/final_anchor_riemann_gate_v1/oe64_t160.raw.jsonl"),
    "nbp_smoke": Path("corrected_runs/final_anchor_nbp_pilot_v1_minicheck/summary.json"),
    "style_projection": Path("corrected_runs/style_projection_generation_reachability_v1/result.json"),
    "mitigation": Path("corrected_runs/final_mitigation_full_v1/summary.json"),
    "dg_lora": Path("/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/mimic_pilot128/task_only_paired_analysis.json"),
}


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_get(obj: Any, *keys: Any, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def fmt_float(x: Any, ndigits: int = 4) -> Any:
    try:
        value = float(x)
    except Exception:
        return x
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> str:
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\begin{tabular}{" + "l" * len(columns) + "}")
    out.append("\\toprule")
    out.append(" & ".join(columns).replace("_", "\\_") + " \\")
    out.append("\\midrule")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.3f}"
            vals.append(str(val).replace("%", "\\%").replace("_", "\\_"))
        out.append(" & ".join(vals) + " \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def build_visual_failure_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    riem_ce = data.get("riemann_ce")
    if riem_ce:
        greedy = safe_get(riem_ce, "compact_metrics", "greedy", "accuracy")
        riem = safe_get(riem_ce, "compact_metrics", "riemann", "accuracy")
        auc = safe_get(riem_ce, "candidate_oracle", "candidate_level_auc_lower_is_better", "riemann_energy")
        nll_auc = safe_get(riem_ce, "candidate_oracle", "candidate_level_auc_lower_is_better", "sequence_nll")
        rows.append({
            "method": "Riemann evidence transport",
            "site": "output candidate geometry",
            "uses_source_or_visual_bank": "yes",
            "n": riem_ce.get("n"),
            "effect": fmt_float((riem - greedy) * 100 if riem is not None and greedy is not None else None, 2),
            "metric": "CE accuracy delta pp",
            "diagnostic": f"AUC {fmt_float(auc,3)} vs NLL {fmt_float(nll_auc,3)}",
            "failure_mode": "source geometry weaker than NLL / hurts accuracy",
            "claim_role": "visual/source geometry negative",
        })
    riem_oe = data.get("riemann_oe")
    if riem_oe:
        greedy = safe_get(riem_oe, "compact_metrics", "greedy", "rouge_l")
        riem = safe_get(riem_oe, "compact_metrics", "riemann", "rouge_l")
        auc = safe_get(riem_oe, "candidate_oracle", "candidate_level_auc_lower_is_better_weak_rouge_oracle", "riemann_energy")
        nll_auc = safe_get(riem_oe, "candidate_oracle", "candidate_level_auc_lower_is_better_weak_rouge_oracle", "sequence_nll")
        rows.append({
            "method": "Riemann evidence transport",
            "site": "OE candidate geometry",
            "uses_source_or_visual_bank": "yes",
            "n": riem_oe.get("n"),
            "effect": fmt_float((riem - greedy) * 100 if riem is not None and greedy is not None else None, 2),
            "metric": "OE ROUGE-L delta pp",
            "diagnostic": f"AUC {fmt_float(auc,3)} vs NLL {fmt_float(nll_auc,3)}",
            "failure_mode": "candidate headroom exists but source geometry does not select it",
            "claim_role": "visual/source geometry negative",
        })
    nbp = data.get("nbp_smoke")
    if nbp:
        rescue_break = safe_get(nbp, "rescue_break_vs_greedy", default={}) or {}
        # Prefer primary shift group if present; otherwise summarize smoke/minicheck.
        nbp_keys = [k for k in safe_get(nbp, "groups", default={}) if "|nbp|" in k]
        greedy_keys = [k for k in safe_get(nbp, "groups", default={}) if "|greedy|" in k]
        effect = None
        if nbp_keys and greedy_keys:
            nbp_b = safe_get(nbp, "groups", nbp_keys[0], "balanced_accuracy")
            greedy_b = safe_get(nbp, "groups", greedy_keys[0], "balanced_accuracy")
            if nbp_b is not None and greedy_b is not None:
                effect = (nbp_b - greedy_b) * 100
        rows.append({
            "method": "NBP normal-bundle projection",
            "site": "visual feature projection",
            "uses_source_or_visual_bank": "yes",
            "n": safe_get(nbp, "groups", nbp_keys[0], "n") if nbp_keys else None,
            "effect": fmt_float(effect, 2),
            "metric": "balanced accuracy delta pp",
            "diagnostic": str(rescue_break)[:120],
            "failure_mode": "no reliable rescue; projection lacks causal leverage",
            "claim_role": "visual feature negative",
        })
    style = data.get("style_projection")
    if style:
        summary = style.get("summary", {})
        base = safe_get(summary, "0.0", "ce", "accuracy")
        best_delta = None
        best_alpha = None
        for alpha, item in summary.items():
            acc = safe_get(item, "ce", "accuracy")
            if acc is not None and base is not None:
                delta = (acc - base) * 100
                if best_delta is None or delta > best_delta:
                    best_delta, best_alpha = delta, alpha
        rows.append({
            "method": "Style subspace projection",
            "site": "visual feature subspace",
            "uses_source_or_visual_bank": "yes",
            "n": safe_get(summary, "0.0", "ce", "n"),
            "effect": fmt_float(best_delta, 2),
            "metric": "best CE accuracy delta pp",
            "diagnostic": f"best alpha={best_alpha}",
            "failure_mode": "large projection mostly no-op or unsafe; small probe only",
            "claim_role": "visual feature negative",
        })
    dg = data.get("dg_lora")
    if dg:
        rows.append({
            "method": "DG/task LoRA pilot",
            "site": "trainable projector adapter",
            "uses_source_or_visual_bank": "partly",
            "n": dg.get("n"),
            "effect": fmt_float(dg.get("delta_pp"), 2),
            "metric": "MIMIC pilot accuracy delta pp",
            "diagnostic": f"flips={dg.get('paired_flips')}",
            "failure_mode": "task-only/DG component not independently positive in this cache",
            "claim_role": "adapter DG negative/diagnostic",
        })
    mitigation = data.get("mitigation")
    if mitigation:
        summary = mitigation.get("summary", {})
        for ds in ("knowledge_ce", "cxr_vishal", "mm_vishal"):
            greedy = safe_get(summary, ds, "greedy", "accuracy") or safe_get(summary, ds, "greedy", "strict_accuracy")
            vcd = safe_get(summary, ds, "VCD", "accuracy") or safe_get(summary, ds, "VCD", "strict_accuracy")
            if greedy is not None and vcd is not None:
                rows.append({
                    "method": f"VCD on {ds}",
                    "site": "contrastive decoding",
                    "uses_source_or_visual_bank": "no",
                    "n": safe_get(summary, ds, "greedy", "n"),
                    "effect": fmt_float((vcd - greedy) * 100, 2),
                    "metric": "accuracy delta pp",
                    "diagnostic": "official mitigation summary",
                    "failure_mode": "visual contrastive decoding degrades LLaVA-Med in this setup",
                    "claim_role": "decoding negative",
                })
    return rows


def build_output_positive_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sm = data.get("source_margin")
    if sm:
        iface = safe_get(sm, "interfaces", "rule_greedy_pope_baseline", default={}) or {}
        rows.append({
            "method": "Source-margin / boundary calibration",
            "task": "CE VQA",
            "dataset": "RULE MIMIC-CXR",
            "n": sm.get("n"),
            "baseline": fmt_float(iface.get("base_accuracy"), 4),
            "method_score": fmt_float(iface.get("calibrated_accuracy"), 4),
            "delta_pp": fmt_float(iface.get("delta_pp"), 2),
            "ci95_pp": str([fmt_float(x, 2) for x in sm.get("patient_cluster_bootstrap_delta_ci95_pp", [])]),
            "mcnemar_p": iface.get("mcnemar_exact_p"),
            "output_side_variable": "answer-boundary / generated sentence calibration",
        })
        iface2 = safe_get(sm, "interfaces", "complete_sequence_identity", default={}) or {}
        if iface2:
            rows.append({
                "method": "Complete-sequence identity + source margin",
                "task": "CE VQA",
                "dataset": "RULE MIMIC-CXR",
                "n": sm.get("n"),
                "baseline": fmt_float(iface2.get("base_accuracy"), 4),
                "method_score": fmt_float(iface2.get("calibrated_accuracy"), 4),
                "delta_pp": fmt_float(iface2.get("delta_pp"), 2),
                "ci95_pp": str([fmt_float(x, 2) for x in sm.get("patient_cluster_bootstrap_delta_ci95_pp", [])]),
                "mcnemar_p": iface2.get("mcnemar_exact_p"),
                "output_side_variable": "complete-sequence boundary",
            })
    wc = data.get("word_center_metrics")
    if wc:
        overall = safe_get(wc, "overall", default={}) or {}
        means = overall.get("means", {})
        delta = safe_get(overall, "paired_bootstrap_vs_baseline", "source_word_center", "rougeL", default={}) or {}
        rows.append({
            "method": "Source word-center output anchor",
            "task": "OE report",
            "dataset": "MMed-RAG all",
            "n": overall.get("n"),
            "baseline": fmt_float(safe_get(means, "baseline", "rougeL"), 4),
            "method_score": fmt_float(safe_get(means, "source_word_center", "rougeL"), 4),
            "delta_pp": fmt_float(delta.get("delta", 0) * 100, 2),
            "ci95_pp": str([fmt_float(x * 100, 2) for x in delta.get("ci95", [])]),
            "mcnemar_p": "n/a",
            "output_side_variable": "report length / style anchor",
        })
        for ds, item in wc.get("by_dataset", {}).items():
            means = item.get("means", {})
            delta = safe_get(item, "paired_bootstrap_vs_baseline", "source_word_center", "rougeL", default={}) or {}
            rows.append({
                "method": "Source word-center output anchor",
                "task": "OE report",
                "dataset": ds,
                "n": item.get("n"),
                "baseline": fmt_float(safe_get(means, "baseline", "rougeL"), 4),
                "method_score": fmt_float(safe_get(means, "source_word_center", "rougeL"), 4),
                "delta_pp": fmt_float(delta.get("delta", 0) * 100, 2),
                "ci95_pp": str([fmt_float(x * 100, 2) for x in delta.get("ci95", [])]),
                "mcnemar_p": "n/a",
                "output_side_variable": "report length / modality template",
            })
    sanity = data.get("oe_prompt_sanity")
    if sanity:
        summaries = sanity.get("summaries", {})
        # Include real-view prompt sanity rows.
        for key, item in summaries.items():
            if item.get("view") == "real":
                rows.append({
                    "method": f"Prompt sanity: {item.get('conv_mode')} + {item.get('prompt_mode')}",
                    "task": "OE report sanity",
                    "dataset": "MIMIC report n=8",
                    "n": item.get("n"),
                    "baseline": "template collapse diagnostic",
                    "method_score": fmt_float(item.get("rouge_l"), 4),
                    "delta_pp": "n/a",
                    "ci95_pp": "n/a",
                    "mcnemar_p": "n/a",
                    "output_side_variable": f"normal_template={fmt_float(item.get('normal_template_rate'),3)}, unique={fmt_float(item.get('unique_output_rate'),3)}",
                })
    return rows


def raw_word_center_dataset_stats(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import re
    normal_re = re.compile(r"\b(normal|unremarkable|no significant abnormal|no acute|clear)\b", re.I)
    abnormal_re = re.compile(r"\b(pneumonia|pneumothorax|cardiomegaly|effusion|edema|consolidation|opacity|atelectasis|fracture|congestion|mass|nodule|infiltrate)\b", re.I)
    out = []
    datasets = sorted({r.get("dataset", "unknown") for r in raw_rows})
    for ds in datasets:
        subset = [r for r in raw_rows if r.get("dataset", "unknown") == ds]
        for method in ("ground_truth", "baseline", "source_word_center"):
            if method == "ground_truth":
                texts = [r.get("ground_truth", "") for r in subset]
            else:
                texts = [safe_get(r, "candidates", method, default="") for r in subset]
            if not texts:
                continue
            out.append({
                "dataset": ds,
                "method": method,
                "n": len(texts),
                "mean_words": float(np.mean([len(str(t).split()) for t in texts])),
                "unique_outputs": len(set(texts)),
                "unique_rate": len(set(texts)) / len(texts),
                "normal_template_rate": float(np.mean([bool(normal_re.search(str(t))) for t in texts])),
                "abnormal_finding_rate": float(np.mean([bool(abnormal_re.search(str(t))) for t in texts])),
            })
    return out


def collect_token_points_from_riemann(raw_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    for row in raw_rows:
        for cand in row.get("candidates", []):
            acq = cand.get("acquisition", "candidate")
            if acq == "greedy" or cand.get("candidate_id") == "candidate-0":
                group = "riemann_oe_greedy"
            elif acq == "nucleus_sample":
                group = "riemann_oe_sampled"
            else:
                group = f"riemann_oe_{acq}"
            logps = cand.get("image_token_log_probabilities") or []
            token_count = max(1, len(logps))
            for i, lp in enumerate(logps):
                nll = -float(lp)
                groups.setdefault(group, []).append([nll, i / token_count, 1.0 if i == token_count - 1 else 0.0])
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in groups.items() if vals}


def collect_proxy_points_from_summaries(data: dict[str, Any], raw_stats: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    for row in raw_stats:
        # Proxy token trajectory when token scores are unavailable: length and EOS
        # collapse descriptors.  This is lower-fidelity than raw token logprobs,
        # but enough for domain-attractor heatmaps.
        key = f"{row['dataset']}|{row['method']}"
        mean_words = float(row["mean_words"])
        normal = float(row["normal_template_rate"])
        abnormal = float(row["abnormal_finding_rate"])
        groups[key] = [[mean_words / 160.0, normal, abnormal]]
    sanity = data.get("oe_prompt_sanity") or {}
    for key, item in (sanity.get("summaries", {}) or {}).items():
        if item.get("view") == "real":
            groups[f"sanity|{item.get('conv_mode')}|{item.get('prompt_mode')}"] = [[
                float(item.get("mean_words", 0)) / 160.0,
                float(item.get("normal_template_rate", 0)),
                float(item.get("abnormal_finding_rate", 0)),
            ]]
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in groups.items()}


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, projections: int = 64, quantiles: int = 64, seed: int = 0) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    dim = max(a.shape[1], b.shape[1])
    if a.shape[1] < dim:
        a = np.pad(a, ((0, 0), (0, dim - a.shape[1])))
    if b.shape[1] < dim:
        b = np.pad(b, ((0, 0), (0, dim - b.shape[1])))
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(projections, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-12)
    qs = np.linspace(0.0, 1.0, quantiles)
    vals = []
    for direction in dirs:
        qa = np.quantile(a @ direction, qs)
        qb = np.quantile(b @ direction, qs)
        vals.append(np.mean((qa - qb) ** 2))
    return float(math.sqrt(np.mean(vals)))


def distance_matrix(groups: dict[str, np.ndarray], projections: int = 64, seed: int = 0) -> dict[str, Any]:
    names = sorted(groups)
    matrix = []
    for left in names:
        row = []
        for right in names:
            row.append(sliced_wasserstein(groups[left], groups[right], projections=projections, seed=seed))
        matrix.append(row)
    return {"names": names, "matrix": matrix}


def plot_effects(path: Path, visual_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    entries = []
    for row in visual_rows:
        if isinstance(row.get("effect"), (int, float)):
            entries.append((row["method"][:28], float(row["effect"]), "visual-side"))
    for row in output_rows:
        if isinstance(row.get("delta_pp"), (int, float)):
            entries.append((f"{row['method'][:20]}\n{row['dataset']}", float(row["delta_pp"]), "output-side"))
    entries = entries[:16]
    if not entries:
        return
    labels, vals, kinds = zip(*entries)
    colors = ["#d95f02" if k == "visual-side" else "#1b9e77" for k in kinds]
    fig, ax = plt.subplots(figsize=(max(9, len(entries) * 0.55), 5))
    y = np.arange(len(entries))
    ax.barh(y, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Effect size (percentage points where applicable)")
    ax.set_title("Visual-side interventions mostly fail; output-side anchors produce the strongest gains")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmap(path: Path, dm: dict[str, Any], title: str) -> None:
    if plt is None:
        return
    names = dm.get("names", [])
    mat = np.asarray(dm.get("matrix", []), dtype=float)
    if not names or mat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.55), max(6, len(names) * 0.45)))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_findings(visual_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], raw_stats: list[dict[str, Any]], token_dm: dict[str, Any]) -> list[dict[str, str]]:
    findings = []
    positive_outputs = [r for r in output_rows if isinstance(r.get("delta_pp"), (int, float)) and r["delta_pp"] > 0]
    negative_visuals = [r for r in visual_rows if isinstance(r.get("effect"), (int, float)) and r["effect"] <= 0]
    findings.append({
        "observation": f"{len(positive_outputs)} output-side rows show positive deltas, while {len(negative_visuals)} visual-side rows are non-positive.",
        "interpretation": "The most reproducible correctable signal appears in generation behavior rather than feature-space/image-space transport.",
        "implication": "Output-side DG is a defensible paper framing if we avoid claiming visual recognition is universally solved.",
        "next_step": "Run a small confirmatory token-trace audit on structured prompt outputs with clinical metrics restored.",
    })
    mimic = [r for r in raw_stats if r.get("dataset") == "mimic" and r.get("method") in {"baseline", "source_word_center"}]
    if len(mimic) == 2:
        b = next(r for r in mimic if r["method"] == "baseline")
        s = next(r for r in mimic if r["method"] == "source_word_center")
        findings.append({
            "observation": f"MIMIC report baseline mean length {b['mean_words']:.1f} words and normal-template rate {b['normal_template_rate']:.3f}; word-center length {s['mean_words']:.1f} and abnormal-finding rate {s['abnormal_finding_rate']:.3f}.",
            "interpretation": "The report baseline is trapped in a short normal attractor; output anchoring changes the attractor more than visual methods did.",
            "implication": "OE gains should be framed as attractor correction, not proof of better visual grounding until clinical metrics are restored.",
            "next_step": "Evaluate structured/word-center outputs with RadGraph/RaTEScore in the known working MedUniEval metric environment.",
        })
    names = token_dm.get("names", [])
    if names:
        findings.append({
            "observation": f"Token/proxy output distribution matrix contains {len(names)} groups.",
            "interpretation": "The analysis can visualize domain/prompt separation in output space; raw token logprobs are available for Riemann OE and proxy descriptors for full report domains.",
            "implication": "The SW2 heatmap is supportive rather than final because full word-center raw lacks per-token logprobs.",
            "next_step": "Collect token logprob traces for 16 MIMIC/Harvard structured-vs-dataset runs if this becomes a main claim.",
        })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sw-projections", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = {key: load_json(path) for key, path in PATHS.items() if path.suffix == ".json"}
    raw_wc = load_jsonl(PATHS["word_center_raw"])
    raw_riem_oe = load_jsonl(PATHS["riemann_oe_raw"])

    visual_rows = build_visual_failure_table(data)
    output_rows = build_output_positive_table(data)
    raw_stats = raw_word_center_dataset_stats(raw_wc)

    token_groups = collect_token_points_from_riemann(raw_riem_oe)
    proxy_groups = collect_proxy_points_from_summaries(data, raw_stats)
    token_dm = distance_matrix(token_groups, projections=args.sw_projections, seed=20260727) if token_groups else {"names": [], "matrix": []}
    proxy_dm = distance_matrix(proxy_groups, projections=args.sw_projections, seed=20260727) if proxy_groups else {"names": [], "matrix": []}

    findings = build_findings(visual_rows, output_rows, raw_stats, token_dm)
    payload = {
        "version": VERSION,
        "fingerprint": stable_sha256({"version": VERSION, "paths": {k: str(v) for k, v in PATHS.items()}}),
        "paths": {key: {"path": str(path), "exists": path.exists()} for key, path in PATHS.items()},
        "acceptance": {
            "output_side_positive_rows": len([r for r in output_rows if isinstance(r.get("delta_pp"), (int, float)) and r["delta_pp"] > 0]),
            "visual_side_nonpositive_rows": len([r for r in visual_rows if isinstance(r.get("effect"), (int, float)) and r["effect"] <= 0]),
            "has_oe_collapse_evidence": data.get("oe_collapse") is not None and not data["oe_collapse"].get("admissible_for_report_generation_claim", True),
            "has_token_trace_distance": bool(token_groups),
            "claim_supported": "partial",
            "claim_boundary": "Supports generation-side/output-distribution framing as a paper hypothesis; does not prove visual recognition is unaffected or that clinical OE factuality improved.",
        },
        "visual_side_failures": visual_rows,
        "output_side_positives": output_rows,
        "report_generation_stats": raw_stats,
        "token_trace_sw2": token_dm,
        "proxy_output_sw2": proxy_dm,
        "findings": findings,
    }
    (args.output_dir / "output_side_dg_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(args.output_dir / "visual_side_failure_table.csv", visual_rows)
    write_csv(args.output_dir / "output_side_positive_table.csv", output_rows)
    write_csv(args.output_dir / "report_generation_stats.csv", raw_stats)

    tex = []
    tex.append(latex_table(visual_rows, ["method", "site", "n", "effect", "metric", "failure_mode"], "Visual-side and source-geometry interventions did not yield reliable gains.", "tab:visual_side_negative"))
    tex.append(latex_table(output_rows, ["method", "task", "dataset", "n", "baseline", "method_score", "delta_pp", "output_side_variable"], "Output-side interventions provide the strongest positive evidence.", "tab:output_side_positive"))
    (args.output_dir / "tables.tex").write_text("\n".join(tex))

    plot_effects(args.output_dir / "visual_vs_output_effects.png", visual_rows, output_rows)
    plot_heatmap(args.output_dir / "token_trace_sw2_heatmap.png", token_dm, "Token-level output trajectory SW2 (Riemann OE raw)")
    plot_heatmap(args.output_dir / "proxy_output_sw2_heatmap.png", proxy_dm, "Proxy output-distribution distance across report domains/prompts")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "summary": str(args.output_dir / "output_side_dg_summary.json"),
        "visual_rows": len(visual_rows),
        "output_rows": len(output_rows),
        "report_stats_rows": len(raw_stats),
        "token_groups": len(token_groups),
        "proxy_groups": len(proxy_groups),
        "acceptance": payload["acceptance"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
