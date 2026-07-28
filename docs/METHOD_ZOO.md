# ANCHOR Method Zoo and Migration Notes

This document summarizes methods that are valuable enough to preserve for the next server/data migration. It separates publishable candidates, baselines, diagnostics, and stopped directions. Do not treat this file as a result-to-claim approval; it is a migration map.

## Current Main Candidate: Layer Evidence Transport (LET)

**Idea.** During normal autoregressive generation, mix final-layer logits with an intermediate decoder-layer evidence distribution:

```text
z_t = (1 - alpha) z_final,t + alpha z_layer,t
```

The method is training-free, one-pass, and works on complete generated sentences rather than canonical Yes/No token logits. Mathematically it is a reverse-KL barycenter / natural-parameter interpolation between the final distribution and an evidence-rich intermediate layer.

**Useful artifacts.**

- Code entrypoints: `anchor/corrected_sgta/run_anchor_layer_expert_pilot.py`, `anchor/corrected_sgta/run_anchor_let_rule75.py`, `anchor/corrected_sgta/run_anchor_let_report.py`
- Mechanism docs: `docs/MECHANISM_REPORT_LET.md`, `docs/LET_VISTA_MECHANISM_PILOT.md`, `docs/WEEKLY_REPORT_LET_20260727.md`
- Full RULE/MIMIC summary: `corrected_runs/final_anchor_let_rule75_v1_standard_full/summary.json`

**Evidence boundary.**

- RULE/MIMIC full sentence, 3470 rows: baseline 75.45%, LET 80.75%, +5.30pp; exact McNemar `p=5.76e-16`; patient-cluster 95% CI `[+4.10,+6.55]pp`.
- Same run increases recall strongly but lowers specificity, so report sensitivity/specificity and balanced accuracy.
- A native-generation variant reported 80.12% -> 82.54% (+2.42pp), but protocol differs from official mitigation baselines.
- Do not claim LET is novel without citing and comparing VISTA Self-Logits Augmentation; current novelty is a medical-domain specialization/mechanism, not the logit-mixing operator alone.

## Closest Comparator: VISTA / Self-Logits Augmentation

**Idea.** VISTA mixes augmented logits from earlier/windowed layers into final logits. It collides with LET at the operator level.

**Useful artifacts.**

- Vendored code: `third_party/VISTA/`, `third_party/baselines/VISTA/`
- Comparison doc: `docs/LET_VISTA_MECHANISM_PILOT.md`
- Pilot result: `corrected_runs/final_let_vista_mechanism_pilot_v1_n128_combined_analysis.json`

**Evidence boundary.**

On n=128 RULE/MIMIC, normalized LET and normalized VISTA-window both reach about 76.56% and differ mainly in operating point. The pilot does **not** support claiming VISTA fails. A paper-grade comparison needs a frozen full run including VISTA SLA and, if available, VSV+SLA.

## Strong Baselines and Mitigation Methods

These should be migrated because they are important comparators even when they do not help.

| Method | Scope | Status | Notes |
|---|---|---|---|
| Greedy / Beam | VQA and reports | Required baseline | Beam was strongest among official RULE-style mitigation runs in one common 3466-row table. |
| OPERA | VQA/OE/report adapters | Useful baseline | Positive on RULE-style binary VQA in prior run; keep as strong decoding baseline. |
| DoLa | VQA/OE/report adapters | Important negative/neutral | No robust gain under current medical protocol; useful to distinguish positive evidence transport from subtractive layer contrast. |
| VCD | VQA/OE/report adapters | Important negative | Degraded on medical binary VQA in current evidence; keep to show generic visual contrast may not transfer. |
| M3ID | VQA/OE/report adapters | Important negative | Similar medical-transfer failure pattern to VCD. |
| PAI | VQA/OE/report adapters | Near-greedy | Keep for broad mitigation comparison. |
| AVISC / DAMRO / AGLA / ClearSight / VHR | VLM mitigation / visual steering | Migration candidates | Code exists locally under `third_party/`; require per-method bridge and protocol audit before claims. |
| SCA-T / LAME / LATA | Constrained CE | CE-only comparators | Useful for fixed Yes/No classification, but not a full-sentence VQA/report solution. |

Primary method registry: `configs/methods.yaml`. Heavy implementations live under `anchor/corrected_sgta/` and `third_party/`; registry-only smoke outputs are not accuracy results.

## Source-Guided / DG Methods to Preserve

| Method family | Core object | Representative files | Current role |
|---|---|---|---|
| SGTA / Fourier source views | Low-frequency source-style counterfactuals | `sgta_confgen.py`, `run_sgta_tuned.sh`, `run_feature_sgta.sh` | Motivation and diagnostic; previous view aggregation often had no reliable headroom. |
| Source-margin calibration | Source-conditioned binary decision margin | `rule_source_absolute_margin.py`, `evaluate_rule_source_robust_margin.py` | Valuable RULE/MIMIC CE result path; task-specific, not final general method. |
| Source-word-center report generation | Output-side source word/statistics center | `run_mmedrag_word_center_final.py` | Useful report-generation diagnostic and baseline; avoid overclaiming clinical factuality. |
| ANCHOR-Flow / Output-path SGTA | Complete output evidence path and source-success path energy | `run_anchor_flow_sgta_gate.py`, `analyze_anchor_flow_sgta_gate.py`, `docs/ANCHOR_FINAL_DIRECTION.md` | Conceptually aligned with DG story; current pilot gate did not pass. |
| ANCHOR-Riemann gate | Token trajectory manifold distance, sliced Wasserstein, Dirichlet/Fisher energies | `riemann_geometry.py`, `run_anchor_riemann_gate.py`, `analyze_anchor_riemann_gate.py` | Good mathematical diagnostic; current selector underperforms despite oracle headroom. |
| ANCHOR-NBP | Normal-bundle proximal correction from local source feature patches | `nbp_geometry.py`, `run_anchor_nbp_pilot.py`, `analyze_anchor_nbp_pilot.py` | Interesting source-center geometry; only mini/smoke evidence so far. |
| Visual-odds amplification (VAF) | Multiply visual-token attention odds by `exp(eta)` while preserving within-image attention distribution | `run_rule_vaf_gate.py` | Clean visual-grounding intervention; needs full evidence before inclusion. |
| Style nuisance / output-side DG analysis | Separate style nuisance from clinical evidence and output path | `analyze_style_nuisance_subspace.py`, `analyze_output_side_dg_hypothesis.py` | Keep as theory/diagnostic material. |

## Open-Ended / Report Protocol

Report/OE evaluation must be task-aware. Do not reuse binary VQA parser logic for report claims.

- Prompt/task utilities: `anchor/corrected_sgta/report_protocol.py`
- OE generation: `anchor/corrected_sgta/infer_oe.py`
- Report evaluation: `anchor/corrected_sgta/evaluate_oe_reports.py`, `anchor/corrected_sgta/evaluate_medheval_report_clinical.py`
- Protocol doc: `docs/OE_EVALUATION_PROTOCOL.md`

Current report pilot for LET: `corrected_runs/final_anchor_let_report_v1_pilot32/analysis.json`. Its clinical metric direction checks passed, but n=32 is a pilot; do not turn it into a full report-generation claim without a locked full run.

## Stopped or Demoted Directions

- Naive Fourier/FedDG view ensembling: useful motivation but insufficient as final method unless a future gate shows candidate oracle headroom and safe selection.
- Threshold-only source distance / selective prediction: too brittle and not aligned with the desired simple general method.
- Riemann/NLL candidate selection as currently implemented: CE selector underperformed greedy despite oracle headroom; keep for diagnostics.
- ConfGen-v1 for OE: previous ROUGE-based admissibility produced vacuous sets; only revisit with clinical admissibility and sufficient calibration.
- Canonical Yes/No single-token scoring as final interface: useful analysis, but paper should evaluate complete generated text and parsed full sentence where possible.

## Migration Priorities

1. Preserve the exact LET full-run artifacts and all fingerprints.
2. Preserve official baseline/mitigation outputs and parser code.
3. Preserve source-bank/source-view builders for DG motivation and future experiments.
4. Preserve OE/report protocol code and metric direction checks.
5. Migrate VISTA/ClearSight/AGLA/VHR code carefully; ignore unresolved upstream LFS pointer assets unless reacquired.
6. On the new server, first reproduce smoke and LET n=256, then rerun the frozen full LET-vs-greedy/beam/OPERA comparison on a common manifest.

## Claim Discipline

- LET can currently be described as the strongest empirically supported direction in this workspace, but with specificity tradeoff.
- VISTA must be treated as a close prior/comparator, not a failed strawman.
- DG/source-center methods currently support motivation and diagnostics more strongly than final accuracy claims.
- Any AAAI claim needs a locked common sample set, paired statistics, and an independent result-to-claim audit.
