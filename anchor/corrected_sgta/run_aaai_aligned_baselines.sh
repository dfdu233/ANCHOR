#!/usr/bin/env bash
set -euo pipefail

# Protocol-aware baseline runner for AAAI experiments.
# It does not merge incompatible protocols. CE-logit methods are the main
# cross-model comparison; official generative mitigation is supplementary.

MODE="${1:-summarize}"
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
cd "$ROOT"

case "$MODE" in
  manifest)
    python -u -m corrected_sgta.aaai_baseline_registry
    ;;
  ce-reuse)
    test -d corrected_runs/full_v52
    python -u -m corrected_sgta.aaai_baseline_registry
    python -u -m corrected_sgta.summarize_aaai_aligned_baselines \
      --ce-dir corrected_runs/full_v52 \
      --output corrected_runs/aaai_aligned_baseline_summary_v1.json
    ;;
  ce-full)
    bash corrected_sgta/run_full_ce.sh
    bash corrected_sgta/run_scat.sh
    python -u -m corrected_sgta.summarize_aaai_aligned_baselines \
      --ce-dir corrected_runs/full_v52 \
      --output corrected_runs/aaai_aligned_baseline_summary_v1.json
    ;;
  generative-official)
    python -u -m corrected_sgta.run_official_mitigation \
      --datasets knowledge_ce cxr_vishal mm_vishal \
      --methods greedy DoLa PAI opera avisc m3id VCD damro \
      --chunk-size 512 \
      --max-new-tokens 8 \
      --continue-on-error
    python -u -m corrected_sgta.summarize_official_mitigation \
      --root corrected_runs/aaai_medheval_mitigation_full_v1 \
      --output corrected_runs/aaai_medheval_mitigation_full_v1/llava_med_official_mitigation_summary.json
    ;;
  summarize)
    python -u -m corrected_sgta.aaai_baseline_registry
    python -u -m corrected_sgta.summarize_aaai_aligned_baselines \
      --ce-dir corrected_runs/full_v52 \
      --generative-summary corrected_runs/aaai_medheval_mitigation_full_v1/llava_med_official_mitigation_summary.json \
      --output corrected_runs/aaai_aligned_baseline_summary_v1.json
    python -u -m corrected_sgta.plot_aaai_baseline_figure \
      --summary corrected_runs/aaai_aligned_baseline_summary_v1.json \
      --out-dir /root/autodl-tmp/AuthorKit27/Figures
    python -u -m corrected_sgta.make_aaai_unified_table \
      --summary corrected_runs/aaai_aligned_baseline_summary_v1.json \
      --output /root/autodl-tmp/AuthorKit27/result_tables/TABLE_tab_unified_protocol_aware_baselines.tex
    ;;
  *)
    echo "Usage: $0 [manifest|ce-reuse|ce-full|generative-official|summarize]" >&2
    exit 2
    ;;
esac
