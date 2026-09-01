#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR
corpus=corrected_runs/paper_baselines_v1/full_matrix_v1/rag/combined_corpus/corpus.jsonl
root=corrected_runs/paper_baselines_v1/full_matrix_v1/rag/bm25
mkdir -p "$root"

run_one() {
  local dataset=$1
  local queries="corrected_runs/unified_eval/inputs/baseline_matrix_v1/${dataset}.json"
  local out="$root/$dataset"
  PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.retrieve_common_rag \
    --corpus "$corpus" --queries "$queries" --output-dir "$out" --top-k 3
  PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.prepare_shared_rag_prompts_v1 \
    --queries "$queries" --retrieval "$out/retrieval.jsonl" --output "$out/rag.json"
}
export -f run_one
export corpus root
printf '%s\n' cxr_vishal knowledge_mimic_ce slake_fine_grained vqa_rad_official_oe visual_mimic_oe \
  | xargs -n1 -P3 bash -c 'run_one "$0"'
