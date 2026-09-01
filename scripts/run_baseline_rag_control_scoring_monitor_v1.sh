#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
root=corrected_runs/paper_baselines_v1/full_matrix_v1
controls="$root/rag_controls_n200/knowledge_mimic_ce"
manifest="$controls/selected_rag.json"
state=corrected_runs/detached_jobs/baseline-rag-control-scoring-monitor-v1.state.jsonl
mkdir -p "$controls/comparisons"

complete_rows() { [[ -f "$1" && "$(wc -l < "$1")" -eq "$2" ]]; }

while true; do
  for model in huatuo hulu llava qwen; do
    relevant="$root/shared_rag_generation/$model/knowledge_mimic_ce/rag/answers.jsonl"
    no_context="$root/shared_rag_generation/$model/knowledge_mimic_ce/no_context/answers.jsonl"
    shuffled="$controls/generation/$model/shuffled_context/answers.jsonl"
    image_swap="$controls/generation/$model/image_swap/answers.jsonl"
    target="$controls/comparisons/$model"
    mkdir -p "$target"
    complete_rows "$relevant" 2000 || continue
    complete_rows "$no_context" 2000 || continue
    complete_rows "$shuffled" 200 || continue
    complete_rows "$image_swap" 200 || continue
    if [[ -f "$target/rag_vs_shuffled_context.json" \
       && -f "$target/rag_vs_image_swap.json" \
       && -f "$target/rag_vs_no_context.json" ]]; then
      continue
    fi
    rc=0
    for spec in "rag|$relevant" "no_context|$no_context"; do
      IFS='|' read -r name source <<<"$spec"
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
        -m anchor.medeval.subset_answers_by_manifest_v1 \
        --manifest "$manifest" --answers "$source" --output "$target/$name.answers.jsonl" \
        >"$target/score.log" 2>&1 || rc=$?
    done
    for spec in \
      "shuffled_context|$shuffled" \
      "image_swap|$image_swap" \
      "no_context|$target/no_context.answers.jsonl"; do
      IFS='|' read -r name baseline <<<"$spec"
      [[ "$rc" -eq 0 ]] || break
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.compare_ce_arms \
        --manifest "$manifest" --baseline "$baseline" \
        --candidate "$target/rag.answers.jsonl" \
        --output "$target/rag_vs_${name}.json" --bootstrap-draws 5000 --seed 42 \
        >>"$target/score.log" 2>&1 || rc=$?
    done
    /opt/miniconda3/bin/python - "$state" "$model" "$rc" "$target" <<'PY'
import datetime,json,sys
path,model,rc,target=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model":model,"status":"completed" if rc=="0" else "failed","returncode":int(rc),"output":target}
with open(path,"a") as handle: handle.write(json.dumps(row)+"\n")
PY
  done
  sleep 300
done
