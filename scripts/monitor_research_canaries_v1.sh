#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/detached_jobs/research-canaries-queue-v1.state.jsonl
heartbeat=corrected_runs/detached_jobs/research-canaries-monitor-v1.log
mkdir -p "$(dirname "$heartbeat")"

while true; do
  now=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  baseline_answers=0
  baseline_path=corrected_runs/paper_baselines_v1/full_matrix_v1/native/hulu/mimic_cxr_report/beam/answers.jsonl
  [[ -f "$baseline_path" ]] && baseline_answers=$(wc -l < "$baseline_path")
  research_answers=$(find \
    corrected_runs/matched_retrieval_polarity_pilot_v1/generated_answers \
    corrected_runs/polarity_firewall_canary_v1/generated_answers \
    -type f -name answers.jsonl -print0 2>/dev/null | xargs -0 -r wc -l | tail -1 | awk '{print $1}')
  research_answers=${research_answers:-0}
  printf '%s baseline_mimic_beam=%s research_answer_rows=%s\n' \
    "$now" "$baseline_answers" "$research_answers" >>"$heartbeat"

  if ! tmux has-session -t research_canaries_v1 2>/dev/null; then
    if ! { [[ -f "$state" ]] && tail -20 "$state" | grep -q '"status": "queue_completed"'; }; then
      if ! pgrep -f '[r]un_research_canaries_queue_v1.sh' >/dev/null; then
        tmux new-session -d -s research_canaries_v1 \
          'cd /home/dbw/ANCHOR && bash scripts/run_research_canaries_queue_v1.sh'
        printf '%s restarted_research_queue=1\n' "$now" >>"$heartbeat"
      fi
    fi
  fi
  sleep 60
done
