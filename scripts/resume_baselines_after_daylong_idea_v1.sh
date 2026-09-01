#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/daylong_idea_search_v1/baseline_resume_state.jsonl

stamp() {
  /opt/miniconda3/bin/python - "$state" "$1" "$2" <<'PY'
import datetime,json,sys
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session": sys.argv[2],
        "status": sys.argv[3],
    }) + "\n")
PY
}

stamp supervisor waiting_for_idea_fatal_experiments_v2
while tmux has-session -t idea_fatal_experiments_v2 2>/dev/null; do
  sleep 30
done
stamp supervisor idea_queue_terminal

launch() {
  local session=$1 script=$2
  if tmux has-session -t "$session" 2>/dev/null; then
    stamp "$session" already_running
    return
  fi
  tmux new-session -d -s "$session" "cd /home/dbw/ANCHOR && bash $script"
  stamp "$session" restarted
}

launch baseline_matrix_v1 scripts/run_baseline_native_long_queue_v1.sh
launch baseline_llava_methods_v2 scripts/run_baseline_llava_methods_long_queue_v1.sh
launch baseline_cross_methods_v3 scripts/run_baseline_cross_model_methods_long_queue_v1.sh
launch baseline_shared_rag_v1 scripts/run_baseline_shared_rag_long_queue_v1.sh
launch baseline_vhr_full scripts/run_vhr_official_full_queue_v1.sh
stamp supervisor complete
