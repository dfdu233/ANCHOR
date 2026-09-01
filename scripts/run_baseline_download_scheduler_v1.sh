#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR

biomed=/home/dbw/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/open_clip_pytorch_model.bin
while [[ ! -f "$biomed" || "$(stat -c %s "$biomed" 2>/dev/null || echo 0)" -ne 783705670 ]]; do sleep 30; done

if [[ ! -f /home/dbw/model_cache/report_metrics/download_audit.json ]]; then
  tmux has-session -t baseline_report_metrics_download 2>/dev/null || \
    tmux new-session -d -s baseline_report_metrics_download "cd /home/dbw/ANCHOR && bash scripts/download_report_metric_checkpoints_v1.sh"
fi
while [[ ! -f /home/dbw/model_cache/report_metrics/download_audit.json ]] || ! /opt/miniconda3/bin/python - <<'PY'
import json
raise SystemExit(0 if json.load(open('/home/dbw/model_cache/report_metrics/download_audit.json')).get('passed') else 1)
PY
do sleep 30; done

if [[ ! -f /home/dbw/.venvs/vhr-official-4.45/env_audit.json ]]; then
  tmux has-session -t baseline_vhr_env 2>/dev/null || \
    tmux new-session -d -s baseline_vhr_env "cd /home/dbw/ANCHOR && bash scripts/prepare_vhr_official_env_v1.sh"
fi
while [[ ! -f /home/dbw/.venvs/vhr-official-4.45/env_audit.json ]]; do sleep 30; done

checkpoint=/home/dbw/models/llava-hf-llava-1.5-7b-hf-16952161b5e90aea6e332e36a6fe99024096dd0a/download_audit.json
if [[ ! -f "$checkpoint" ]]; then
  tmux has-session -t baseline_vhr_checkpoint 2>/dev/null || \
    tmux new-session -d -s baseline_vhr_checkpoint "cd /home/dbw/ANCHOR && bash scripts/download_vhr_official_llava15hf_v1.sh"
fi
