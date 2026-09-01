#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
# The local proxy repeatedly terminates Xet CDN TLS handshakes.  Standard HF
# HTTP/LFS honors the existing .incomplete range file and resumes reliably.
export HF_HUB_DISABLE_XET=1
cli=/opt/miniconda3/envs/huatuo/bin/huggingface-cli

while read -r repository directory; do
  if [[ "$directory" == llava-v1.5-7b ]] \
    && [[ -f /home/dbw/models/llava-v1.5-7b/pytorch_model-00001-of-00002.bin ]] \
    && [[ -f /home/dbw/models/llava-v1.5-7b/pytorch_model-00002-of-00002.bin ]]; then
    continue
  fi
  until "$cli" download "$repository" --local-dir "/home/dbw/models/$directory" --max-workers 1; do
    sleep 10
  done
done <<'EOF'
liuhaotian/llava-v1.5-7b llava-v1.5-7b
juliozhao/hadpo-llava-1.5 hadpo-llava-1.5
zhyang2226/opadpo-lora_llava-v1.5-7b opadpo-lora-llava-v1.5-7b
Artanic30/DA-DPO_llava_v1.5_7B da-dpo-llava-v1.5-7b
psp-dada/LLaVA-v1.5-7B-SENTINEL sentinel-llava-v1.5-7b
yuezih/llava-v1.5-7b-selective-23k-lora less-is-more-llava-v1.5-7b
EOF

mkdir -p /home/dbw/ANCHOR/corrected_runs/detached_jobs
touch /home/dbw/ANCHOR/corrected_runs/detached_jobs/trained-baseline-downloads-v1.done
