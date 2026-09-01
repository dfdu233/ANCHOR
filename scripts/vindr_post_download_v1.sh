#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR

root=/home/dbw/datasets/physionet/vindr-cxr/1.0.0
while tmux has-session -t vindr-selective-download 2>/dev/null; do
  sleep 60
done

expected=$(wc -l < "$root/manifests/image_urls.txt")
actual=$(find "$root/train" -maxdepth 1 -type f -name '*.dicom' | wc -l)
if [[ "$actual" -ne "$expected" ]]; then
  echo "download incomplete: $actual/$expected; audit/triplets not started" >&2
  exit 2
fi
bash scripts/download_vindr_subset.sh triplets "$root"
