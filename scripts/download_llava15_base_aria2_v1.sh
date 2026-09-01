#!/usr/bin/env bash
set -euo pipefail

target=/home/dbw/models/llava-v1.5-7b
mkdir -p "$target"

aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
  -d "$target" -o pytorch_model-00001-of-00002.bin \
  'https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model-00001-of-00002.bin?download=true' &
first=$!
aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
  -d "$target" -o pytorch_model-00002-of-00002.bin \
  'https://huggingface.co/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model-00002-of-00002.bin?download=true' &
second=$!
wait "$first"
wait "$second"
