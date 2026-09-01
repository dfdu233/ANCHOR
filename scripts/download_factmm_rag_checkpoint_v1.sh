#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
target_dir=/home/dbw/model_cache/factmm_rag/official_retriever_v1
target=$target_dir/factmm_rag_retriever_checkpoint.pt
partial=$target.partial
url='https://drive.google.com/file/d/1qV-atZdKX-PwBSWzEocf5I63vuF9Ri-g/view?usp=sharing'
mkdir -p "$target_dir"

free_bytes=$(df -B1 --output=avail /home/dbw | tail -1 | tr -d ' ')
if (( free_bytes < 120000000000 )); then
  echo "refusing FactMM-RAG checkpoint download with less than 120GB free" >&2
  exit 2
fi

if [[ ! -s "$target" ]]; then
  find "$target_dir" -maxdepth 1 -type f -name 'factmm_rag_retriever_checkpoint.pt.partial*.part' -delete
  rm -f "$partial"
  /home/dbw/ANCHOR/.venv-full/bin/gdown --fuzzy "$url" -O "$partial"
  test -s "$partial"
  mv "$partial" "$target"
fi

PYTHONPATH=. .venv-full/bin/python - "$target" "$url" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json

path = Path(sys.argv[1])
record = {
    "protocol_version": "factmm-rag-official-checkpoint-download-v1",
    "source_url": sys.argv[2],
    "source_role": "official README RAG checkpoint link",
    "path": str(path.resolve()),
    "bytes": path.stat().st_size,
    "sha256": sha256_file(path),
    "nonempty": path.stat().st_size > 0,
    "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "free_bytes_after": os.statvfs("/home/dbw").f_bavail * os.statvfs("/home/dbw").f_frsize,
    "paper_native_generator_released": False,
    "paper_native_efficacy_authorized": False,
}
record["fingerprint"] = sha256_json(record)
atomic_write_json(path.parent / "download_provenance.json", record)
print(json.dumps(record, indent=2))
PY
