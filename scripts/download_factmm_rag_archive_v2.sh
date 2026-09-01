#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
target_dir=/home/dbw/model_cache/factmm_rag/official_retriever_v1
target=$target_dir/model.zip
partial=$target.partial
url='https://drive.usercontent.google.com/download?id=1qV-atZdKX-PwBSWzEocf5I63vuF9Ri-g&export=download&confirm=t'
expected_bytes=11200064402
reserve_bytes=100000000000
mkdir -p "$target_dir"

free_bytes=$(df -B1 --output=avail /home/dbw | tail -1 | tr -d ' ')
remaining_bytes=$expected_bytes
if [[ -f "$partial" ]]; then
  partial_bytes=$(stat -c '%s' "$partial")
  if (( partial_bytes > expected_bytes )); then
    echo "partial archive is larger than the official object" >&2
    exit 2
  fi
  remaining_bytes=$((expected_bytes - partial_bytes))
fi
if (( free_bytes - remaining_bytes < reserve_bytes )); then
  echo "refusing FactMM-RAG archive download: the 100GB reserve would be breached" >&2
  exit 2
fi

if [[ ! -s "$target" ]]; then
  # The official endpoint supports byte ranges; -c resumes the exact .partial file.
  wget -c --output-document="$partial" "$url"
  test -s "$partial"
  actual_bytes=$(stat -c '%s' "$partial")
  if (( actual_bytes != expected_bytes )); then
    echo "official archive size mismatch: expected $expected_bytes, got $actual_bytes" >&2
    exit 3
  fi
  mv "$partial" "$target"
fi

actual_bytes=$(stat -c '%s' "$target")
if (( actual_bytes != expected_bytes )); then
  echo "completed archive size mismatch: expected $expected_bytes, got $actual_bytes" >&2
  exit 3
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
    "protocol_version": "factmm-rag-official-archive-download-v2",
    "source_url": sys.argv[2],
    "source_role": "official FactMM-RAG README RAG archive link",
    "content_disposition_filename": "model.zip",
    "path": str(path.resolve()),
    "bytes": path.stat().st_size,
    "sha256": sha256_file(path),
    "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "free_bytes_after": os.statvfs("/home/dbw").f_bavail * os.statvfs("/home/dbw").f_frsize,
    "archive_semantics_verified": False,
    "paper_native_generator_released": False,
    "paper_native_efficacy_authorized": False,
    "supersedes": "factmm-rag-official-checkpoint-download-v1",
    "supersession_reason": "official object is a ZIP archive, not a directly loadable .pt checkpoint",
}
record["fingerprint"] = sha256_json(record)
atomic_write_json(path.parent / "download_provenance_v2.json", record)
print(json.dumps(record, indent=2))
PY
