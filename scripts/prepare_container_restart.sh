#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR

state=corrected_runs/detached_jobs/common-rag-ce-ladder-v3.json
marker=corrected_runs/detached_jobs/container-restart-requested.json
checkpoint=corrected_runs/detached_jobs/container-restart-checkpoint.json

mkdir -p "$(dirname "$marker")"
/opt/miniconda3/envs/huatuo/bin/python - "$marker" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1]); t = p.with_suffix(p.suffix + ".tmp")
t.write_text(json.dumps({"requested_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
os.replace(t, p)
PY

# Native CE runners append and fsync one complete qid at a time. Freeze the
# complete process group before the snapshot so an immediate restart can later
# resume the exact manifest prefix without rerunning completed samples.
if [[ -s "$state" ]]; then
  /opt/miniconda3/envs/huatuo/bin/python - "$state" <<'PY'
import json, os, signal, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1]); row = json.load(p.open())
if row.get("status") not in {"starting", "running"}:
    raise SystemExit(0)
pid = int(row.get("pid", -1))
try:
    if open(f"/proc/{pid}/stat").read().split()[2] == "Z":
        raise ProcessLookupError
    os.killpg(os.getpgid(pid), signal.SIGSTOP)
except (OSError, ProcessLookupError, ValueError, IndexError):
    raise SystemExit(0)
row["status_before_container_restart"] = row["status"]
row["status"] = "paused_for_container_restart"
row["paused_at"] = datetime.now(timezone.utc).isoformat()
t = p.with_suffix(p.suffix + ".tmp")
t.write_text(json.dumps(row, indent=2) + "\n")
os.replace(t, p)
PY
fi

/opt/miniconda3/envs/huatuo/bin/python - "$marker" "$checkpoint" <<'PY'
import hashlib, json, os, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

root = Path("/home/dbw/datasets/physionet/vindr-cxr/1.0.0")
urls = root / "manifests/image_urls.txt"
summary = root / "manifests/summary.json"
if not urls.is_file() or not summary.is_file():
    raise SystemExit("VinDr persistent-volume prerequisites are missing")
payload = {
    "protocol_version": "anchor-container-restart-checkpoint-v1",
    "checkpoint_id": str(uuid.uuid4()),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "persistent_root": "/home/dbw",
    "vindr_root": str(root),
    "vindr_summary_sha256": sha(summary),
    "image_urls_sha256": sha(urls),
    "expected_dicoms": sum(1 for line in urls.read_text().splitlines() if line.strip()),
    "observed_dicoms": len(list((root / "train").glob("*.dicom"))),
    "rag_state": json.load(open("corrected_runs/detached_jobs/common-rag-ce-ladder-v3.json")),
}
for target in map(Path, sys.argv[1:]):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, target)
sentinel = Path("/home/dbw/.anchor_persistent_volume_v1.json")
temp = sentinel.with_suffix(".tmp")
temp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temp, sentinel)
print(json.dumps(payload, indent=2))
PY

# Stop downstream waiters before stopping the resumable wget session, so an
# intentional restart is not mistaken for a completed download.
for session in common-rag-finalize-v3 vindr-mechanism-boundary vindr-post-download; do
  tmux kill-session -t "$session" 2>/dev/null || true
done
if tmux has-session -t vindr-selective-download 2>/dev/null; then
  tmux send-keys -t vindr-selective-download C-c
  for _ in $(seq 1 20); do
    tmux has-session -t vindr-selective-download 2>/dev/null || break
    sleep 1
  done
  tmux kill-session -t vindr-selective-download 2>/dev/null || true
fi

# The marker prevents recovery while the old container is being shut down.
/opt/miniconda3/envs/huatuo/bin/python - <<'PY'
import json, os, signal
from pathlib import Path
p = Path("corrected_runs/detached_jobs/research-watchdog-v1.json")
if p.exists():
    row = json.load(p.open())
    for key in ("child_pid", "pid"):
        try: os.kill(int(row[key]), signal.SIGTERM)
        except (KeyError, ProcessLookupError, ValueError): pass
PY
sync
echo "Research jobs are quiesced and on-disk state is synchronized. Container restart is safe."
