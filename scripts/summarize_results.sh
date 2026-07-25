#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path

root = Path("runs")
rows = []
if root.exists():
    for path in sorted(root.glob("*/*/*/*/summary.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
print(json.dumps({"runs": rows, "n": len(rows)}, indent=2))
PY
