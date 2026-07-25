from __future__ import annotations

import json
from pathlib import Path

from anchor.runners.registry import REPO_ROOT


def main() -> None:
    rows = []
    for path in sorted((REPO_ROOT / "runs").glob("*/*/*/*/summary.json")):
        try:
            row = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            row = {"status": "unreadable", "path": str(path), "error": str(exc)}
        rows.append(row)
    reference = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "results_reference").glob("**/*.json"))
    payload = {"runs": rows, "reference_json": reference}
    out = REPO_ROOT / "runs" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
