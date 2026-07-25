"""Fail-closed frozen report for a matching diagnostic cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import freeze_alignment_report_release2 as implementation


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    cache = Path(argument("--cache"))
    diagnostic = Path(argument("--diagnostic-analysis"))
    cache_meta = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    diagnostic_payload = json.loads(diagnostic.read_text())
    if diagnostic_payload.get("fingerprint") != cache_meta.get("fingerprint"):
        raise RuntimeError("diagnostic analysis/cache fingerprint mismatch")
    implementation.main()


if __name__ == "__main__":
    main()
