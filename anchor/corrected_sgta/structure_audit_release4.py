"""Fail-closed structure reconstruction using the exact inference Source Bank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from corrected_sgta import structure_audit_v2 as implementation
from corrected_sgta.frequency_alignment_release3 import feddg_frequency_interpolation_release3
from corrected_sgta.source_bank_v2 import load_manifest, sha256_file
from corrected_sgta.source_bank_v3 import verify_source_artifacts
from corrected_sgta.structure_audit_wave_a import structure_proxy


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    cache = Path(argument("--cache"))
    source_bank = Path(argument("--source-bank"))
    metadata = json.loads(cache.with_suffix(cache.suffix + ".meta.json").read_text())
    expected = metadata["config"]["source_bank_sha256"]
    actual = sha256_file(source_bank)
    if actual != expected:
        raise RuntimeError(f"structure audit/source-bank mismatch: {actual} != {expected}")
    verify_source_artifacts(load_manifest(source_bank))
    implementation.feddg_frequency_interpolation_v2 = feddg_frequency_interpolation_release3
    implementation.structure_proxy = structure_proxy
    implementation.main()


if __name__ == "__main__":
    main()
