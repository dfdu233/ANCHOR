"""Fail-closed entry for the description-derived PubMedVision bank builder."""

from __future__ import annotations

import sys
from pathlib import Path

from corrected_sgta import augment_pubmedvision_source_bank_release3 as implementation


def argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    base_manifest = Path(argument("--base-source-bank")).resolve()
    output_manifest = (Path(argument("--output-dir")) / "source_bank.json").resolve()
    if output_manifest == base_manifest:
        raise RuntimeError(
            "output Source Bank must differ from the immutable base Source Bank"
        )
    implementation.main()


if __name__ == "__main__":
    main()
