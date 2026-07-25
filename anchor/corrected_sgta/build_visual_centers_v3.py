"""Strict-provenance entry point for visual Source Bank centers."""

from __future__ import annotations

from corrected_sgta import build_visual_centers_v2 as implementation
from corrected_sgta.provenance_v3 import code_identity, model_identity


def main() -> None:
    implementation.code_identity = code_identity
    implementation.model_identity = model_identity
    implementation.main()


if __name__ == "__main__":
    main()
