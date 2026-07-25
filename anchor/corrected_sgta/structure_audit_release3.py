"""Independent reconstruction audit for release-3 alignment."""

from __future__ import annotations

from corrected_sgta import structure_audit_v2 as implementation
from corrected_sgta.frequency_alignment_release3 import feddg_frequency_interpolation_release3
from corrected_sgta.structure_audit_wave_a import structure_proxy


def main() -> None:
    implementation.feddg_frequency_interpolation_v2 = feddg_frequency_interpolation_release3
    implementation.structure_proxy = structure_proxy
    implementation.main()


if __name__ == "__main__":
    main()
