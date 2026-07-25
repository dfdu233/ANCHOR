"""Independent reconstruction audit for release-2 spectral alignment."""

from __future__ import annotations

from corrected_sgta import structure_audit_v2 as implementation
from corrected_sgta.frequency_alignment_release2 import feddg_frequency_interpolation_release2
from corrected_sgta.structure_audit_wave_a import structure_proxy


def main() -> None:
    implementation.feddg_frequency_interpolation_v2 = feddg_frequency_interpolation_release2
    implementation.structure_proxy = structure_proxy
    implementation.main()


if __name__ == "__main__":
    main()
