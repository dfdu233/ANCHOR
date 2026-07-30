# Layerwise style-orbit mechanism result

This folder contains compact outputs from a controlled Qwen2.5-VL-7B
representation probe.

- `summary.json`: all layerwise paired statistics and provenance fingerprint.
- `alignment_mechanism.png`: main mechanism figure.
- `layerwise_orbits.png`: full layer-by-layer supplementary figure.

The 72 MB raw feature tensor is intentionally excluded from Git and stored at:

`/root/autodl-fs/data/dbw/anchor_alignment_contraction_v1/layerwise_style_orbit40_v1.npz`

Its SHA256 and metadata fingerprint are recorded in `summary.json`. The
experiment uses 40 frontal MIMIC development images, 38 patients, and the exact
same six source styles as the preceding alignment-contraction audit.

Main observation: in this one seed-42 comparison, the matched lineage has
5.48% lower final prompt-state style susceptibility than the fixed
image-permuted lineage (95% patient-cluster CI 4.00--7.43% lower). It is
accompanied by lower style drift and greater real-versus-null distance. The
corresponding complete-sentence clinical-evidence change is not certified.
This is a mechanism diagnostic, not a utility result.
