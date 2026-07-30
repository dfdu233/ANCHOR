# Multi-Lineage Style-Orbit Confirmation

This is a prespecified replication of the exploratory seed-42 result in
`corrected_runs/layerwise_style_orbit_v1/`. It tests only the final
language-layer prompt state (`llm_27_prompt`) and does not search across
layers.

Five paired lineages were evaluated: the exploratory seed 42 plus confirmatory
seeds 7, 19, 123, and 2027. Seed 42 is shown but excluded from confirmatory
inference because it selected the endpoint. Within each new seed, the matched and image-permuted
models use the same 2,048 source records, initialization, optimizer, ordering,
250-step budget, and trainable visual-merger parameters. The four new
permutations have distinct hashes, zero fixed pairs, and zero same-PMC/figure
pairs.

All lineages use the same 40 exposed frontal MIMIC development images
(38 patients), six fixed PubMedVision Fourier styles, report prompt, real
image, and null image. The primary endpoint was fixed before looking at the
new lineages:

\[
\kappa_{27}(x)=
\frac{\operatorname{RMS}_s[h_{27}(T_sx)-h_{27}(x)]}
{\operatorname{RMS}[h_{27}(x)-h_{27}(x_\varnothing)]}.
\]

## Result

The seed-42 contraction did **not** replicate in the four new lineages:

- negative matched-minus-permuted effect in 2/4 new lineages;
- mean confirmatory seed-level median relative effect: **+0.33%**;
- seed-by-patient crossed bootstrap 95% CI:
  **[-1.09%, +1.67%]**;
- seed-level t 95% CI: **[-1.65%, +2.31%]**;
- one-sided exact sign-test \(p=0.6875\).

The seed-42 estimate (-5.48%) was much larger than the new seed estimates
(-0.60%, +1.92%, -0.72%, +0.72%). Style drift and real-null leverage were
also inconsistent. A discovery-inclusive five-lineage summary is retained in
`summary.json` for transparency but is not the confirmatory result.

Therefore ordinary correct image--text pairing is **not shown** to produce a
training-stable late-fusion style-orbit contraction under this protocol.
The earlier seed-42 result must be treated as a lineage-specific exploratory
finding, not a mechanism or method target.

Raw activation arrays and checkpoints remain outside Git under
`/root/autodl-fs/data/dbw/anchor_alignment_contraction_v1/`. The compact
`summary.json` records hashes and fingerprints.
