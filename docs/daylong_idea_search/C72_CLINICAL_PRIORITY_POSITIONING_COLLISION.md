# C72 — Clinical Priority Positioning collision audit

## Decision

**NO-GO as an ICLR method; do not interrupt the baseline queue.** The proposed
operation preserves the projected visual-token multiset and sorts it so that
high frozen-specialist CAM values occupy later visual addresses, nearer the
following question tokens. It is executable and has a clean control panel, but
its irreducible novelty is only the source of a known token-reordering score.

## Proposed primitive

For projected visual tokens `v_1,...,v_n`, specialist relevance `s_i`, and a
decoder positional-influence surrogate `w_j`, choose a permutation `pi` and
feed `v_{pi(1)},...,v_{pi(n)}`. If the target were the separable linear
surrogate `sum_i s_i w_{pi(i)}` and `w` increased monotonically toward the
question, the rearrangement inequality would make similarly sorted `s` and `w`
optimal. This fact cannot serve as a theorem for a real VLM: RoPE attention is
a sum of oscillatory phase terms, visual tokens interact nonlinearly across
layers, and the pretrained projector relies on spatial index structure.

## Exact collisions and failure modes

- CCA (NeurIPS 2024) already makes RoPE distance decay the mechanism and
  reorders visual positions to shorten their distance to instruction tokens,
  together with a causal mask intended to preserve spatial locality.
- DAPE-BR (Findings EMNLP 2025) explicitly identifies spatial aliasing and
  pretrained visual--text index mismatch caused by CCA-style reordering.
- Attention Calibration (2025) reports architecture-specific spatial
  perception biases, so “later is stronger” is not a universal operating law.
- Recent gradient/saliency-guided visual-token reordering further occupies the
  only remaining system delta: replacing a fixed spatial score by an adaptive
  relevance score.

Clinical CAM sorting additionally destroys 2-D neighbourhood and laterality
relations. Hence a positive canary would mean only that one architecture
benefits from one saliency-conditioned reordering, not that a new computation
primitive has been discovered.

## Work completed without GPU

- A frozen 70-claim panel: seven findings, per finding four small unanimous
  positives, two large unanimous positives, and four 0/3 negatives.
- Exact TorchXRayVision CAMs satisfying
  `logit = classifier_bias + spatial_mean(CAM)`.
- Native / clinical-priority / reverse-priority / shuffled-priority runtime
  with token-count and norm-multiset invariants.

Artifacts:

- `anchor/corrected_sgta/prepare_clinical_priority_positioning_v1.py`
- `anchor/corrected_sgta/run_clinical_priority_positioning_v1.py`
- `corrected_runs/daylong_idea_search_v1/clinical_priority_positioning_v1/panel_cams.npz`

These remain a reusable baseline or diagnostic, not a paper contribution.
