# ANCHOR Final Direction: Source-Guided Output-Path DG

## Decision

The paper should not present target thresholds, source-margin cutoffs, or report-length medians as the final ANCHOR method. Those observations are useful diagnostics, but they are too brittle and too close to task-specific calibration tricks to satisfy the intended contribution.

The final research direction is instead:

> Unknown medical domain shift is probed by source-style counterfactual views, but the actionable correction is defined on the VLM's complete output evidence path, not on single answer logits or hand-chosen length statistics.

This keeps the original DG motivation while avoiding a naive threshold story.

## Mathematical Object

For an image-question pair `(x, q)` and a generated sentence `y=(y_1,...,y_T)`, define the output evidence path

```text
u_t(y) = [ -log p_theta(y_t | x,q,y_<t),
           log p_theta(y_t | x,q,y_<t) - log p_theta(y_t | x_null,q,y_<t),
           t / T_max,
           1[y_t = EOS] ].
```

The source center is not a pixel or pooled-feature centroid. It is the source-success path geometry: the empirical distribution of `u_1:T` from source-domain outputs that are correct or clinically reliable.

A candidate sentence is scored by

```text
E(y) = sequence_NLL(y) + lambda * D_path(u(y), M_source_success).
```

`D_path` is implemented with the existing source-success trajectory bank and sliced-Wasserstein / nearest-manifold machinery. This is a continuous path energy, not a threshold and not a target-domain statistic.

## Role of FedDG / SGTA Views

FedDG-style transforms remain important, but their role is reduced to a counterfactual probe:

```text
x -> { original, source-spectrum, low-frequency, gamma/window }
```

These views test whether acquisition-style perturbations expose unstable output paths. They are not assumed to directly improve accuracy. If the style orbit has no candidate oracle headroom, the view route must stop.

## Current Gate Evidence

Implemented runner:

```text
anchor/corrected_sgta/run_anchor_flow_sgta_gate.py
```

Implemented analyzer:

```text
anchor/corrected_sgta/analyze_anchor_flow_sgta_gate.py
```

Current gate result:

```text
corrected_runs/final_anchor_flow_sgta_gate_v1/gate_analysis.json
```

Summary:

| Split | n | Greedy | ANCHOR-Flow | Oracle | View disagreement | Oracle headroom | Continue |
|---|---:|---:|---:|---:|---:|---:|---|
| RULE/MIMIC CE strong SGTA | 8 | 0.750 | 0.750 | 0.750 | 0.000 | 0.000 | no |
| CheXpert OE strong SGTA | 4 | 0.076 ROUGE-L | 0.076 | 0.076 | 0.250 | 0.000 | no |

Interpretation: current SGTA/FedDG views do not create useful full-sentence candidate headroom on this pilot. CE outputs are identical across style views. OE sometimes changes wording, but not utility.

## Paper Framing Consequence

Do not claim that Fourier/FedDG view ensemble is the final effective module unless a future gate shows candidate headroom.

Safe framing:

> Source-style counterfactuals reveal when visual-domain perturbations fail to move the VLM's generation dynamics; ANCHOR therefore anchors the complete evidence path rather than the image itself.

This is consistent with the title/abstract with only small changes:

- Replace “adaptively aggregates source-guided views” with “uses source-guided views as counterfactual probes and calibrates complete evidence paths”.
- Replace “reducing visual distribution discrepancy” with “correcting source-conditioned evidence-to-generation mismatch under visual domain shift”.

## Next Executable Step

The next method attempt should not tune thresholds or report lengths. It should implement direct online path-energy decoding:

1. During generation, compute top-M token path features for the current partial sentence.
2. Penalize tokens whose next-step evidence state has low density under source-success paths.
3. Decode one complete sentence with the adjusted distribution.
4. Evaluate CE and OE with the same generated-sentence protocol.

This keeps the method simple: one source-success path bank, one continuous energy, one greedy generation pass with source-path shaping.

## Non-Claims

- The current pilot does not prove FedDG views improve accuracy.
- The current pilot does not prove clinical OE factuality improves.
- Source-margin and word-center results are not the final method.
- Yes/No logits are not a valid final prediction interface for this paper direction.
