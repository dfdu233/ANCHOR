# C75 — The full specialist state is not a VLM-agnostic clinical operator

## Question and strict transfer test

C74 showed that a finding-conditioned linear readout of all 18 XRV disease
logits is much stronger than injecting only the matching disease score.  This
audit asks whether that 18D readout is a reusable clinical operator or merely a
model-specific stacker.

On a source VLM's frozen development split, fit

`p(y_c=1) = sigmoid(a_c + b m + v_c^T s)`,

where `m` is the VLM claim margin, `s` is the standardised 18D XRV state, and
`v_c` is the finding-conditioned specialist direction.  When transferring to
the other VLM, freeze every coordinate of `v_c`.  Target development labels
may fit only finding intercepts, one VLM-margin coefficient, and either one
global or seven finding-specific scalar multipliers on the one-dimensional
score `v_c^T s`.  They cannot relearn an 18D direction.

Both VLMs use the same 280 development and 840 image-disjoint confirmation
claims over seven unanimous VinDr findings.  The audit is CPU-only and does not
read, pause, or modify the baseline GPU process.

The preregistered gate is deliberately incremental: in both directions the
transferred direction must retain at least 70% of the **native full-state gain
beyond target-scalar fusion**, and that increment over target-scalar fusion
must have an image-cluster bootstrap 95% CI above zero.

## Results

| source -> target | VLM | target scalar | target-native full18 | transferred full18 direction | retained total gain | retained increment beyond scalar |
|---|---:|---:|---:|---:|---:|---:|
| Huatuo -> Hulu | .8606 | .8708 | .8888 | .8758 | 53.9% | **27.7%** |
| Hulu -> Huatuo | .7667 | .8264 | .8633 | .8321 | 67.7% | **15.3%** |

The source and target directions look superficially similar (mean cosine
`.939`; every finding is `.930`--`.960`).  Nevertheless, the transferable
part adds only `+.0050` AUROC over target-scalar fusion on Hulu and `+.0056` on
Huatuo, compared with native full-state increments of `+.0179` and `+.0369`.
Under 5,000 image-cluster bootstrap draws, neither transferred increment is
distinguishable from zero: Hulu `[-.0216,+.0304]`, Huatuo
`[-.0297,+.0391]`.  The target-native increment is significant on Huatuo
`[+.0076,+.0658]`, but not on Hulu `[-.0040,+.0392]`.

## Decision

**NO-GO: model-specific stacking.**  Neither direction retains 70% of the
increment that made full-state fusion interesting.  A common clinical
direction exists, but almost all of its transferable benefit is already
captured by the ordinary matching-disease scalar.  The extra off-diagonal
18D gain depends on which VLM is being corrected and cannot support the claim
of a universal "clinical differential operator."

This does not negate the narrower C74 observation that the complete specialist
state can improve a particular VLM.  It changes its interpretation: the 18D
state is a useful supervised fusion feature, not yet a simple general-purpose
hallucination-mitigation primitive.

## Artifacts

* `anchor/corrected_sgta/screen_xrv_clinical_operator_transfer_v1.py`
* `corrected_runs/daylong_idea_search_v1/xrv_clinical_operator_transfer_v1/result.json`
