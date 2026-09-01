# CECD Stage-1 behavioral collision gate: outcome-blind audit

**Date:** 2026-08-03
**Scope:** protocol, source and synthetic tests only. No sealed Stage-1 outcome was
opened and no GPU/model run was started.

## Verdict

The previous analyzer was leakage-aware but no longer sufficient under the
latest collision scan. It controlled clean margin, render/prompt marginals,
three-state entropy and a full-orbit scalar, but it did **not** close prompt and
response length, generic two-axis stability, or PID-style synergy. Its
per-finding requirement used point directions rather than cluster intervals,
and its output did not state sharply enough that group-OOF dev rows are not a
locked-test confirmation.

The corrected Stage-1 is now a strict **dev-only phenomenon screen**. A pass may
authorize one separately frozen comparison, but never establishes any of the
following:

- locked-test confirmation of the behavioral phenomenon;
- a representation-level PID result;
- a CECD causal mechanism;
- full mitigation-baseline or ICLR-oral closure.

## Frozen behavioral ladder

Every predictor is fit only on the outer fold's training images. Numeric
scaling, finding/acquisition-view encoding, reader-equivalent scale and score
center are re-fit inside that fold. The held rows are therefore image-group OOF
within **dev**, not paper test data.

The frozen nested ladder is:

1. clean reader-oriented margin + three-state entropy + input prompt length +
   fixed one-token response length;
2. render and prompt harmful main effects and both marginal RMS values;
3. full-orbit score plus generic visual-axis, language-axis and full-grid
   stability: slice score dispersion, entropy, probability dispersion,
   cell-to-orbit KL and orbit predictive entropy;
4. a transparent behavioral MMI PID-style control;
5. only then add signed and absolute centered mixed derivative.

For the behavioral PID control, render and prompt are uniform independent
sources and the model's Yes/No/Maybe distribution is the stochastic target:

\[
S_{\mathrm{MMI}}=I(R,P;Y)-\max\{I(R;Y),I(P;Y)\}.
\]

This is an output-distribution control. It is explicitly not hidden-state PID,
causal synergy, or a substitute for the later mechanism analysis.

The one-token CE answer contract means answer length is exactly one rather than
a varying nuisance. Prompt token count remains a fold-local numeric control.
The analyzer now rejects a non-one answer length, malformed/missing three-state
logits, or any mismatch among logits, signed score, commitment and entropy.

## Frozen pass rule

A model passes only if all conditions hold without threshold relaxation:

- centered-interaction RMS is at least 0.25 reader-equivalents and its
  image-cluster interval is above zero;
- adding the mixed derivative to the **strongest behavioral rung** improves
  OOF error AUROC by at least 0.03, with image-cluster 95% CI above zero and at
  least 95% valid bootstrap draws;
- harmful signed interaction is larger in error than correct cells with
  cluster CI above zero;
- identity-render and duplicate-prompt noise is at most one tenth of the
  clinical interaction;
- at least three of four findings independently satisfy delta-AUROC >= 0.03,
  delta-AUROC cluster CI above zero, harmful-alignment cluster CI above zero,
  and bootstrap-validity requirements;
- all four finding-specific reader-direction scales have image-cluster CI above
  zero;
- two distinct model families pass independently.

The real 160-claim screen has only 20 clear-vote orbits per finding before any
whole-orbit exclusions. Wide per-finding intervals therefore produce a strict
NO-GO/inconclusive result; the threshold must not be weakened after viewing
outcomes.

## Synthetic falsification added

The previous positive fixture contained a large true mixed derivative that
also made generic stability almost perfectly identify errors. Under the old
gate it passed; under the corrected ladder its strongest generic baseline has
AUROC above 0.99 and the CECD increment is below 0.03, so it correctly fails.
This is important: a nonzero reader-aligned factorial interaction is not CECD
evidence when ordinary inconsistency already explains it.

Additional synthetic contracts verify:

- additive grids fail;
- nonlinearity without incremental clinical-error information fails;
- one image cannot cross folds;
- invalid cells remove the complete orbit rather than being imputed;
- fixed answer length and bound readout arithmetic fail closed;
- a uniform output tensor has zero behavioral MMI synergy.

## Phenomenon gate versus method gate

The current ten-arm dual-semantics envelope contains:

- two Treble common-protocol variants, which are static/global activation
  controls but neither is an exact paper-native reproduction;
- CECD interaction projection;
- full-orbit, render-only, prompt-only, random, sign-permuted and main-effect
  controls.

It does **not** contain an official-compatible dynamic/query-adaptive or
multimodal activation baseline such as a compatible DMAS/CausalLens/HulluEdit/
CAI implementation, nor representation-level PID. Therefore an envelope win
can establish only `cecd_treble_envelope_advantage_established`. It now leaves
`cecd_causal_claim_authorized`, `full_method_gate_authorized`,
`oral_baseline_closure_authorized` and `paper_claim_authorized` false.

Before a causal or method claim, a separate locked-test stage must add:

1. the same behavioral incremental ladder on an image-disjoint locked split;
2. actual representation-level PID or an explicitly bounded nonclosure;
3. at least one official-compatible dynamic/multimodal activation baseline;
4. noising/denoising, activation-distance and PIE/interaction validity controls;
5. CECD joint-cell specificity and CE + aligned-OE replication without
   shortening, omission, refusal or blanket hedging.

## Changed sources and verification

- `anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py`
- `anchor/corrected_sgta/run_cecd_factorial_v1.py`
- `anchor/corrected_sgta/treble_collision_contract.py`
- `anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py`
- focused synthetic/contract tests under `tests/`

Outcome-blind focused regression at audit time:

```text
42 passed
```

No scientific result is inferred from this code audit.
