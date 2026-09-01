# Specificity Ratchet parent-state identifiability audit

Date: 2026-08-03  
Decision: **NO-GO for calling the current estimator a parent-to-child crossing**

## Outcome-blind question

The existing layer curve scores added-constraint tokens against matched tokens.
It can show late constraint amplification, but it identifies a crossing only if
the frozen natural answer first contains an observable parent state and later
adds the proposed constraint. This audit read only
`candidates.blinded.jsonl`; it did not open reviewer or adjudication files and
did not assign visual support.

The automatic strict criterion is deliberately conservative: the exact parent
proposal must occur in the observed answer and close with sentence-final
punctuation before the earliest exact added-constraint character. A deleted or
substituted parent is counterfactual. An unfinished prefix such as “the opacity
is” is observable text, but not a completed parent claim.

## Result on the frozen 127-edge / 70-case pack

| Surface state | Edges | Interpretation |
|---|---:|---|
| Counterfactual parent only | 76 | The parent was produced by deleting a modifier such as laterality or size; it never occurred before the constraint. |
| Exact but sentence-unclosed parent prefix | 27 | A prefix exists, but automatic evidence does not establish a completed parent claim. |
| Strict sentence-closed parent before constraint | 24 | The only mechanically defensible parent-before-constraint subset. |

The strict subset contains 22 cases and only two edge types (etiology and
subtype). Under the already frozen image-disjoint split it has 8 dev cases and
14 test cases. It contains **zero repeated exact-constraint blocks** in dev and
zero in test, versus the frozen minimum of ten per split. The pack also lacks a
pre-frozen independent `semantic_block_id`; semantic equivalence cannot be
invented after seeing the candidates. Even before physician exclusions, this
substrate therefore cannot issue a Construct–Prevalence Certificate.

The relaxed exact-surface criterion does not rescue it: all 51 such edges are
connector edges, their 51 exact constraint strings are unique, and only two
edge types remain. The 76 laterality/size modifier edges that provide most of
the repeated vocabulary are exactly the edges whose parent is counterfactually
created by deletion.

## Scientific consequence

For this pack, any favorable curve from the current constraint-versus-matched
token estimator must be named **late constraint amplification**. It must not be
reported as a parent-to-child crossing, commitment transition, or ratchet.
Likewise, suppressing the scored modifier is not a selective rescue of a
preserved parent unless parent identity is independently observed.

The current pack may still serve as a bounded physician construct pilot. It
must not authorize the scientific GPU replay. A future crossing experiment
needs either:

1. natural outputs with a completed, physician-admitted parent claim before an
   adjacent observable constraint, plus pre-frozen semantic block IDs and at
   least ten repeated blocks in each split across at least three edge types; or
2. a separately frozen parent-identity readout validated on held-out examples,
   with parent stability tested jointly with constraint reversal.

Until one of these exists, the mechanism contribution is not identified.
