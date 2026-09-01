# Directional Uncertainty Is Not a Four-Token Reader Scale

**Decision:** 2026-08-03  
**Status:** hard NO-GO for a VinDr four-verbalizer mechanism screen; retained as
an evaluation invariant only.

## Candidate that was audited

The candidate arose from an asymmetric result in the construct-corrected ASCC
screen: the neutral `uncertain` realization behaved differently near the
`0/3 <-> 1/3` and `2/3 <-> 3/3` reader boundaries.  A tempting repair is to
replace one neutral state with four ordered surface realizations:

```text
absent < uncertain unlikely < uncertain likely < present
```

and to interpret VinDr vote bins `0/3, 1/3, 2/3, 3/3` as those four states.
That interpretation is not construct-valid and is not a novel research
question.

## Fatal collision

CheXpert's board-certified radiologist validation protocol already used the
four categories `present`, `uncertain likely`, `uncertain unlikely`, and
`absent`.  It then binarized the likely pair as positive and the unlikely pair
as negative.  The same paper also showed that the meaning of an uncertain
report label is finding dependent: treating uncertain mentions as positive was
best for atelectasis and edema, whereas consolidation behaved differently.

Primary source:

- [CheXpert, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/3834)

Therefore a four-label radiology certainty scale, uncertainty directionality,
and finding-specific uncertainty semantics are all prior art.  A medical VLM
teacher-forcing experiment over the same four words would at most measure
verbalizer admission or reproduce this established ordinal lexicon.

## Fatal construct mismatch

VinDr supplies three independent binary reader decisions for a fixed panel.
Its four vote counts are panel support states, not four within-reader certainty
judgments.  In particular:

```text
1/3 positive readers != one reader saying uncertain unlikely
2/3 positive readers != one reader saying uncertain likely
```

The equality would conflate disagreement across people with the probability
communicated by one person.  This distinction is explicitly required by the
existing Virtual-Reader Sufficiency protocol, which reconstructs only the
fixed pseudonymous panel and forbids calling its target within-reader
uncertainty or clarity.

The broader statistical literature already provides methods for evaluating
class-probability estimates under expert disagreement; that ingredient is also
occupied rather than novel:

- [Diagnostic Uncertainty Calibration, AISTATS 2021](https://proceedings.mlr.press/v130/mimori21a.html)

## Why paraphrase controls cannot rescue it

Matched phrases such as `unlikely`, `less likely`, `possible`, and `probably`
would reduce token-prior confounding, but they cannot create the missing
construct link between panel votes and individual linguistic certainty.
Radiology certainty is contextual: the same cue can refer to a finding, a
differential diagnosis, image quality, temporal comparison, or an inability to
evaluate.  A phrase-only ordinal likelihood is consequently neither clinical
truth nor a mechanism.

## Frozen decision

Do not:

1. build a `0/3 -> absent, 1/3 -> unlikely, 2/3 -> possible, 3/3 -> present`
   target;
2. spend GPU time on a four-token admission screen using VinDr votes;
3. claim a missing fourth state or directional uncertainty as novelty;
4. use a global certainty lexicon to relabel open-ended claims.

Retain two valid consequences:

1. OE claims continue to serialize `prediction_polarity` and
   `prediction_uncertainty` independently.  A hedged positive is still a
   positive-content claim and cannot disappear from fabrication or matched
   coverage.
2. Any future certainty analysis must be finding-, referent-, and
   construction-aware.  The clinical physician review may test whether a
   method improves certainty appropriateness, but no automatic four-word scale
   may define that truth.

## What would reopen a mechanism question

Only a new dataset in which the same radiologists independently provide both
binary finding decisions and calibrated certainty categories on the same
images could test a cross-level mechanism.  Even then the unique question must
concern a causal VLM computation not already explained by ordinal calibration,
lexical priors, or finding-specific report conventions.  The current VinDr
substrate cannot do this.

