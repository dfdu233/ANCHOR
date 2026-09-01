# Source/prevalence pseudo-evidence: identification audit

Decision date: 2026-08-03

## Decision

**NO-GO for the current pretrained Huatuo, Hulu, and LLaVA-Med assets.** Do not
launch a source-style or prevalence-based decoder and do not describe image
restyling as a causal intervention on a model's training source.

The scientifically interesting question is real:

> When visual evidence is weak, does a medical VLM substitute a disease prior
> learned from a particular training source and express that prior as if it
> were image evidence?

But the current proposed experiment cannot identify that mechanism. An image
render transformation acts on the test image. Training-source membership and
training-set disease prevalence are properties of the data-generating and
optimization process. The first is not an intervention on either of the
latter. Correlated output changes therefore cannot distinguish source-prior
recall from ordinary low-margin sensitivity, clinical-visibility change,
token prior, acquisition shortcut, or generic cross-modal conflict.

## Why this is not merely a low-power objection

### 1. The causal variable is unobserved

For the deployed checkpoints, no verified per-training-example source ledger
is available locally. A latent "source-domain center" cannot be assigned to a
test image from its style without circularly defining the cause from the same
representation or output used to establish the effect.

Known medical-imaging shortcut results make such effects plausible, not
identified. Jabbour et al. directly manipulated spurious class skew in chest
X-ray training data and showed models exploit correlated attributes
(<https://proceedings.mlr.press/v126/jabbour20a.html>). Lotter et al. showed
that acquisition and processing parameters affect learned demographic signals
and downstream bias (<https://www.nature.com/articles/s41467-024-52003-3>).
Those studies used known labels or controlled training distributions. They do
not license inferring a hidden VLM training source from a transformed image.

### 2. The repository already rejects the easy observable signatures

- In the frozen 128-case style phenomenon test, all three mild-transform flip
  rates were below 5%; original margin predicted errors better than raw style
  drift. The additive source-style null was unreliable, and the apparent
  evidence-gain correlation vanished after removing mathematical coupling.
- The later 160-claim VinDr DICOM-render experiment completed without runtime
  errors but passed 0/4 frozen findings. Descriptive orbit flips did not yield
  a held-out common display/source-center direction.
- The three-model SLAKE prior-titration screen crossed stated 10/50/90%
  background probabilities. Huatuo, Hulu, and LLaVA-Med behaved differently,
  and the predeclared worst-prior score failed its matched-coverage gate in all
  three. This does not rule out implicit training prevalence, but it rules out
  a universal additive evidence-update law and the obvious low-cost decoder.

The generic premise is also crowded. SumGD explicitly attributes LVLM
hallucination to increasing language-prior reliance and reports that direct
output calibration can degrade text quality
(<https://aclanthology.org/2025.findings-naacl.235/>). Treble Counterfactual
VLMs already decomposes vision, language, and cross-modal causal effects
(<https://aclanthology.org/2025.findings-emnlp.1000/>). A medical dataset swap
plus an unobserved "source" label would not clear the novelty or mechanism
gate.

## Falsifiable identification requirements for reopening

The hypothesis may reopen only under one of these designs:

1. **Controlled training intervention.** Train architecture-matched medical
   VLMs on the same images and reports while independently changing finding
   prevalence within each known source. Evaluate a held-out, prevalence-
   balanced multi-site test set. The causal estimand is a difference across
   trained models, not across test-image styles.
2. **Auditable mixture model.** Use a checkpoint with a complete per-example
   training-source ledger and source-specific prevalence. Fit a source
   posterior from acquisition metadata unavailable to the answer decoder, then
   test preregistered source-by-evidence interactions on external sites. A
   source probe trained on decoder outputs is forbidden.
3. **Natural experiment with invariance.** A known protocol change must alter
   source prevalence while leaving acquisition and reporting policy
   sufficiently stable, or vice versa. Both the intervention and invariance
   assumptions require independent evidence.

All designs must additionally:

- hold finding ontology, patient split, acquisition view, and reporting policy
  fixed or model them explicitly;
- use independent clinical support, not a benchmark answer, as the outcome;
- compare against clean margin, temperature, test-time prevalence calibration,
  acquisition metadata, and image/null/shuffle controls;
- demonstrate incremental held-out prediction of unsupported commitment in at
  least two model families and two known sources;
- reject any gain explained by shorter answers, claim deletion, hedging,
  refusal, or omission.

## Current consequence

The 00:26 status suggestion to move directly from failed ASCC to
"source/prevalence prototypes acting as pseudo-evidence" is superseded. The
question remains a valuable future controlled-training study, but the current
repository lacks the intervention and source ledger required to answer it.
Available GPU must not be spent on another style bank or prior decoder.

The active evidence path remains the real blinded physician OE review. Further
mechanism exploration must either use an already identifiable VinDr construct
or first acquire a dataset/model substrate that makes its causal variable
observable.
