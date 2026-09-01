# CIPCA: frozen research contract

Date: 2026-08-02  
Status: **superseded before GPU execution** by
`ASCC_FROZEN_RESEARCH_CONTRACT_20260802.md`.  The free-generation pair design
was KILLed because it conditions on post-prompt outputs and confounds marker
replacement with differential-diagnosis deletion.  Its artifacts remain only
as a reproducible negative audit.

## 1. Frozen question

The project no longer asks whether prompts generally change medical-VLM
answers.  That problem is occupied by prompt-induced hallucination, question
framing and calibration work.  The surviving question is narrower:

> After an image-grounded observation and diagnosis have already been selected,
> can the pragmatic goal of the question increase the linguistic commitment of
> that unchanged clinical proposition, even when the question explicitly
> licenses uncertainty?

The provisional name is **Claim-Invariant Pragmatic Commitment Amplification
(CIPCA)**.  A positive result requires a post-selection mechanism that is
separable from both visual-evidence change and the generic verbal-uncertainty
feature.  “Prompt wording matters,” “medical VLMs are overconfident,” and “an
uncertainty direction exists” are explicitly barred claims.

## 2. Discovery observation and its evidential ceiling

In the completed 200-image Huatuo generation cache, four images independently
produced the same normalized observation and diagnosis under both frozen
prompts:

```text
neutral:     bilateral patchy opacities, suggesting possible pneumonia
existential: bilateral patchy opacities, which are consistent with pneumonia
```

The local diagnostic sentences differ by only 1--2 whitespace words.  All four
surface shifts point from the tentative class to the higher-commitment class,
but the exact two-sided sign-test value is 0.125.  This is a hypothesis generator,
not a result.  The discovery images are excluded from every confirmatory split.

Reproducible artifacts:

- `corrected_runs/specificity_ratchet/pragmatic_commitment_pairs_audit_v1.json`
- `corrected_runs/specificity_ratchet/diagnostic_completion_substrate_audit_v1.json`
- `anchor/corrected_sgta/audit_pragmatic_commitment_pairs_v1.py`
- `anchor/corrected_sgta/audit_diagnostic_completion_substrate_v1.py`

The automatic `uncertain/definite` labels are surface diagnostics only.  They
cannot establish a clinical certainty order.  Formal scoring requires blinded
radiologist judgments; the recent LELEX consensus and Lunguage++ uncertainty
ranking are external priors, not replacement truth.

## 3. Confirmatory substrate frozen before new outputs

VinDr-CXR directly provides three independent image-level reader labels for
both Lung Opacity and Pneumonia.  The estimand is therefore a radiograph-
attributable diagnostic impression, not patient-level etiology or pathology
truth.

Selection is all exact R8/R9/R10-panel images satisfying
`Lung Opacity >= 1/3 OR Pneumonia >= 1/3`, after excluding all 200 discovery
images.  Membership never uses model output.  Within every `(opacity vote,
pneumonia vote)` stratum, image IDs are deterministically alternated into
image-disjoint dev/test splits.

- selected images: 1,393;
- dev/test: 698/695;
- prompt conditions per image: neutral and existential;
- total native generations: 2,786;
- substrate fingerprint:
  `a862dab832f8866fd71750aeaadf879d45168abbe13284ca020e1d3b02c9114c`.

Authoritative files:

- `corrected_runs/pragmatic_commitment/confirmatory_substrate_v1/substrate_config.json`
- `corrected_runs/pragmatic_commitment/confirmatory_substrate_v1/selected_manifest.jsonl`
- `anchor/corrected_sgta/prepare_pragmatic_commitment_confirmatory_v1.py`
- `anchor/corrected_sgta/run_huatuo_pragmatic_commitment_generation_v1.py`

The two prompts are byte-for-byte identical to the discovery run.  Both say
“State uncertainty explicitly rather than guessing”; the existential prompt
also explicitly asks for “present or uncertain abnormalities.”

## 4. Behavioral admission

An admitted pair must have, under both prompts:

1. the same image;
2. the same VinDr parent label (`Lung Opacity`);
3. the same VinDr child diagnosis (`Pneumonia`);
4. the same normalized observation prefix;
5. a local diagnostic-sentence word-count gap no greater than four.

Two radiologists, blinded to prompt condition and reader votes, independently
rank the commitment conveyed by each sentence.  Disagreement is adjudicated by
a third radiologist.  The primary paired outcome is

\[
P(\text{existential more committed})-
P(\text{neutral more committed}).
\]

The behavioral gate requires, separately on dev and untouched test:

- at least 20 admitted image clusters;
- image-cluster bootstrap 95% CI lower bound above zero;
- the shift after adjustment for local sentence length, full-answer length,
  claim count and hedge-token count;
- the direction to remain within the 0/3, 1/3, 2/3 and 3/3 child-reader strata
  rather than being created by a different support mixture.

Failure of either split terminates CIPCA.  Thresholds, phrase lexicons and
identity matching are not relaxed after generation.

## 5. Three competing mechanisms

### A. Framing-induced visual blindness

The prompt changes attention to the image or the visual support for the
observation/diagnosis.  This is the mechanism studied by Tinted Frames and is
not a new headline.  Test it by teacher-forcing the same complete observation
and diagnosis prefix under both prompts and measuring projector/decoder visual
attention, image-swap sensitivity and diagnosis-token evidence.

If visual evidence changes enough to explain the commitment marker, CIPCA is
KILLed as a medical boundary replication.

### B. Register, length or report-template substitution

The model merely switches genres or sentence templates.  Freeze the complete
answer prefix and compare a predeclared equal-register marker set, including
`possible`, `suggestive of`, `concerning for`, and `consistent with`.  Match
local length and syntax; include full-answer length, claim count and token
frequency as nuisances.

If the effect disappears, CIPCA is KILLed as template selection.

### C. Post-selection pragmatic commitment

Observation and diagnosis evidence remain stable, while only the marker
distribution changes after content selection.  Neutral-to-existential
activation interchange at the marker position must reproduce the commitment
shift; the reverse patch must restore hedging.  Observation identity,
diagnosis identity, polarity and claim count must remain unchanged.

This is the only mechanism that allows CIPCA to proceed.

## 6. Mandatory generic-VUF separation

EMNLP 2025 already identifies a single linear verbal-uncertainty feature and
uses activation intervention to reduce confident hallucinations.  Therefore a
medical “uncertainty direction” is not novel.

Estimate the generic verbal-uncertainty feature on a disjoint, non-medical
calibration corpus.  For every layer, residualize the prompt-paired CIPCA
contrast against that feature and matched random directions.  The residual
must:

- retain prompt-condition information beyond the generic VUF;
- predict physician-ranked within-claim commitment beyond reader votes,
  diagnosis logits and sentence features;
- causally change only commitment markers when patched;
- reproduce in at least two medical VLM architectures.

If orthogonalization removes the effect, the project becomes a domain audit of
the known VUF and is not an ICLR main contribution.

## 7. Conditional mitigation boundary

No mitigation is authorized before the behavioral and causal gates pass.
Direct neutral-minus-existential contrastive decoding is barred because it is
method-equivalent to existing instruction/contrastive decoding.

The only admissible future method is a **claim-locked** correction derived from
the causal residual: intervene at commitment realization after the observation,
diagnosis and polarity are fixed, while preserving activation norm and the
generic VUF component.  It must not delete, swap or negate a claim.

Success requires:

- at least 20% relative reduction in physician-rated overcommitment;
- unchanged diagnosis identity, polarity and positive-claim count by
  construction;
- no increase in finding or diagnosis omission;
- clear 3/3-reader cases within 1 percentage point of baseline;
- superiority to temperature scaling, prompt normalization, generic VUF
  steering, random/norm-matched steering and instruction contrastive decoding;
- OE/report replication with fixed claim coverage in at least two models.

## 8. Collision boundary

The paper must distinguish CIPCA from:

- ACL 2026 prompt-induced hallucination heads (false/leading prompts);
- Tinted Frames (framing-dependent visual attention);
- EMNLP 2025 generic verbal-uncertainty feature;
- generic confidence--decisiveness and VLM calibration work;
- instruction/visual contrastive decoding;
- Findings-to-Impression formal verification and post-hoc filtering.

The novelty cell is the conjunction of **same clinical claim, same image,
reader-grounded support, uncertainty explicitly licensed, post-selection
localization, generic-VUF orthogonality, and content-preserving causal change**.
Removing any one of those qualifiers collapses the project into prior work.
