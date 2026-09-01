# AR-SoS VinDr substrate admission v1

## Decision

The exact VinDr `R8/R9/R10` eight-finding substrate is a **no-go for the
original decisive AR-SoS experiment**.  It is large enough to establish that
some A--C pairs co-occur in the development split, but it is not large enough
to supply six distinct A--B pairs with forty locked-confirmation images meeting
`A=3/3, B=3/3, C=0/3`.  No VLM should be run on this substrate and no formal
mechanism manifest is emitted.

This is a substrate failure, not evidence against or for Autoregressive
Satisfaction of Search.

The stop is explicit under the fatal-flaw taxonomy:

- **F6 / construct admission fails:** VinDr contains labels but no native
  report sentences.  In particular, `same_support_other_image_wording` cannot
  be certified as expressing only A without a two-clinician wording review.
  Model/automatic phrases may be candidate stimuli but cannot close this gate.
- **F7 / data cardinality fails:** even exhaustive label-only enumeration has
  zero A/B pairs at the forty-image confirmation quota (maximum 39).

Either failure prohibits a GPU run; both are present.

## Truth and split contract

- Clinical states are the three independent official readers `R8`, `R9`, and
  `R10`.  `A=3/3`, `B=3/3`, and `C=0/3` are preserved as vote vectors; a
  majority label is never substituted.
- The fixed eight findings are aortic enlargement, cardiomegaly, lung opacity,
  nodule/mass, other lesion, pleural effusion, pleural thickening, and pulmonary
  fibrosis.
- Global image hashing with seed 42 yields 1,072 pilot, 1,061 development, and
  3,368 confirmation images.  Every image has exactly eight reference rows.
- All 56 ordered A-to-C co-occurrence tests are fitted on development only.
  An association requires at least ten development `A3/C3` images, smoothed
  lift at least 2, and one-sided Fisher Bonferroni p at most 0.05.
- A/B/C triple identities are ranked and frozen on development only.  Opening
  confirmation is limited to an availability count for those frozen identities;
  it cannot change their order or identity.  No model output is inspected.
- Generated phrases, regexes, automatic labelers, and LLMs never define A, B,
  C, support, or clinical correctness.

The executable audit is
`anchor/corrected_sgta/audit_ar_sos_vindr_substrate_v1.py`; its no-go result is
stored outside Git under
`corrected_runs/vindr_v2/ar_sos_substrate_audit_v1/substrate_audit.json`.

## Observed feasibility

Eight ordered A-to-C directions pass the development-only association gate:

| A | C | dev A3/C3 | smoothed lift |
|---|---|---:|---:|
| aortic enlargement | cardiomegaly | 137 | 7.63 |
| cardiomegaly | aortic enlargement | 137 | 4.24 |
| pulmonary fibrosis | pleural thickening | 23 | 7.16 |
| pleural thickening | pulmonary fibrosis | 23 | 6.37 |
| pleural effusion | lung opacity | 18 | 71.23 |
| lung opacity | pleural effusion | 18 | 25.12 |
| pleural thickening | pleural effusion | 18 | 11.23 |
| pleural effusion | pleural thickening | 18 | 10.89 |

The dev-only ranking freezes only five distinct A--B pairs with at least eight
dev targets.  Their dev/locked-confirmation counts are respectively `13/19`,
`10/39`, `9/29`, `9/15`, and `8/20`; therefore **0/5** reaches forty.  This
passes the weak “at least four pair identities exist” screen on dev, but fails
both the original six-pair design and the minimum four adequately powered
confirmation-pair gate.  Across all 48 strong-association A/B/C combinations—not
only the five frozen ones—the maximum confirmation count is 39 and zero reaches
40.  The previous
claim that the eight-finding ontology could supply `6 x 40` images was an
unverified cardinality assumption and is now withdrawn.

## Prefix intervention contract for a future substrate

The four conditions remain frozen, but their literal strings must be admitted
before a future GPU run:

1. `supported_A`: a current-image prefix expressing exactly the unanimous
   finding A;
2. `unsupported_A_prime`: a same-image prefix expressing a distinct `0/3`
   finding A-prime;
3. `clinical_neutral`: a radiology-register sentence expressing no finding;
4. `same_support_other_image_wording`: wording sourced from a disjoint A=`3/3`
   donor image and physician-admitted as expressing A without B or C.

Model output may propose candidate wording, but two blinded clinical reviewers
must establish the semantic contents.  Thus wording is an intervention, never
truth.  For each model tokenizer, the existing exact-serialization and offset
logic in `specificity_ratchet_teacher_forcing_v1.py` should be reused.  A row is
admitted only if all four assistant prefixes have exactly the same contextual
token count and put the first B/C target token at exactly the same serialized
position.  No length caliper, standalone tokenization, or post-hoc padding is
allowed.  Failure to find an exact four-way match excludes the row.

For each prefix `p`, score present-versus-absent teacher-forced margins for B
and C on the same image:

`M_q(p) = mean_logP(q_present | x,p) - mean_logP(q_absent | x,p)`.

The primary bivariate estimand is evaluated as an intersection-union test:

- B suppression: `M_B(supported_A) - M_B(clinical_neutral) < 0`;
- C amplification: `M_C(supported_A) - M_C(clinical_neutral) > 0`.

Both directions must pass; their average cannot cancel one failure.  Report
the same contrasts for actual image minus same-support image swap and actual
image minus text-only input.  The teacher-forcing adapter must additionally
record complete target NLL, target token IDs/offsets, prefix token count, target
start position, visible character length, layer IDs, activation norms, and the
standard-forward final-logit equality proof.

## Required controls and kill gates

- Reverse A and B while retaining the same image and triple; add a
  non-clinical same-length position control.
- Match or regress clean B/C margin, finding prevalence, dev A--C and A--B
  co-occurrence, target/prefix token frequency, prefix position, visible length,
  and bbox area.  Bboxes are salience covariates, never truth.
- Include same-support image swap, text-only, random donor wording, random
  layer/state direction, norm-matched patch, and length/position permutations.
- Use image-cluster bootstrap and pair leave-one-out.  Confirmation is tested
  once, after layer/prefix decisions are frozen on development.
- At least four of six A--B pairs in each of two model families must show both
  B suppression and C amplification with 95% CIs excluding zero.  Grouped-CV
  incremental AUROC over position/length/co-occurrence/frequency/clean-margin
  controls must be at least 0.03.
- Same-support-other-image wording being equivalent to supported-current-image
  wording, or a comparable effect in text-only input, diagnoses a language
  prefix prior and kills the image-grounded AR-SoS claim.
- Prefix-end state patching must selectively restore B and reduce C without
  changing A identity/polarity by more than 1%.  Only then may Claim-Boundary
  State Reset be evaluated in OE; fixed-K hallucination must fall by at least
  20% without omission or rare-finding recall loss.

## GPU budget after a future admission

Do not reserve GPU time for current VinDr.  A passing substrate would require
`240 images x 4 prefixes x 2 models = 1,920` primary teacher-forced image
forwards.  Same-support swap and text-only controls bring the screening stage
to 5,760 forwards; bidirectional A/B order doubles it to 11,520.  Run only the
two best dev-frozen layers for causal patching, budgeted at a further 3,840
forwards.  Expected total is 6--12 GPU-hours depending on model architecture
and cache reuse, followed by OE only if every causal gate passes.
