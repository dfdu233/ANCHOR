# Post-CEBC mechanism pivot search: **NO-PIVOT**

**Date:** 2026-08-03  
**Search horizon:** primary conference/journal papers and arXiv available through
2026-08-03  
**Mode:** outcome-blind literature and substrate audit. No sealed CECD result,
model output, checkpoint forward pass, or GPU was read.  
**Decision:** **NO-PIVOT. Keep CECD as the only conditional mainline.**

## 1. Why this search returns no replacement idea

The post-CEBC question was not “what other decoder can be tried?”, but:

> Is there a clinically important, falsifiable mechanism that is not already a
> relabeling of evidence-bounded editing, confidence calibration, visual/prior
> conflict attribution, generic steering, multi-label serialization, or template
> collapse—and that can be tested now with VinDr's independent reader vectors and
> the frozen 14-class listing substrate?

No retrieved candidate passed all four hard gates simultaneously:

1. a real local phenomenon rather than a speculative mechanism;
2. no mechanism-level collision;
3. a distinguishing falsification rather than “the metric did not improve”;
4. an endpoint that can reduce fabricated claims without increasing omission or
   merely shortening the answer.

The result is deliberately conservative. [CEBC](https://aclanthology.org/2026.acl-long.2142/)
already occupies conformally evidence-bounded minimal editing; [VLI](https://aclanthology.org/2026.acl-long.1784/)
occupies instance-specific conflict diagnosis plus bi-causal steering;
[HalluTrace](https://aclanthology.org/2026.alvr-main.29/) occupies
visual-grounding/language-prior/cross-modal attribution plus source-targeted
decoding; and [ConRad](https://arxiv.org/abs/2603.29492) occupies calibrated
confidence expression in radiology reports. A new threshold, projection,
external verifier, uncertainty word, or adaptive steering rule is therefore not
a new mechanism.

This conclusion also respects the local negative evidence. Spatial reader
consensus showed no reliable decodability or erasure pattern
([audit](SPATIAL_READER_CONSENSUS_FAST_NO_GO_20260803.md)); source identity,
anatomy–finding binding, claim-boundary reset, specificity, directional
uncertainty, and source-domain-center all lack either the required substrate or
a surviving mechanism delta. Reopening one under a new name would be outcome
shopping.

## 2. Candidate tree and adversarial verdicts

### Candidate A — Absolute-to-relative evidence compression in OE

**Tempting question.** Does a medical VLM estimate each finding independently,
but convert those absolute supports into a within-answer rank during list
generation, so a weak claim is fabricated on an otherwise normal image while a
clear claim is omitted on a multi-finding image?

**Why it was attractive.** It would unify fabrication and omission without
assuming one scalar confidence axis. VinDr supplies the same 14 reader-voted
claims per image, natural normal/single/multi-finding strata, and a direct
open-cardinality listing output. A clean probe could keep a target claim fixed
while varying the number of 0/3 decoys versus 3/3 competing findings in the
listed ontology.

**Fatal collision.** This is not an unoccupied mechanism:

- [Large Language Models Do Multi-Label Classification Differently, EMNLP
  2025](https://aclanthology.org/2025.emnlp-main.126/) directly shows that
  autoregressive LLMs behave as sequential single-label classifiers: one label
  receives a sharp probability peak at each generation step, other labels are
  suppressed, and high-probability labels are not consistently revisited. It
  also evaluates alignment to empirical human annotation distributions and
  proposes max-over-generation correction.
- [Multi-Object Hallucination in Vision-Language Models](https://arxiv.org/abs/2407.06192)
  already shows that testing multiple objects increases hallucination and that
  the queried object-class distribution changes behavior.
- [Two Causes, Not One](https://arxiv.org/abs/2509.00371) explicitly separates
  omission from low-confidence visual-to-language mapping and fabrication from
  spurious cross-modal co-occurrence.
- [Template Collapse](https://arxiv.org/abs/2605.30984) already identifies
  normal-template bias and rare-finding loss in medical generation, then
  separates “what to say” from language realization.

Changing the objects to radiographic findings, using reader votes, or calling
the sharp sequential distribution “evidence rank compression” changes the
dataset and truth quality, not the mechanism. The medical setting does not
invalidate an assumption of the closest works. **G2 fails: direct/cosmetic
collision. Prune.**

#### The experiment that would have been decisive—but cannot earn novelty

For completeness, the outcome-blind test would have been a nested-ontology
factorial on the frozen VinDr dev split: same image and target finding; equal
candidate-set size; add either 0/3 decoys or 3/3 competitors; compare target
membership and target logit after controlling finding, target reader support,
canonical score, label-token length, label order, realized claim count, and
response length. A target-specific competitor effect at least 0.05 AUROC beyond
those controls in both Huatuo and Hulu, with image-cluster bootstrap 95% CI
excluding zero, would establish the behavior. It would still be an application
of known multi-label serialization dynamics, not an ICLR-level pivot.

Any derived independent-threshold decoder must be evaluated at matched positive
claim count and matched response length; 0/3 inclusion must fall by at least
20% relative with no decrease in 3/3 recall. Deleting claims, selecting `K=0`,
adding hedges, or increasing format failures/refusals is failure. These controls
remain useful for CECD's OE transfer, but they do not define a new paper.

### Candidate B — Reader-threshold aliasing (“the model as a fourth reader”)

**Sharper question.** Is apparent overcommitment on 1/3 and 2/3 cases actually
clarity erasure, or does the VLM implement a stable reader-like operating point
that aligns with one member of the fixed VinDr panel? The unique prediction is
that, conditional on finding and vote count, model commitment depends on *which*
reader dissented, and the same reader alignment recurs across findings, prompts,
and CE/OE.

This question is meaningfully different from CEBC/VLI/HalluTrace: its causal
variable is the clinical decision boundary represented by reader identity, not
visual detector confidence or visual–language conflict. It is also stricter
than merely fitting the marginal human-answer distribution. [Mind the
Uncertainty in Human Disagreement, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32468)
evaluates VLM agreement with human response distributions, while
[CheXthought](https://arxiv.org/abs/2604.26288) predicts human–human and human–AI
disagreement. Neither retrieved source asks whether a generative medical VLM
selects one coherent radiologist operating point rather than erasing
disagreement. [NUTMEG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.144/)
is the important adjacent warning: annotator-specific systematic disagreement
must be separated from noise before aggregation.

#### Outcome-blind substrate check performed

The official VinDr wide labels contain an exact `R8/R9/R10` panel on 5,501
images, so reader composition is fixed rather than confounded by changing
panels. Reader prevalences nevertheless differ substantially. Examples from the
official wide table are:

| Finding | R8 positive | R9 positive | R10 positive | Total per reader |
|---|---:|---:|---:|---:|
| Aortic enlargement | 2,064 | 2,546 | 2,353 | 5,501 |
| Lung opacity | 353 | 859 | 745 | 5,501 |
| Other lesion | 251 | 742 | 510 | 5,501 |
| Pleural effusion | 502 | 866 | 671 | 5,501 |
| Pleural thickening | 766 | 1,580 | 826 | 5,501 |

Identity-specific disagreement cells are not empty. For cardiomegaly, the six
one-dissenter/one-supporter patterns contain 110–244 images each; for pulmonary
fibrosis they contain 91–326. Across all eight primary findings the smallest
such cell ranges from 19 to 110, so a pooled identity test is feasible, while
some per-finding tests are not. These counts were computed without model
outputs from
`annotations/image_labels_train.csv`; they confirm a measurable reader effect,
not the VLM mechanism.

#### Why it still fails as a main pivot

1. **It does not explain clear-case hallucination.** Reader identity can alter
   the interpretation of 1/3 and 2/3 claims, but cannot turn a 0/3 fabricated
   claim or 3/3 omission into a reader-operating-point choice.
2. **The observational prediction is not yet causal.** Alignment with R9, for
   example, may reflect that R9 is more liberal, not a reader-like latent state.
   Pseudonymous `rad_ID` supplies no reader trait or controlled intervention.
3. **Its natural method is already an occupied family.** Aggregating or
   calibrating to annotator-specific distributions is multi-annotator modeling;
   converting it into confidence language returns to ConRad/CEBC territory.
4. **The current CECD protocol already preserves reader identity and requires
   reader/finding effects.** Therefore this is an important alternative
   explanation and evaluation audit, not a new replacement problem.

**G1/G2/G3 pass only for a narrow measurement question; G4/endpoint fails for a
hallucination-mitigation paper. Do not pivot.**

#### Low-cost exclusion control that is worth retaining

Before interpreting 1/3–2/3 CECD errors as overcommitment, fit on `dev_fit` only:

```text
error_or_commitment ~ vote_count + positive_reader_pattern
                      + finding + model + clean_margin
```

and apply once to `confirmation_locked`, with image-cluster bootstrap. Reopen
this as a *mechanism candidate* only if all conditions hold:

1. reader-pattern terms improve held-out pooled AUROC by at least 0.05 and NLL
   by at least 5% beyond vote count/finding/clean score, with 95% CIs excluding
   zero;
2. one reader-alignment ordering is directionally stable in at least 6/8
   findings and in both Huatuo and Hulu;
3. the same ordering transfers from atomic CE to direct 14-class listing under
   matched claim count and answer length;
4. it predicts error on 0/3 or 3/3 clear cases beyond generic sensitivity,
   otherwise the claim remains “reader-disagreement semantics,” not
   hallucination mechanism;
5. a future intervention changes reader-operating-point alignment while
   preserving claim identity, output length, 3/3 recall, and 0/3 precision.

Kill it if identity gain is below 0.05 AUROC, unstable across findings/models,
or disappears after a single finding-specific sensitivity term. The control
must not inspect `confirmation_locked` while choosing readers, thresholds,
findings, or transformations.

## 3. Collision matrix

| Work | What it occupies | Candidate A | Candidate B | Remaining delta |
|---|---|---|---|---|
| [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) | detector-bounded minimal editing with conformal risk | Any support-threshold edit collides | No reader identity | Reader threshold is different, but no causal endpoint |
| [VLI, ACL 2026](https://aclanthology.org/2026.acl-long.1784/) | instance conflict diagnosis and adaptive bi-causal steering | Generic adaptive visual correction collides | No reader panel | Reader alignment remains observational |
| [HalluTrace, ALVR 2026](https://aclanthology.org/2026.alvr-main.29/) | causal failure-source decomposition and targeted decoding | visual/prior explanation collides | No annotator boundary | Reader identity is not a failure source yet |
| [ConRad, 2026](https://arxiv.org/abs/2603.29492) | calibrated confidence language for radiology reports | confidence output collides | confidence endpoint collides | Identity-specific behavior only |
| [LLMs Do Multi-Label Classification Differently, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.126/) | sequential single-labelization and human-distribution alignment | **Direct mechanism collision** | Marginal distributions, not coherent radiologist selection | Narrow reader-identity test |
| [Multi-Object Hallucination](https://arxiv.org/abs/2407.06192) | multi-object burden and class-set effects | **Direct phenomenon collision** | No multi-reader semantics | None for set competition |
| [Template Collapse](https://arxiv.org/abs/2605.30984) | normal-template bias, rare-finding loss, what/how separation | Medical endpoint collision | No reader identity | No main-method delta |
| [CheXthought](https://arxiv.org/abs/2604.26288) | multi-reader reasoning/attention and disagreement prediction | Independent grounding support | Closest medical mechanism competitor | Coherent virtual-reader selection only |
| [NUTMEG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.144/) | separating systematic annotator disagreement from noise | — | Adjacent statistical collision | VLM-specific causal transition missing |

No mechanism-equivalent work was retrieved for the narrow “coherent fourth
reader” observation under the documented searches, but absence of a direct
paper is not enough: the candidate lacks a causal intervention and a clear-case
hallucination endpoint.

## 4. Scoring and construction-path audit

Scores follow the mechanism-discovery hard gates and `I/M/N/E` rubric; a failed
hard gate prunes regardless of weighted score.

| Candidate | I | M | N | E | Base | Hard-gate verdict |
|---|---:|---:|---:|---:|---:|---|
| A. Absolute-to-relative evidence compression | 3 | 2 | 0 | 3 | 2.0 | **Prune: G2 direct/cosmetic collision** |
| B. Reader-threshold aliasing | 2 | 2 | 2 | 3 | 2.2 | **Do not pivot: mitigation endpoint/causal-variable intervention missing** |

Four exemplar construction paths were used as analytical reconstructions, not
claims about the authors' private discovery histories:

- **SigLIP:** remove accidental coupling. This motivated asking whether OE
  incorrectly couples otherwise independent findings. Exact search showed that
  sequential multi-label coupling is already characterized, so the branch was
  pruned rather than renamed.
- **ViT:** redefine the basic unit. Reader-specific decision vectors, rather
  than anonymous vote counts, are a more faithful unit for disagreement. This
  improves the control design but does not by itself create a hallucination
  mechanism.
- **Model Collapse:** reduce a broad failure to one state variable. “Normal
  template as absorbing state” was considered, but Template Collapse already
  owns the problem and a what-to-say/how-to-say remedy.
- **Chinchilla:** identify the correct limiting relation. Matching claim count,
  response length, and coverage is necessary to separate hallucination gains
  from a reporting-budget trade, but this is an experimental law/control, not a
  new mechanism.

The user-provided contract already froze the problem (medical VLM
image-grounded overcommitment), mechanism-search standard (mechanism-first,
high novelty), and substrate (VinDr fixed reader panel, Huatuo/Hulu, 14-class
listing). No scientific assumption was silently changed.

## 5. Final research decision

1. **Do not replace CECD with either candidate.** CECD's independently admitted
   render-by-wording product-orbit clinical-error residual remains narrower but
   more defensible than the alternatives.
2. Add reader-pattern identity only as a pre-registered alternative-explanation
   control for disagreement cases. It cannot authorize, reject, or retune the
   0/3–3/3 primary gate by itself.
3. Use the 14-class listing track to test CECD transfer with fixed claim count,
   matched length, required-claim recall, fabricated inclusion, format failure,
   and refusal jointly. Do not market generic multi-label competition as a new
   mechanism.
4. If CECD locked confirmation fails, report a credible negative result and
   return to new-substrate acquisition. The existing VinDr substrate has been
   mined to the point where another same-data decoder story is more likely to be
   a collision than an ICLR-oral idea.

**Bottom line:** as of 2026-08-03, the scientifically correct post-CEBC decision
is **NO-PIVOT**, not a third calibration or steering variant.
