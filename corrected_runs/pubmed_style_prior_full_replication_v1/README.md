# Full PubMedVision-CXR Source-Distribution Replication

This source-only replication extends the frozen 2,048-image style-prior
protocol to all 5,549 unique strict-CXR images (4,894 PMC groups) available in
the prepared PubMedVision instruction split. It changes only sample size:
features, lexicon, group splitting, estimators, controls, seed, and gates are
identical. No target image or label is accessed.

## Additive question-control audit

The test split contains 1,541 images; 4,008 form the training split.

| Concept | question AUROC | style-only AUROC | question+style minus question | group-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Pneumothorax | .764 | .536 | -.025 | [-.053, .002] |
| Effusion | .696 | .573 | +.016 | [-.011, .042] |
| Opacity | .803 | .567 | -.004 | [-.012, .003] |
| Cardiomegaly | .791 | .586 | -.010 | [-.040, .020] |
| Edema | .770 | .649 | -.002 | [-.026, .024] |
| Fracture | .754 | .560 | +.000 | [-.073, .089] |
| Device | .870 | .541 | -.011 | [-.027, .005] |
| Normal | .884 | .585 | -.025 | [-.071, .021] |

Zero of eight concepts pass the frozen criterion (increment at least .03,
confidence lower bound above zero). More source data narrows most intervals
around zero rather than revealing a hidden conditional style signal.

## Bilinear question-style audit

Five-fold GroupKFold compares question+style against the same model with a
full \(32\times12\) question-style Kronecker interaction. A style shuffle
within question family is the matched negative control.

| Concept | interaction minus additive AUROC | 95% CI |
|---|---:|---:|
| Pneumothorax | -.049 | [-.073, -.027] |
| Effusion | -.040 | [-.059, -.021] |
| Opacity | -.044 | [-.053, -.035] |
| Cardiomegaly | -.068 | [-.091, -.045] |
| Edema | -.085 | [-.102, -.067] |
| Fracture | -.060 | [-.121, .001] |
| Device | -.052 | [-.066, -.039] |
| Normal | -.074 | [-.103, -.046] |

Zero concepts pass the interaction gate. Effusion's real interaction exceeds
the shuffled interaction by `.034` (CI `[.005,.061]`), but remains `.040`
below the additive model, so it is not usable conditional prior information.

## Claim boundary

Supported:

> In the audited PubMedVision-CXR subset, low-level presentation proxies have
> weak marginal association with several reference concepts, while complete
> question text contains substantially stronger predictive information.

Not supported:

> Style independently switches the clinical prior after conditioning on the
> question, HuatuoGPT-Vision causally uses such a switch, or this source
> property explains target-domain hallucination.

The independent natural-image content-preserved/content-removed experiment
remains necessary. Its style-only drift must reproduce a frozen disease
direction; an average answer flip or target-fitted direction is insufficient.

## Reproduction

```bash
PYTHONPATH=. python -m anchor.corrected_sgta.analyze_pubmed_style_prior \
  --manifest \
  /autodl-fs/data/data/dbw/anchor_center_native_v1/prepared_pubmed_full_cxr/instruction_train.jsonl \
  --output corrected_runs/pubmed_style_prior_audit_full_v1 \
  --max-images 6000

PYTHONPATH=. python -m \
  anchor.corrected_sgta.analyze_pubmed_style_question_interaction \
  --manifest \
  /autodl-fs/data/data/dbw/anchor_center_native_v1/prepared_pubmed_full_cxr/instruction_train.jsonl \
  --output corrected_runs/pubmed_style_question_interaction_full_v1 \
  --max-images 6000
```
