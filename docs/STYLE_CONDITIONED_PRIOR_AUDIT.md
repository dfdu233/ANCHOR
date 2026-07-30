# PubMedVision-CXR Style-Conditioned Prior Audit

## Question

Does the PubMedVision CXR training distribution contain a statistical
precondition for style-conditioned clinical prior switching?

The frozen mechanism is:

\[
s \rightarrow z_s \rightarrow \pi(y\mid q,z_s).
\]

This audit tests the first statistical link only. It does not test whether a
VLM causally uses the shortcut.

## Protocol

- Source-only: 2,048 unique strict-CXR PubMedVision images.
- Split: 70/30 by PMC `group_id`; figures from the same article cannot cross
  train/test.
- Style proxy: phase-free radial log-power, border/intensity/gradient
  quantiles, aspect ratio, and saturation statistics.
- Outcomes: eight clinical concepts extracted from reference answers using a
  frozen lexicon.
- Control: word and bi-gram TF-IDF of the complete question.
- Models: logistic regression with an identical split and fixed seed 2027.
- Uncertainty: 500-draw PMC-group bootstrap.
- Target data: none.

The confirmatory criterion was fixed before the run: at least three concepts
must gain at least 0.03 AUROC from adding style to the question model, with a
group-bootstrap 95% interval lower bound above zero.

## Result

| Concept | Style-only AUROC | Question + style minus question | 95% CI of increment |
|---|---:|---:|---:|
| Pneumothorax | .592 | -.000 | [-.052, .055] |
| Effusion | .599 | +.021 | [-.035, .070] |
| Opacity | .580 | +.005 | [-.013, .024] |
| Cardiomegaly | .593 | -.021 | [-.066, .018] |
| Edema | .578 | -.047 | [-.098, -.004] |
| Fracture | .599 | -.075 | [-.189, .026] |
| Device | .573 | -.031 | [-.084, .010] |
| Normal | .474 | -.097 | [-.199, -.021] |

Six concepts had a descriptive style-only AUROC of at least .55 with a
group-bootstrap lower bound above .50. This establishes weak marginal
style--concept coupling across unseen PMC articles. No concept met the
conditional increment criterion; the confirmatory gate failed.

![Style-prior audit](../results_reference/pubmed_style_prior_audit_v1/style_prior_audit.png)

## Interpretation

The result supports a limited statement:

> PubMedVision-CXR contains marginal acquisition/presentation confounding for
> several clinical concepts.

It does not support:

> Style independently selects the clinical answer prior after conditioning on
> the question, or HuatuoGPT-Vision uses this association.

This distinction matters because style-only predictability may arise from
case-mix or coarse clinical content retained in Fourier amplitude. The paired
content-removal experiment must therefore demonstrate directional alignment
between style-only prior drift and content-preserving answer drift after
controlling the question and original margin.

### Bilinear interaction follow-up

Because \(\pi(y\mid q,z_s)\) could act through a question--style interaction
rather than an additive style term, a second source-only audit used five-fold
PMC-group cross-validation. Questions were mapped to a fixed 32-dimensional
TF-IDF/SVD representation, styles to 12 principal components, and the model
received the full \(32\times12\) Kronecker product. A within-question-family
style shuffle provided a matched negative control.

The interaction gate also failed. The real interaction reduced AUROC relative
to the additive model for every concept:

| Concept | Interaction minus additive AUROC |
|---|---:|
| Pneumothorax | -.074 |
| Effusion | -.082 |
| Opacity | -.093 |
| Cardiomegaly | -.104 |
| Edema | -.093 |
| Fracture | -.088 |
| Device | -.056 |
| Normal | -.129 |

For seven concepts the point estimate was significantly negative; the real
interaction did not reliably outperform the shuffled interaction for any
concept. Thus, the negative conditional result is not explained by a simple
low-rank bilinear prior switch.

![Question-style interaction audit](../results_reference/pubmed_style_question_interaction_v1/style_question_interaction.png)

### Training-lineage shared-content probe

A third, model-level probe tested a prerequisite that is orthogonal to the
target-domain \(2\times2\) experiment. We formed six PubMedVision style
clusters from 2,048 source CXR images. For each cluster, three independently
estimated low-frequency log-amplitude centers were applied to the same blurred
normal-CXR median. The 18 prototypes therefore share phase and high-frequency
content (mean correlation with the common base: .951) while varying only the
source-derived presentation statistic.

HuatuoGPT-Vision-7B, whose medical training lineage includes PubMedVision,
answered six full-sentence disease questions for every prototype. All
108/108 answers were affirmative. Parse rate was 100%, but the mean
between-style decision range was exactly zero and no disease flipped across
clusters. The necessary within-medical-model switch criterion therefore
failed before an unadapted Qwen control could affect the conclusion.

The same prototypes were evaluated using an open radiology-report prompt to
avoid binary-answer saturation. The model produced 14 unique reports from 18
inputs (mean 68.8 words). Effusion and cardiomegaly were each positively
asserted in 50.0% of reports. Opacity had a nominal cluster association
(\(p=.024\)), but it did not survive Holm correction across the six concepts
(\(p_{\mathrm{Holm}}=.145\)). No concept passed the corrected gate. This
analysis uses sentence-local negation and excludes uncertain mentions such as
“may represent” or “cannot exclude”; the earlier permissive parser
overcounted these as positive findings.

![Training-lineage style probe](../results_reference/pubmed_style_lineage_probe_v1/style_lineage_probe.png)

This is a useful negative distinction: the controlled images expose a strong
question-conditioned affirmative prior and unstable report hallucinations,
but not a stable **style-conditioned** clinical prior. The exact-size base
Qwen comparison is now being used as an orthogonal lineage control on natural
MIMIC images rather than as an attempt to rescue this failed synthetic gate.
These synthetic prototypes are diagnostic rather than natural target images.

### Competing mechanism: question-conditioned prior

The affirmative saturation suggested an alternative explanation. On the same
18 prototypes and six diseases, we crossed style with three complete-sentence
question frames: positive inquiry (“is there evidence of ...”), neutral
presence/absence adjudication, and negative confirmation. Clinical presence
was parsed from the meaning of the complete sentence; a leading “yes” in
“yes, the image does not show ...” was correctly treated as absence.

The model asserted the queried disease on 100% of positive inquiries and 0%
of neutral or negative frames. The mean paired prompt range was 1.00, whereas
the mean source-style-cluster range was 0.00 (cluster-bootstrap difference
95% CI \([1.00,1.00]\)); semantic parse rate was 100%. Thus, under
content-weak probes, question wording completely determined the clinical
claim while the tested source style did not.

![Prompt-style factorial](../results_reference/pubmed_style_lineage_probe_v1/prompt_style_factorial.png)

This model behavior has a matching source-distribution property. Across 5,549
unique strict-CXR PubMedVision instruction examples (4,894 PMC groups), a
clinical concept named in the question was much more likely to be positively
asserted in the reference answer:

| Concept | Cue in question | No cue | Difference |
|---|---:|---:|---:|
| Pneumothorax | .659 | .038 | +.621 |
| Effusion | .629 | .078 | +.551 |
| Opacity | .730 | .298 | +.432 |
| Cardiomegaly | .711 | .046 | +.665 |
| Device | .858 | .061 | +.797 |

All five met the frozen PMC-group-bootstrap criterion (difference CI lower
bound \(>.25\)). This is expected semantic alignment in a generated VQA
corpus, not proof of annotation leakage. Together with the controlled model
probe, however, it identifies a stronger competing mechanism: when image
content is weak, HuatuoGPT-Vision follows a question-conditioned clinical
prior consistent with the source instructions rather than selecting a prior
from the tested CXR presentation style.

![PubMedVision question prior](../results_reference/pubmed_question_prior_audit_v1/question_prior.png)

The finding is consistent with independent evidence that hidden data
acquisition biases can become medical shortcuts, but it is not a replication
of that result in a generative VLM:

- [Shortcut learning in medical AI hinders generalization](https://www.nature.com/articles/s41746-024-01118-4)
- [There Are No Shortcuts to Anywhere Worth Going](https://proceedings.mlr.press/v250/boland24a.html)
- [LENS](https://openreview.net/pdf?id=oh3c2ieVab) studies cross-view stability,
  not style-conditioned priors.
- [VCD](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)
  assumes its distorted branch provides a meaningful visual counterfactual;
  this source-data audit alone cannot establish whether that assumption fails.

### Natural-image question-frame control

To test whether the prompt effect survives beyond synthetic prototypes, we
evaluated the first 64 fixed RULE/MIMIC questions (13 unique images) using the
same HuatuoGPT-Vision checkpoint. The original question and a neutral frame
were paired on every image and both produced a complete sentence. A blinded
DeepSeek-V4-Flash judge mapped sentence meaning to Yes/No without seeing the
method, frame, or reference answer; the deterministic RULE parser is retained
only as a secondary audit because explanatory sentences can contain both
“yes” and “no”.

| Frame | Accuracy | Balanced accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Original | .750 | .722 | 34 | 14 | 14 | 2 |
| Neutral | .703 | .728 | 19 | 26 | 2 | 17 |

Neutral framing changed 95.3% of generated texts. It sharply reduced false
positives, from 14 to 2, but increased false negatives from 2 to 17. The
accuracy difference was \(-4.69\) percentage points (image-cluster bootstrap
95% CI \([-.172,.108]\)); there were 15 rescues and 18 harms. The intervention
therefore moved the sensitivity--specificity operating point without
improving accuracy.

![Natural MIMIC question-frame probe](../results_reference/mimic_question_frame_probe_v1/question_frame_mimic64.png)

This result is a stronger competing-mechanism control, not a new method:
question wording can switch a clinical response prior on natural images, but
the switch does not supply additional visual evidence. The subset was already
exposed during development, contains only 13 image clusters, and uses one
model checkpoint and one semantic judge, so it cannot support a population
claim about prompt dominance. It does show why prompt or threshold changes
must not be mistaken for visual DG gains.

## Evidence Grade and Next Decision

Evidence grade: **C-level mechanism clue** under the unified evaluation policy.
It is a source-distribution audit, not a model-performance result and must not
enter the main benchmark table.

A fresh same-family result-to-claim review returned
`claim_supported = no` with high confidence
(`acceptance_status = provisional`). The reviewer agreed that the experiment
supports weak marginal confounding, but not clinical-prior information beyond
the question, causal prior switching, or VLM shortcut use.

Across the source audit, bilinear audit, binary lineage probe, open-report
probe, and prompt factorial, the evidence now favors **question-conditioned
language-prior dominance under content-weak inputs**, not a discrete prior
selected by the tested style clusters. This is still C-level diagnostic
evidence: it does not identify the causal contribution of PubMedVision
training without an exact base-model comparison. The frozen \(2\times2\)
real-image experiment is the remaining decisive test. If content-preserving
and content-removed style drifts do not align after controls, the
style-conditioned prior-switching mechanism is falsified despite the marginal
source-data association.

The remaining decisive condition is consequently narrower. On fixed natural
questions, a style-conditioned-prior claim requires the independent
content/style \(2\times2\) experiment to show that (i) style-only,
content-removed views induce a reproducible disease-direction drift and (ii)
the same drift aligns with the content-preserving style view. If either link
fails after controlling the original margin, the tested style-switch
mechanism should be frozen as unsupported.

The subsequent [Visual Evidence Chord Audit](VISUAL_EVIDENCE_CHORD_AUDIT.md)
tests a stricter vector-valued prediction on 64 natural MIMIC images. It finds
that source style identity explains only 3.28% of evidence-drift variance on
the frontal subset, while patient and patient-by-style terms explain 73.43%
and 23.29%. This refutes a uniform style-prior switch for the tested source
transform and supplies a quantitative ceiling for any global additive source
correction.
