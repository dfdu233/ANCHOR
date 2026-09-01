# Model-Specific Source Envelope Calibration: Research Log

Last updated: 2026-07-29

## Frozen Scope

The intervention is restricted to the input image. The VLM weights, prompt,
tokenizer, generation parameters, and output text are unchanged. The method
must emit one ordinary free-text answer and must work for binary VQA,
open-ended VQA, and report generation.

Excluded from the main search:

- answer voting, candidate selection, or NLL-based reranking;
- logit, attention, hidden-state, or decoder modification;
- retrieval, similar reports, external knowledge, or prompt modification;
- task-specific rules that only work for yes/no questions.

The initial method family is deliberately small:

1. Native image.
2. Radial source-mean calibration.
3. One robust radial source envelope with identity fallback.

Multicenter clustering, SIGReg splitting, learned restoration, and extra
verifiers are deferred until a simple input-only method shows a real benefit.

## Literature And Code Verification

### FedDG

FedDG is a training-time federated domain-generalization method. Its official
implementation stores per-image amplitude spectra and samples concrete spectra
from other clients for low-frequency exchange while retaining the current
image phase. It does not construct one fixed global source center.

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html
- Code: https://github.com/liuquande/FedDG-ELCFS

Consequence: the current method is `FedDG-inspired source-prototype image
calibration`, not an official FedDG reproduction.

### FACT

FACT mixes amplitude during training and uses prediction consistency to make
the trained network rely more on phase. The paper itself reports that replacing
the full amplitude can be too aggressive. The training and consistency
components violate the frozen-model constraint.

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html
- Code: https://github.com/MediaBrain-SJTU/FACT

### TF-Cal / TAF-Cal

TF-Cal is the closest published baseline in intent: it estimates a source
amplitude prototype and convexly calibrates target amplitude at test time.
However, it operates on early CNN feature maps and is complemented by
training-time amplitude augmentation. Direct pixel-space use with a frozen VLM
is an adaptation, not the published method.

- Paper: https://www.ijcai.org/proceedings/2022/240
- PDF: https://www.ijcai.org/proceedings/2022/0240.pdf

### FDA And DAC

FDA supports low-frequency amplitude transfer with phase retention, but it is
an adaptation/training framework. Its paper also documents visible artifacts
for aggressive frequency replacement. DAC explicitly treats content change as
a risk of normalization and decomposes frequency processing inside a trained
network. Both support conservative gain limits; neither directly validates a
frozen-VLM pixel calibrator.

- FDA: https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- DAC: https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Decompose_Adjust_Compose_Effective_Normalization_by_Playing_With_Frequency_for_CVPR_2023_paper.html

### Additional Boundary Evidence

- NeurIPS 2024 shows that frequency shortcuts can create misleading apparent
  DG gains. This strengthens the need for paired harm/rescue analysis rather
  than aggregate accuracy alone:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/377235d5cee7b104501407c0e5066c92-Abstract-Conference.html
- WACV 2026 reports VLM corruption gains from trained denoisers, but learned
  denoisers are outside this study:
  https://openaccess.thecvf.com/content/WACV2026/html/Latif_Enhancing_Vision_Language_Corruption_Robustness_using_Cross-Distribution__Prompted_Denoisers_WACV_2026_paper.html
- MedHEval separates visual misinterpretation from knowledge deficiency and
  context misalignment. Input calibration can directly target only the first:
  https://arxiv.org/abs/2503.02157

## Implemented Method

The Huatuo-specific source bank uses accessible PubMedVision training images
that are conservatively filtered as single-view CXR. Every source and target
image is first transformed to the model-visible 336x336 square-padded image.

For each image, the descriptor is a 64-bin radial median of log Fourier
amplitude on luminance. The bank stores:

- mean and median radial profiles;
- MAD-based robust scale;
- 5th/95th percentile bounds;
- a scalar validation inflation chosen for 95% simultaneous source identity.

At inference, the envelope method clips only descriptor components outside the
source envelope. It converts the clipped radial difference into an isotropic
gain, applies that gain to the original RGB amplitudes, preserves phase, and
reconstructs one image. Images already inside the envelope are bitwise
unchanged.

Bank:

- Path: `/home/dbw/data/mosec_banks/huatuo_pubmedvision_cxr/cxr_radial_envelope.npz`
- SHA256: `94aa056d8fbff2b3ccb05c18c31282c1ddf2582b313d9fd9d3b1ae1a5413e921`
- Retained source images: 5,846
- Train/validation: 4,649 / 1,197
- Validation identity: 94.99%

The audit sheet is mostly clean single-view CXR but retains a small amount of
source noise, including at least one composite figure. This is a documented
limitation, not silently treated as a perfect source distribution.

The originally proposed
`data/modality_centers/PubMedVision/train/ct__chest.npy` is a 384x384 complete
2-D amplitude mean built from 18,000-plus metadata rows labeled CT/chest. It is
not a CXR-specific bank and is therefore not used as the main Huatuo CXR
source. PubMedVision's raw modality metadata also labels many caption-confirmed
CXR figures as computed tomography, so the replacement bank uses conservative
caption, geometry, and visual audit filters rather than trusting that metadata
field alone.

## Evaluation Integrity Findings

### CE

Huatuo emits full sentences. RULE's negative-word convention can incorrectly
map `Yes, ... was not present previously` to `No`. A leading-only parser then
incorrectly rejects natural answers such as `Based on the image, there is no
evidence ...`.

The Huatuo primary diagnostic is therefore `decision_first`:

1. Use an explicit leading `Yes` or `No` when present.
2. Otherwise use the strict semantic binary parser.
3. Treat unresolved outputs as invalid and incorrect.

Legacy RULE-normalized, POPE-compatible, strict, and leading-only results are
retained as diagnostics.

### OE

MedHEval visual OE/MIMIC contains 490 unique images and one fixed report
generation prompt. References have a median of 47 words. It is report
generation, not a diverse set of free-form question types.

Lexical BLEU, ROUGE, and METEOR measure overlap, not clinical factuality.
MedHEval's public clinical scripts contain hard-coded local paths, while its
knowledge judge depends on a private Bedrock client. Local results must name
the exact replacement implementation.

Clinical metrics are kept separate:

- RadGraph F1 measures clinical entity/relation overlap and has stronger
  radiologist correlation than generic overlap metrics:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844/
- CheXbert provides a 14-observation chest-report labeler:
  https://aclanthology.org/2020.emnlp-main.117/
- RaTEScore is entity-aware, synonym-aware, and negation-sensitive:
  https://aclanthology.org/2024.emnlp-main.836/

The initial 128-token run had a 60-80% maximum-length rate, so it is retained
only as a runtime/risk screen. The valid comparison uses 256 tokens and reports
the maximum-token rate.

## Current Small-Sample Evidence

### RULE/MIMIC CE, n=16

| Method | Accuracy | Parse | Rescue | Harm | Identity |
|---|---:|---:|---:|---:|---:|
| Native | 68.75% | 87.5% | - | - | 100% |
| Mean 0.05 | 68.75% | 87.5% | 0 | 0 | 0% |
| Mean 0.10 | 62.50% | 87.5% | 0 | 1 | 0% |
| Mean 0.20 | 68.75% | 87.5% | 0 | 0 | 0% |
| Envelope 0.25/0.5/1.0 | 68.75% | 87.5% | 0 | 0 | 100% |

No method has a positive CE signal. Mean 0.10 is harmful. Envelope equality is
caused by identity, not improved reasoning.

### MedHEval Visual OE/MIMIC, n=16, 256 Tokens

| Method | Avg BLEU | BLEU-4 | ROUGE-L | METEOR | Errors | Max-token |
|---|---:|---:|---:|---:|---:|---:|
| Native | 9.08 | 3.11 | 20.31 | 25.20 | 0 | 0% |
| Mean 0.05 | 9.07 | 2.98 | 21.00 | 26.65 | 0 | 0% |
| Envelope 1.0 | 9.14 | 3.11 | 20.52 | 25.37 | 0 | 0% |

Mean 0.05 changes 14/16 outputs. It wins ROUGE-L on 9 and loses on 4, but
manual review finds both genuine partial rescues and newly unsupported severe
findings. The lexical increase is not sufficient evidence of lower
hallucination. Envelope changes only 1/16.

### Low-Frequency Mean Ablation, n=16

To match FedDG/FDA more closely, a minimal ablation changes only the first 2
of 64 radial frequency bands. It adds no model, selector, or output rule.

- CE: weights 0.05 and 0.10 both produce 0 rescue, 0 harm, and 0 decision
  flips versus native.
- OE: native average BLEU/ROUGE-L is 9.08/20.31; weight 0.05 gives
  8.62/19.57 and weight 0.10 gives 8.57/19.46.
- A two-band weight-0.05 perturbation still creates severe unsupported
  findings in the first inspected OE case.

Restricting calibration to low frequencies does not recover the method.

### Unlabeled Target Shift Audit

| Target | Audited | Envelope identity |
|---|---:|---:|
| RULE/MIMIC | 500 | 98.8% |
| IU-Xray | 590 | 95.8% |
| CheXpert processed subset | 500 | 99.8% |

The robust source envelope sees little frequency shift on these processed CXR
targets. This explains why a conservative envelope rarely activates and why
forcing every image toward a mean can create harm.

### Source Coverage Ablation

The source validation set was also used to construct fixed simultaneous
coverage banks without target labels:

| Bank | Source validation identity | RULE n=500 | IU-Xray n=590 | CheXpert n=500 |
|---|---:|---:|---:|---:|
| c50 | 53.47% | 39.8% | 65.4% | 45.4% |
| c80 | 79.95% | 81.0% | 83.2% | 86.2% |
| c95 | 94.99% | 98.8% | 95.8% | 99.8% |

The c80 bank is the most distributionally coherent natural-data setting.
On CE n=16 it changes 3 images but produces no decision flip, rescue, or harm.
The aggressive c50 bank changes 7/16 CE images without a decision rescue and
degrades OE average BLEU from 9.08 to at most 8.42 and ROUGE-L from 20.31 to
at most 18.88. It also introduces unsupported pneumothorax and trauma claims.

A gamma=2 controlled style shift increases c80 activation, but calibration
reduces CE accuracy from 68.75% to 62.50% (0 rescue, 1 harm). The envelope
therefore fails even when a controlled shift is detected.

### Local Source Frequency Projection

ANCHOR's source-only half of `qls_tr.py` was isolated from its forbidden
question/logit trust-region component. A deterministic index uses 1,024
PubMedVision CXR train images, a 16x16 normalized log-amplitude descriptor,
rank-16 PCA, 8-neighbor KDE, and a source-derived step radius. It produces one
phase-preserving image and never reads a question, label, logit, or output.

- Structure n=16: `l=0.03` PSNR 27.27 dB / edge correlation 0.9976;
  `l=0.1` gives 26.97 dB / 0.9929.
- CE: `l=0.03` is neutral (0 rescue, 0 harm); `l=0.1` has 0 rescue, 1 harm.
- OE `l=0.03`: average BLEU 8.75 vs 9.08 native, ROUGE-L 19.78 vs 20.31.
- The 16-pair judge prefers native/local/tie in 5/4/7 cases and gives local
  slightly worse average factuality and hallucination scores.

The local prototype avoids a single global center but does not fix the
input-reconstruction failure.

### Source RGB Mean/Std Projection

The final breadth branch matches only RGB first/second moments toward the same
1,024 source images. It is much milder than Fourier reconstruction:

- `s=0.1`: PSNR 43.63 dB, edge correlation 0.99978.
- `s=0.25`: PSNR 36.06 dB, edge correlation 0.99967.
- CE `s=0.25`: 68.75% to 75.00%, with 1 rescue and 0 harm.
- OE `s=0.25`: average BLEU 9.08 to 8.63 and ROUGE-L 20.31 to 19.75;
  METEOR rises from 25.20 to 25.90.
- Manual review finds an unsupported right pneumothorax/bilateral effusion
  and loss of bilateral-effusion/device specificity in another case.
- A previously described right-to-left laterality flip (q216) is corrected
  here: the reference says the left effusion is larger, so this is a partial
  rescue, although the method still omits the small right effusion, nodule,
  apical mass, and lower-lobe consolidation.
- The 8-pair auxiliary judge gives native/source/tie preferences of 2/1/5.

The CE rescue is real on this pilot but is not task-general evidence. A c80
envelope gate cannot retain it safely: the rescued CE image is inside c80,
while the severe OE harm image is outside c80. A c95 gate protects the harm
image but also suppresses the CE rescue and leaves 15/16 OE images unchanged.

## Current Decision

- Do not scale any branch to 32, 128, 512, or full.
- Stop mean 0.10 and mean 0.20.
- Stop full-spectrum mean 0.05: the 16-pair reference-grounded judge gives
  8 ties, 4 native wins, and 4 mean wins, with slightly worse mean factuality
  and hallucination averages.
- Stop both low-frequency mean settings.
- Treat c95/c80 envelopes as valid identity-heavy neutral results and c50 as a
  negative result.
- Stop local source-frequency projection: multicenter source matching does not
  remove OE clinical harm.
- Record source RGB mean/std `s=0.25` as a CE-positive but non-general pilot,
  not as an effective final method.
- No searched input-only source-bank method passes both CE and OE gates.
- Do not add SIGReg, learned restoration, output selection, or task-specific
  logic to force a positive result under the frozen scope.

## Independent Holdout Gate (2026-07-30)

The source RGB mean/std branch was re-tested on data disjoint from the earlier
16-example development pilot. Sampling uses the same stable SHA-256 ordering
with development ranks 0--15 and holdout ranks beginning at 16.

### RULE/MIMIC CE Holdout

The 32-example holdout ran as four independently evaluated 8-example
checkpoints. Under the Huatuo decision-first parser:

- Native: 21/32 (65.625%).
- Source RGB statistics, `s=0.25`: 22/32 (68.750%).
- Paired changes: 1 rescue, 0 harm, 1 decision flip, 0 inference errors.

The rescued item asks whether a right PICC tip reaches the mid SVC. Native
answers no, the calibrated image answers yes, and the reference label is yes.
This passes the CE-only gate, but the effect is one example and is not by
itself sufficient for depth scaling.

### MedHEval Visual OE Holdout

The first independent 8-example checkpoint has a small lexical improvement:

- Corpus average BLEU: 6.672% to 6.784%.
- ROUGE-L: 17.207% to 18.004%.

It nevertheless introduces at least two major reference-grounded
contradictions:

- q462: the reference says normal heart size and no pleural effusion; the
  calibrated output asserts cardiomegaly and bilateral pleural effusions.
- q465: the reference says small bilateral effusions and stable cardiomegaly;
  the calibrated output asserts a large right effusion and a normal heart.

This crosses the registered OE catastrophic-harm stop threshold at the first
checkpoint. The remaining OE holdout chunks and the 128/512/full stages were
not run. The lexical increase is retained as evidence that BLEU/ROUGE can move
opposite to clinical factuality.

### Clinical Metric Environment

An isolated CPU environment was created at
`/opt/miniconda3/envs/anchor-clinical-eval` with RadGraph 0.1.18,
F1CheXbert 0.0.2, and RaTEScore 0.6.0. The repository's clinical runner cannot
yet be executed because the pinned checkpoint cache and
`docs/medheval_report_metric_manifest.json` are absent. This is recorded as a
checkpoint-contract block rather than silently downloading unregistered
weights.

### Updated Decision

- Keep the CE n=32 result as a positive but small task-specific observation.
- Stop `sourcestats_s0.25` before n=128 because the joint CE+OE gate fails.
- Do not run 512/full for this branch.
- Do not reinterpret the OE lexical gain as hallucination reduction.

## Public Dataset-Domain Hypothesis Audit (2026-07-30)

The source-style premise was tested directly on three balanced public CXR
proxies: MIMIC-CXR, IU-Xray, and the unverified CheXpert report subset. The
audit used 500 images per source, model-visible resizing, five-fold
cross-validation, and a border-removing center-80% control. Public dataset
identity is only a proxy for institution plus export pipeline; it is not
treated as pure hospital acquisition style.

Artifact:

- `corrected_runs/public_domain_hypotheses/audit_v1.json`

### Source Identifiability

| View | Intensity stats | All radial | Lowest 8 bands | Middle 8 bands | Highest 8 bands |
|---|---:|---:|---:|---:|---:|
| Full image | 82.73% | 79.40% | 59.20% | 69.67% | 63.87% |
| Center 80% | 78.33% | 87.60% | 61.27% | 70.53% | 66.33% |

Chance is 33.33%. Public sources are strongly distinguishable, and the signal
persists after removing the outer border. However, the source signal is much
stronger in the full spectrum than in the lowest frequencies targeted by
FedDG-style exchange. Equal-dimensional eight-band controls also place both
middle and highest frequencies above the lowest band. The evidence therefore
supports dataset-domain shift but rejects the stronger premise that the
relevant shift is primarily a low-frequency amplitude style.

### FedDG-Style Transfer Fidelity

Concrete cross-source amplitude spectra were sampled from the training split,
and the official FedDG-style centered low-frequency patch was replaced while
retaining source phase.

| Low-frequency ratio | PSNR | Edge correlation | Target-probability delta, all radial | Target prediction |
|---|---:|---:|---:|---:|
| 0.01 | 16.30 dB | 0.9593 | +0.031 | 10.13% |
| 0.03 | 15.16 dB | 0.9185 | +0.057 | 14.13% |

The direction is weakly correct on average, but most transformed images are
not classified as the target source. Larger replacement increases structural
distortion without producing reliable source transfer. Pixel structure scores
cannot establish clinical preservation; the existing OE bad cases already
show that active frequency and moment transforms can add unsupported clinical
findings.

### Hypothesis Status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1 public sources are visually distinguishable | Supported | 78-88% source accuracy after shared preprocessing |
| H2 the difference is not only borders | Partially supported | center-80% remains highly separable; lung-mask control is absent |
| H3 FedDG low frequency contains the main source signal | Rejected | equal-width low/middle/high are 61.27/70.53/66.33%; full is 87.60% |
| H4 low-frequency exchange reliably transfers source style | Rejected | target prediction only 10-26%, depending on classifier |
| H5 active transforms preserve clinical content | Rejected for current methods | repeated unsupported OE findings despite high edge similarity |
| H6 frozen Huatuo outputs are image-style sensitive | Supported but nonspecific | CE flips and many OE text changes occur under mild transforms |
| H7 the changes reduce hallucination | Rejected | independent OE checkpoint crosses catastrophic-harm gate |
| H8 current source calibration lowers DG generally | Rejected | CE one-rescue signal does not survive the joint CE/OE gate |

### Data-Only DG Direction

Do not continue stronger source-center projection, additional prototype
clustering, or broader low-frequency search. The next admissible family is a
single deterministic common-support canonicalization applied identically to
every public source and target image. It should remove source-identifying
nuisance rather than synthesize another dataset's appearance.

Breadth ablations are limited to:

1. Geometry/export canonicalization: centered field-of-view crop, square pad,
   one interpolation kernel, grayscale RGB export.
2. Robust monotonic intensity canonicalization: fixed percentile clipping and
   source-pooled percentile mapping, without per-image source selection.
3. Common-resolution canonicalization: one mild source-pooled optical
   transfer cutoff, tested alone before any composition.

Each component first passes a data-only gate: at least a 15-point reduction in
five-fold source balanced accuracy on held-out images, zero read failures, and
better preservation than direct FedDG exchange. A composition is tested only
when each included component independently contributes. It then enters paired
Huatuo CE/OE pilots at n=16 and n=8 respectively. Any new major OE
contradiction stops the branch regardless of lexical gains.

## Strategic Direction Decision (2026-07-30)

The accumulated evidence supports a training-distribution problem but does not
support a single source-domain-center problem. HuatuoGPT-Vision was trained on
the heterogeneous, multimodal PubMedVision collection, while LLaVA-Med was
aligned with broad biomedical figures from PMC. Neither provides
hospital-labeled CXR training data from which one model-specific hospital
center can be identified. A mean of accessible proxy CXR images is therefore
not an identifiable property of either model.

The pure frozen-model input-calibration branch is stopped as the main
mitigation direction. Existing experiments show the characteristic
active-versus-safe failure: conservative transforms are identity-heavy or
neutral, while active transforms create clinically important OE harm. The new
public-domain audit additionally rejects the low-frequency-center mechanism.

The selected main direction is **training-time data-centric domain
generalization**:

1. Keep the VLM architecture, prompt, and decoding unchanged.
2. Establish single-source and multi-source ERM baselines before adding a DG
   algorithm.
3. Construct paired native/appearance-perturbed training examples that share
   the same answer or report; perturbations must be measured from public CXR
   source variation and pass clinical-content preservation gates.
4. Fine-tune only the visual projector or a minimal LoRA adapter so the model
   can learn invariance. This is required because augmentation cannot create
   invariance in an already frozen model.
5. Evaluate with leave-one-public-source-out splits and separate CE,
   open-ended VQA, and report metrics. Multi-source ERM is the primary
   comparator; the DG branch is retained only if it improves unseen-source
   clinical factuality without increasing harm.

If parameter updates remain prohibited, the scientifically defensible output
is instead a public-dataset counterfactual stress-test benchmark. Under that
constraint the project should not claim a general hallucination mitigation
method.

### Correction After Prior DG-LoRA Evidence

The training-time data-centric DG recommendation above is superseded. A prior
ANCHOR experiment already trained a bounded post-projector adapter with:

- original and style-transformed images sharing the same ground-truth answer;
- task-only and generic-augmentation controls;
- source-style counterfactual views;
- raw-logit and gold-evidence consistency objectives;
- a fixed source schedule, trust region, and held-out-domain protocol.

The implementation remains in `anchor/corrected_sgta/train_anchor_dg.py`.
The migrated analysis code identifies the result as a negative DG/task-LoRA
pilot. The exact cached metric referenced by that analysis lived under
`/root/autodl-tmp/Hulu-Med/MedUniEval/corrected_runs/anchor_dg_v2/` and is not
present on this server, so its exact delta must not be reconstructed from
memory. The user confirms that the original-plus-stylized, shared-GT LoRA
experiment failed.

This closes both forms of the visual-style DG hypothesis in the current
project:

1. Frozen-model test-time image calibration is neutral when conservative and
   clinically harmful when active.
2. Training-time style augmentation with a minimal adapter does not convert
   the same intervention into reliable generalization.

Accordingly, **data-only visual-style DG is stopped as a hallucination
mitigation direction**. Public-source and counterfactual image processing are
retained only for causal diagnosis, robustness auditing, and negative
evidence. They should not receive additional center, envelope, augmentation,
or LoRA searches.

For an effectiveness-first mitigation project, the next direction must move
to the VLM evidence/generation path. Existing workspace evidence makes
Layer Evidence Transport the strongest empirical candidate, with VISTA as its
mandatory close comparator. This is a change of intervention family, not a
new data-DG variant. If changing the intervention family is disallowed, the
remaining publishable direction is a diagnostic/negative-results study rather
than a mitigation method.

## 2026-07-30 Evidence-DG Hypothesis Audit

We implemented a generic HuatuoGPT-Vision probe that keeps the prompt,
decoder, and model parameters fixed, and tests whether public-dataset source
shift remains visible along the visual-to-language path and predicts task
quality. The complete protocol, metrics, and bad-case audit are recorded in
`docs/EVIDENCE_DG_PILOT.md`.

The source signal is strong: report-source classification reaches 81.25% from
pixel statistics and 70.83% from post-projector visual features (33.33%
chance), while CE source classification reaches 95.31% and 87.50%
respectively (50% chance). However, the signal does not reliably predict
report quality or CE correctness after controlling for source. The compact
evidence trajectory also fails the predeclared significance gate on both
tasks.

Teacher-forced layer mixing lowers reference NLL, but temperature 1.2 matches
or exceeds that reduction on held-out sources. This indicates that the NLL
gain is primarily a calibration/softening effect rather than evidence that
source-domain transport repairs visual grounding.

Following the breadth-first stopping rule, the experiment stopped at 16
reports per source and 32 CE samples per source. We did not launch 128, 512,
or full intervention runs. Public-dataset style differences are therefore
validated as measurable domain shift, but not as a demonstrated causal target
for hallucination mitigation under the current frozen-model, image-only DG
setting.

## 2026-07-30 Source-Shift Round-Trip Audit

The final missing causal control was implemented as a paired round trip:
native image, a public-CXR-derived domain-residual shift, exact projection to
the PubMedVision-CXR envelope, and projection to a mismatched IU-Xray
envelope. Full details are in `docs/SSRT_PILOT.md`.

On RULE/MIMIC n=8, the controlled shift produces no real CE decision flip and
matched repair produces no rescue. On MedHEval visual OE/MIMIC n=4, matched
repair raises ROUGE-L but creates a catastrophic unsupported pneumothorax,
multiple rib fractures, tracheal deviation, and subcutaneous emphysema in one
case. The mismatched bank can also reduce PubMed source-envelope distance,
showing that geometric return is not model-source-specific evidence.

The registered gates stop the experiment before 16/32. This result rules out
the current source-bank image-calibration mechanism more directly than the
earlier unpaired average comparisons.
