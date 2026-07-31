# Research Findings

## PubMedVision-CXR Style-Conditioned Prior Audit

Status: confirmatory source-data gate failed.

On 2,048 unique strict-CXR PubMedVision images split by PMC article, low-level
Fourier/presentation proxies weakly predicted six answer concepts marginally
(style-only AUROC 0.573--0.599 with grouped-bootstrap lower bounds above
chance). After controlling the complete question with a word/bi-gram model,
no concept achieved the preregistered AUROC gain of 0.03 with a confidence
interval lower bound above zero. The strongest positive point estimate was
effusion at +0.0209, with CI [-0.0349, 0.0704].

A fresh same-family result-to-claim review returned `claim_supported = no`
with high confidence and provisional acceptance. The supported claim is
limited to possible marginal presentation/image--concept confounding. The
results do not establish style-conditioned priors beyond the question, causal
prior switching, or VLM use of the association. The paired \(2\times2\)
content/style model-output experiment remains necessary and decisive.

Evidence:

- `results_reference/pubmed_style_prior_audit_v1/summary.json`
- `docs/STYLE_CONDITIONED_PRIOR_AUDIT.md`
- `.aris/traces/result-to-claim/2026-07-30_run02/`

## Training-Native Spectral Support Projection

Status: stopped at the frozen MIMIC development gate.

The official `HuatuoGPT-Vision-7B-Qwen2.5VL` checkpoint (revision
`451ac32400e36cfd07b41b62cbe63e6894895b38`) loaded successfully. A robust
low-frequency support was built from 512 strict PubMedVision CXR images using
the post-`patch_embed`, pre-window-reorder representation. The source
Mahalanobis-style distance had median 1.064 and 95th-percentile radius 1.394.

On the first 64 questions of the permanently exposed MIMIC development set,
all target distances lay inside that source radius (range 0.860--1.271;
median 0.998). Consequently, the radial support projection correctly reduced
to the identity for all 64 samples. Raw and native-support inference were
identical:

- accuracy: 84.375% vs 84.375%;
- balanced accuracy: 81.667% vs 81.667%;
- explicit parse rate: 90.625% vs 90.625%;
- support activation: 0/64;
- complete-text changes: 0/64;
- rescue / harm: 0 / 0.

This falsifies the operational premise needed by this particular method:
MIMIC is not outside the robust PubMedVision-CXR support in the chosen early
spectral representation. It does **not** prove that hospital domain shift is
absent, only that this source-support statistic neither detects nor modifies
it. Shrinking the radius after seeing MIMIC would be target-domain tuning and
is forbidden. Therefore bank B, sham controls, OE, CE-256, and locked full
evaluation were not run.

Evidence:

- `corrected_runs/final_training_native_support_v1/evidence_ledger.json`
- `corrected_runs/final_training_native_support_v1/ce64/gate_a.json`
- `corrected_runs/final_training_native_support_v1/ce64/raw.summary.json`
- `corrected_runs/final_training_native_support_v1/ce64/native_a.summary.json`

## Center-Native Qwen2.5-VL Feasibility

Status: stopped after a failed development gate.

The experiment tested a complete \(2\times2\) causal design separating
training exposure from inference-time CXR feature centering. With the vision
tower frozen, inference centering harmed the clean-trained model and had
negligible effect after center training. The only permitted fallback trained
`patch_embed` and visual blocks 0–1 equally in Clean and Center branches. It
made the clean-model inference effect small and inconclusive, but the matched
Center-Native effect was harmful.

The intervention clearly changed full generated answers, so failure is not a
no-op or missing decoder leverage. The failure is directional: the fixed
log-amplitude center changes evidence without reliably preserving the clinical
decision.

Do not repeat this route by increasing \(\tau\), unfreezing more layers, adding
another adapter, or opening the final test. Any future center method must first
avoid clean-source harm and pass a fresh development gate with a positive
\(D-C\), positive interaction, and appropriate sham/wrong-center controls.

Evidence:

- `corrected_runs/final_center_native_qwen_v1/evidence_ledger.json`
- `/root/autodl-fs/data/dbw/anchor_center_native_v1/dev_ce/factorial_analysis.json`
- `/root/autodl-fs/data/dbw/anchor_center_native_v1/earlyvision_dev_ce/factorial_analysis.json`

## Clinical–Nuisance Separability Audit

A source-only follow-up found a nontrivial distinction between geometric
identifiability and causal utility. On 32 IU-Xray studies, nuisance directions
from paired acquisition perturbations were highly reproducible at vision
blocks 0–1 and had low first-order overlap with full-sequence clinical
gradients. Nevertheless, removing the clinical-orthogonal nuisance subspace at
vision block 1 reduced shifted balanced accuracy by 2.50pp on a disjoint
64-study source pilot (one rescue, two harms).

Thus, `nuisance subspace is identifiable` is supported, while `nuisance
subspace removal improves generation` is refuted for this estimator and
finite-step operator. Do not tune the projection on MIMIC or reinterpret the
geometry gate as task efficacy.

## Conditional Semantic Domain Projection Pilot

Status: stopped at the MIMIC-64 development gate.

Four branches were trained from the same Qwen2.5-VL medical-alignment
checkpoint with equal data, LoRA capacity, optimizer steps, and two-forward
budget: task-only, FedDG-NLL, semantic-only, and CSDP. Evaluation used complete
greedy sentences followed by the normalized RULE parser; no label-token logits
were used as predictions.

Task-only and semantic-only each obtained 55/64 (85.94%). FedDG-NLL and CSDP
each obtained 54/64 (84.38%). Relative to semantic-only, CSDP changed 40/64
generated texts but produced zero rescues and one harm, for a net -1.56pp.
Its parse rate was also lower (89.06% versus 92.19%). Relative to FedDG-NLL,
CSDP had one rescue and one harm and no aggregate gain.

This separates causal leverage from task utility: the learned objectives
substantially alter wording, but the source-style semantic term does not turn
those changes into more correct unknown-domain answers. CSDP therefore failed
the preregistered gate and must not be expanded to CE-256, OE, or full MIMIC.

Evidence:

- `corrected_runs/final_anchor_csdp_pilot_v1/summary.json`
- `/root/autodl-fs/data/dbw/anchor_csdp_v1/pilot60/`

## Visual Evidence Chord and Exact-Base Lineage Control

Status: the proposed reusable style-prior switch was rejected; a narrower
checkpoint-pair style-contraction diagnostic passed.

On 40 BiomedCLIP-selected frontal MIMIC development images from 38 patients,
six fixed PubMedVision-derived Fourier styles were applied while preserving
coarse pixel and edge structure. Complete positive and negative clinical
sentences for six concepts were scored by teacher forcing. A two-way
decomposition of the Huatuo evidence drift attributed 73.43% to image/study,
3.28% to reusable style, and 23.29% to their interaction. No scalar chord,
global style-offset, or concept-diagonal model passed its preregistered gate.
For a fixed style, the largest fraction removable by any global additive
correction was only 2.2%--14.8%.

The exact Qwen2.5-VL-7B base model was then evaluated on the same inputs. Its
style fraction was 2.06%; the Huatuo-minus-base difference was +1.22pp with
patient-cluster-bootstrap 95% CI [-0.94, 3.66]pp. Thus the medical checkpoint
does not show a reliably stronger reusable style-conditioned prior.

The dimensionless operator susceptibility
\[
\kappa_\theta(x)=
\frac{\sqrt{\mathbb E_s\|e_\theta(T_sx)-e_\theta(x)\|^2}}
{\|e_\theta(x)-e_\theta(\varnothing)\|}
\]
was lower for Huatuo than its exact base: median .229 versus .329, paired
difference -.101, 95% CI [-.144, -.026]. A fresh same-family claim review
returned `claim_A_supported=no` for prior switching and
`claim_B_supported=yes` for this narrowly scoped contraction diagnostic.

This does not establish that medical instruction tuning caused the
contraction, that natural scanner shifts behave similarly, or that
hallucination/accuracy improves. It implies that a single global source
center is the wrong factorization for this probe: most residual drift is
image-conditional, while the small reusable directions are partly shared
with the base architecture.

Evidence:

- `results_reference/visual_evidence_chord_probe_v1_n64/`
- `results_reference/visual_evidence_chord_probe_v1_base_n64/`
- `results_reference/visual_evidence_chord_lineage_v1/`
- `docs/VISUAL_EVIDENCE_CHORD_AUDIT.md`
- `.aris/traces/result-to-claim/2026-07-31_run03/`
## 2026-07-31: Layerwise style-orbit audit

- **Verdict:** partial support, high-confidence same-family review; provisional.
- **Test:** one seed-42 matched-versus-group-deranged visual-merger training
  comparison on Qwen2.5-VL-7B, followed by a completely paired layerwise probe
  over 40 frontal exposed MIMIC development images and six fixed source styles.
- **Supported observation:** the matched lineage has a lower final prompt-state
  synthetic-style/real-null displacement ratio than the fixed deranged
  lineage. The paired relative effect is -5.48%, 95% patient-cluster CI
  [-7.43%, -4.00%], accompanied by smaller style drift and greater real-null
  distance.
- **Not supported:** generated clinical-evidence improvement, hallucination
  mitigation, natural acquisition robustness, general DG, or training-seed
  stability. The earlier complete-sentence evidence effect remains null.
- **Constraint:** do not call this an alignment-induced output improvement.
  Treat it as a coordinate-specific, single-seed late-fusion mechanism
  diagnostic until replicated across seeds/derangements and an untouched
  multi-site cohort.
- **Next decisive experiment:** prespecify layer-27 prompt \(\kappa\), replicate
  matched versus multiple eligible derangements across at least five training
  seeds, and add generated clinical factuality as a key secondary endpoint.

## 2026-07-31: Multi-lineage style-orbit confirmation

- **Verdict:** the prespecified seed-42 late-fusion contraction did not
  replicate.
- **Design:** one exploratory plus four confirmatory paired
  matched/image-permuted visual-merger lineages; the new lineages use four
  training seeds and four distinct eligible derangements;
  the source selection, 250-step budget, 40-image/38-patient MIMIC development
  probe, six Fourier styles, prompt, and primary layer were fixed.
- **Primary result:** after excluding discovery seed 42, 2/4 new lineages had
  negative relative \(\kappa_{27}\) effects. Their mean was +0.33%; crossed
  seed-by-patient bootstrap 95% CI [-1.09%, +1.67%], seed-level t CI
  [-1.65%, +2.31%], and one-sided sign-test \(p=0.6875\).
- **Interpretation:** seed 42 (-5.48%) was a lineage-specific outlier relative
  to the four new estimates (-0.60%, +1.92%, -0.72%, +0.72%). Neither style
  drift nor real-null leverage was stable across lineages.
- **Audit note:** the discovery-inclusive estimate is -0.83%, but it must not
  be reported as confirmatory because seed 42 selected the layer and metric.
- **Decision:** do not implement an explicit late-fusion orbit-contraction
  objective from this observation. It would optimize a non-replicated
  diagnostic. This result does not test or refute the separate
  style-conditioned clinical-prior-switching hypothesis.
- **Evidence:** `corrected_runs/multiseed_style_orbit_confirmation_v1/`.

## 2026-07-31: Acquisition-style field factorization

- **Verdict:** stable support for an image-conditioned, non-global style
  displacement field under the controlled Fourier probe.
- **Exact estimand:** for
  \(\Delta_{i,s}=h(T_sx_i)-h(x_i)\), the case mean
  \(\bar\Delta_{\cdot,s}\) is the orthogonal projection onto all additive
  image-independent per-style corrections. Its explained squared norm is
  therefore the finite-sample optimum for this entire correction class.
- **Result:** across 11 Qwen2.5-VL-7B lineages, per-style offsets explain only
  7.45%--8.35% of final prompt-token displacement; 91.65%--92.55% remains.
  Per-image offsets explain 74.61%--75.60%. At final image tokens,
  style-only offsets explain 13.27%--14.09%.
- **Implication:** an image-independent additive source-center displacement
  or global style direction is too restrictive for these observed fields.
  This is a more stable explanation of previous additive center-method
  failures than the non-replicated seed-42 contraction.
- **Limit:** stored float16 activations, synthetic styles, exposed MIMIC
  development images, one prompt/model family, Euclidean squared
  displacement, correlated checkpoints, and no downstream utility claim.
- **Evidence:** `corrected_runs/style_field_factorization_v1/`.

## 2026-07-31: Conditional style-field and null-prior audit

- **Question:** after rejecting a reusable global style offset, does the
  residual field have a low-complexity image-conditioned structure that could
  support a single-view correction?
- **Exact result:** at the final prompt token, centered case effects explain
  68.20%--69.31% of displacement energy across all 11 lineages, centered style
  effects only 1.42%--1.53%, and interaction 22.97%--23.91%.
- **Transductive structure:** a crossed held-cell predictor using the same
  case's other five style views explains 63.78%--65.21%. It shares the clean
  origin of the held cell and is not a deployable estimate or formal ceiling.
- **Single-view test:** patient-grouped nested linear-kernel ridge prediction
  from the clean state reduces error over patient-LOO style means by only
  0.38%--1.21%. This is little incremental prediction by the tested metric;
  no general actionability claim is supported.
- **Null-alignment observation:** the same-case direction from the real state to its
  null-image state captures 12.72%--13.49% of final-prompt displacement;
  72.5%--76.25% of cells point toward null. Every lineage exceeds its
  case-permuted 95% control (finite permutation \(p=1/201\)).
- **Decisive endpoint control:** a leave-one-patient clean-centroid direction
  explains 22.95%--23.94%, more than null in every lineage, and
  91.25%--93.75% of cells point toward it. Centroid-unique energy is
  15.39%--16.21%, versus 5.03%--5.70% null-unique.
- **Interpretation:** synthetic acquisition style induces a predominantly
  case-conditioned contraction field. The null direction is associated but is
  not specific enough to identify a clinical prior. The final counterfactual
  experiment must residualize generic centroid contraction before claiming
  style-specific prior switching.
- **Decision:** do not launch a single-view conditional operator from this
  branch. Hand the null-prior direction result to the independent \(2\times2\)
  content/style experiment.
- **Limits:** 40 exposed MIMIC development images, six synthetic Fourier
  styles, one prompt, stored float16 states, correlated checkpoints, and no
  output-utility or natural-scanner claim.
- **Evidence:** `corrected_runs/conditional_style_field_v1/` and
  `docs/CONDITIONAL_STYLE_FIELD_AUDIT.md`. Same-family result-to-claim review:
  `partial`, high confidence, trace
  `.aris/traces/result-to-claim/2026-07-31_run05/`.

## 2026-07-31: Layerwise centroid-contraction onset

- **Design:** the full five-layer activation probe was extended from the three
  exploratory lineages to all 11 lineages using the same 40 images, six
  styles, prompt, and float16 capture protocol.
- **Centroid alignment:** median normalized prompt-state energy projected toward a
  leave-one-patient clean centroid is 18.97%, 19.38%, 20.30%, 26.44%, and
  23.61% at LLM layers 0, 7, 14, 21, and 27.
- **Direct distance correction:** positive projection is not sufficient for
  contraction. Median mean log after/before squared centroid-distance ratios are
  +0.414, +0.182, +0.150, -0.095, and -0.106. Every lineage moves farther at
  layers 0/7/14 and closer only at layers 21/27. Median cellwise contraction
  rates rise from 17.92% to 62.50%.
- **Normalized token comparison:** centroid projection at the prompt token exceeds pooled
  image-token projection at every sampled layer in every lineage. Depending
  on layer, 92.08%--98.33% of prompt case-style cells point toward the clean
  centroid. Different summaries and denominators prevent an absolute
  prompt-versus-image amplification claim.
- **Null control:** null-endpoint alignment is weaker than centroid alignment
  at every layer and lineage. It is below the case-permuted control at layer 0
  but exceeds it from layer 7 onward.
- **Conclusion:** the stable representation property is a nonmonotonic,
  late-layer onset of generic case-anchored contraction, not a demonstrated
  clinical-prior switch or causal language-fusion amplification. Output-level
  prior-switching tests must residualize this contraction.
- **Evidence:** `corrected_runs/layerwise_attractor_v1/`.

## 2026-07-31: Complete-sentence clinical-evidence contraction

- **Design:** six complete positive/negative disease sentences define a
  teacher-forced clinical evidence vector. The exact Huatuo and Qwen base
  models were compared on the same 64 exposed MIMIC development images (58
  patients) and six fixed Fourier styles.
- **Direct distance:** Huatuo's mean log after/before squared distance to a
  patient-LOO clean evidence centroid is -0.161, patient-cluster 95% CI
  [-0.316, -0.022], and 65.89% of cells become closer. Qwen base has a
  same-direction point estimate of -0.113, CI [-0.255, +0.042], and 60.68%
  closer.
- **Endpoint control:** clean-centroid directions explain 43.23% (Huatuo) and
  39.76% (base) of style displacement, exceeding null directions at 25.85%
  and 30.43%.
- **Lineage comparison:** paired Huatuo-minus-base difference -0.048, 95% CI
  [-0.241, +0.147]; medical-tuning amplification is unsupported.
- **Scale and heterogeneity:** \(\exp(-0.161/2)=0.923\), about 7.7% geometric
  Euclidean-distance contraction. The patient-balanced sensitivity remains
  below zero, but two of six Huatuo style-specific intervals cross zero.
- **Conclusion:** synthetic style induces generic contraction even in a
  clinically interpretable complete-sentence evidence coordinate. A
  style-conditioned prior-switching mechanism must explain residual
  style-specific disease drift beyond this nuisance effect.
- **Limits:** teacher forcing, six findings, synthetic styles, exposed
  development data, a target-transductive patient-LOO centroid, no generated
  accuracy/factuality, source-only inference, or natural-scanner claim.
- **Evidence:** `corrected_runs/clinical_evidence_attractor_v1/`.

## 2026-07-31: Residual clinical style signature

- **Question:** after removing generic clean-centroid contraction, do fixed
  source-spectrum styles retain reproducible disease-evidence directions?
- **Estimator:** each style displacement is residualized against its
  patient-LOO clean-centroid direction and then centered within case across
  styles. Style prototypes are trained on other patients only; the null
  independently permutes the six fixed style labels within patient.
- **Magnitude:** the residual style term explains 1.02% of Qwen-base and
  1.12% of Huatuo residual energy. Case terms remain dominant at 76.65% and
  69.46%.
- **Cross-patient signal:** held-patient six-way style identification is
  27.08% for base and 21.61% for Huatuo, against chance 16.67%;
  patient-blocked permutation \(p=.001\) and \(p=.034\). Prototype
  \(R^2_0\) is only 2.33% and 0.97%, although both exceed permutation
  (\(p=.001\)).
- **Cross-checkpoint signal:** matched style directions have mean cosine
  0.579 and exact \(6!\)-assignment \(p=.0111\). Disease-wise whitening gives
  cosine 0.582 and \(p=.0069\).
- **Interpretation:** a weak cohort-conditional spectral signature survives
  generic contraction and shows aggregate alignment across two paired
  checkpoints; this is not evidence that every style direction transfers,
  of an independently replicated architecture-wide mechanism, of a
  medical-training-specific clinical prior, or of an effective decoder. Raw
  Huatuo signatures are smaller for all six styles, but that ordering is
  scaling-sensitive and cannot support a causal tuning claim.
- **Decisive handoff:** the independent content-removed \(2\times2\)
  experiment must reproduce the frozen exported disease directions. If it
  does not, style-conditioned prior switching is falsified; if it does but
  generated utility is absent, no mitigation claim follows.
- **Limits:** exploratory multiple metrics, teacher forcing, six findings,
  six synthetic styles, 64 exposed MIMIC development images/58 patients,
  target-transductive centroid, two checkpoints, no natural scanner or
  generated-accuracy result. Huatuo style-ID \(p=.034\) is nominal and does
  not survive a two-endpoint Bonferroni correction.
- **Evidence:** `corrected_runs/residual_style_signature_v1/`.

## 2026-07-31: Disease-specificity of the residual style signature

- **Question:** is the residual signature only a uniform tendency to raise or
  lower all six disease scores?
- **Orthogonal split:** the literal all-ones disease axis accounts for 24.64%
  of base and 20.81% of Huatuo style-signature energy. Patient-cluster 95%
  intervals are [10.14%, 44.73%] and [5.82%, 41.09%], with more than 99% of
  resamples below one half.
- **Dimensionality caution:** the contrast space has five dimensions. Uniform
  energy per dimension is 1.63x/1.31x the average contrast-coordinate energy,
  so the result rejects a pure uniform shift but does not show per-dimension
  contrast dominance.
- **Contrast-only signal:** after removing that axis, held-patient six-way
  style identification is 25.78% for base and 22.14% for Huatuo (chance
  16.67%; blocked-permutation \(p=.001/.017\)). Prototype \(R^2_0\) is
  3.32%/1.30% (\(p=.001\) both).
- **Paired-checkpoint alignment:** contrast-only matched-style cosine is
  0.727, patient-cluster 95% CI [0.350, 0.792].
- **Important downgrade:** identity is uniquely best among all \(6!\) style
  and disease assignments only in the fixed cohort. Patient-bootstrap
  identity-margin intervals are [-0.231, 0.095] and [-0.189, 0.076], so a
  population-unique style-to-disease mapping is unsupported.
- **Rank:** observed entropy effective rank is 2.43--2.58, but it is not
  unusually low under patient-blocked style permutation.
- **Joint nuisance correction:** radial and uniform projections do not
  commute. Directly projecting onto
  \(\operatorname{span}\{r_i,\mathbf1\}^{\perp}\) retains style ID
  28.13%/24.74% and \(R^2_0=5.04\%/1.32\% (base/Huatuo; all permutation
  \(p=.001\)). Matched-style cosine is 0.699, patient-cluster CI
  [0.293, 0.790]; matched disease-profile cosine is 0.717. Identity margins
  remain unstable under patient resampling.
- **Conclusion:** the cohort signal is not reducible to a purely uniform
  answer bias and has positive aggregate disease-contrast alignment across
  paired checkpoints. This still does not identify a low-rank clinical prior,
  causal medical-tuning effect, natural acquisition mechanism, or useful
  decoder.
- **Evidence:** `corrected_runs/style_prior_specificity_v1/`.

## 2026-07-31: Equivalent-language style-prior gate

- **Question:** does the joint-nuisance residual survive semantically
  equivalent question and complete-answer wording?
- **Design:** 16 paired HuatuoGPT-Vision development patients, six diseases,
  and six fixed styles. The `evidence present/absent` frame equalizes
  positive/negative answer token counts.
- **Direction:** original-to-`demonstrates` matched style cosine is 0.570
  (patient-bootstrap 95% CI [0.014, 0.746]); original-to-`evidence` is 0.456
  ([0.051, 0.619]). The residual is therefore not wholly explained by the
  original negation length.
- **Identification failure:** held-patient cross-template style
  identification is 17.7% and 21.9%, against 16.7% chance
  (`p=.379/.058`). Both identity-versus-best-mismatch margins are negative.
- **Decision:** the frozen gate failed, so the base checkpoint was not run.
  The evidence supports weak template-robust susceptibility, not a stable
  style-indexed clinical-prior switch or a mitigation method.
- **Post-hoc frozen-direction sensitivity:** matched-minus-mismatched margins
  using the old 64-case reference are `0.142` for original
  (CI `[0.045,0.242]`, `p=.022`), `0.161` for `demonstrates`
  (`[0.025,0.292]`, `p=.003`), and `0.098` for equal-length `evidence`
  (`[-0.006,0.195]`, `p=.054`). Both paired 50%-retention intervals include
  zero, so this requested sensitivity cannot reverse the gate.
- **Evidence:** `corrected_runs/style_prior_template_probe_v1/`.
