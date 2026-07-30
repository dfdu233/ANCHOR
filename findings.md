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
