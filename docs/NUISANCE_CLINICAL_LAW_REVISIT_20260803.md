# From style sensitivity to a nuisance × clinical-law mechanism

Date: 2026-08-03  
Decision: **no GPU candidate survives the current CPU and collision gates**

## 1. Start from the phenomenon, not the source-domain story

The observation is narrow: changing the display style of one medical image can
change a VLM answer. It does not imply a training-source center, domain
retrieval, or a universal style direction. A scientifically useful successor
must name:

1. a measurable nuisance intervention (T);
2. a clinical law describing how a claim should transform under (T);
3. independently defined clinical truth;
4. a directional violation that cannot be reduced to clean margin or generic
   instability.

This rules out “style drift predicts hallucination” as a research question.
The local audit already showed that mild transforms flip at most 3.13% of
binary decisions and that clean margin predicts errors better than style drift.

## 2. Minimal candidate: Radiodensity-Signed Tone Substitution

### Clinical law

A strictly monotone global tone map changes presentation but preserves pixel
rank and spatial organization. It should not create or remove a finding. If a
model nevertheless treats global brightness as clinical evidence, its drift
should follow the physical sign of the queried finding:

- opacity family: consolidation, effusion, edema, atelectasis and lung opacity;
- lucency family: pneumothorax, emphysema, hyperinflation and subcutaneous gas.

Define

\[
s_i=m_i(\gamma=0.9)-m_i(\gamma=1.1),
\]

where (m) is the frozen Yes-minus-No margin and (gamma=0.9) is the brighter
render. The proposed shortcut requires

\[
\Delta_{tone}=E[s\mid opacity]-E[s\mid lucency] > 0
\]

after stratifying by Yes/No truth. This is not a source-domain hypothesis. It
is a signed nuisance interaction with a radiographic claim family.

### Outcome-blind CPU screen

The screen reused the previously frozen 128-sample Huatuo RULE/MIMIC cache. No
model was rerun and no finding labels were changed after observing gamma
scores. Mixed claims such as “effusion or pneumothorax” were excluded.

| Quantity | Result |
|---|---:|
| Eligible opacity claims | 49 |
| Eligible lucency claims | 16 |
| Opacity Yes / No | 28 / 21 |
| Lucency Yes / No | 3 / 13 |
| Raw opacity-minus-lucency interaction | 0.00485 |
| Truth-stratified interaction | **-0.00544** |
| Bootstrap 95% CI | **[-0.03687, 0.02434]** |
| Within-truth permutation (p) | **0.8214** |
| Frozen minimum effect | 0.05 |

The candidate fails every material gate: the point estimate has the wrong
direction, its interval includes zero, its magnitude is roughly one tenth of
the minimum effect, and the lucency-positive cell has only three cases. A
larger GPU rerun of the same formulation is not justified.

Reproduction:

```bash
PYTHONPATH=. .venv-full/bin/python \
  anchor/corrected_sgta/audit_radiodensity_tone_interaction_v1.py \
  --raw corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/raw.jsonl \
  --output corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/radiodensity_tone_screen_v1.json
```

## 3. Fatal-collision map

| Neighbor | What it already establishes | Consequence here |
|---|---|---|
| [VCD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html) | Distorted visual inputs amplify frequent/co-occurring hallucinations; contrast original and distorted logits for mitigation. | Generic “degradation exposes language prior” and distortion contrastive decoding are occupied. |
| [Perceptual Hallucination, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1237/) | Constructs original/damaged pairs, removes answer-relevant evidence, and analyzes vision-encoder errors propagating through the decoder. | Information loss causing prior-filled output is no longer a new mechanism without a medical law beyond damage granularity. |
| [LENS, active 2026 submission](https://openreview.net/pdf?id=oh3c2ieVab) | Uses a cross-view stability gap from semantics-preserving medical counterfactual views for token-level hallucination mitigation. | Raw medical view consistency or gamma-contrast decoding is directly crowded. |
| [CORAL, arXiv 2026](https://arxiv.org/abs/2607.03647) | Audits medical VLM grounding with blank, shuffled, absent and retrieved hard-negative images; trains against answer invariance. | Image-use and text-shortcut interpretations require stronger controls than an accuracy change under style. |
| [CounterVHD, arXiv 2026](https://arxiv.org/abs/2606.28520) | Uses factual/counterfactual entity grounding uncertainty to detect unsupported clinical entities. | A counterfactual-view detector is not a novel endpoint. |
| [RoentMod, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02497-6) | Uses radiologist-vetted synthetic CXR counterfactuals to identify and mitigate pathology co-occurrence shortcuts. | A clinical shortcut claim must isolate the exact nuisance/claim interaction, not merely show a changed prediction. |
| [DICOM LUT preprocessing study](https://pubmed.ncbi.nlm.nih.gov/39890738/) and [contrast-level pneumothorax study](https://pubmed.ncbi.nlm.nih.gov/36698035/) | VOI-LUT, histogram equalization, contrast and format materially affect CXR classifiers. | Tone sensitivity itself is an established medical-imaging robustness problem, not a VLM hallucination contribution. |
| [MIRP / Your other Left!, MICCAI 2025](https://papers.miccai.org/miccai-2025/1027-Paper0530.html) | Uses rotations/flips and relative-position controls to reveal medical VLM reliance on anatomical priors. | Mirror/rotation × laterality equivariance is directly occupied as a benchmark problem. |
| [MIOH, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Min_Fine-Grained_Multi_Image_Object_Hallucination_Benchmark_CVPR_2026_paper.html) | Crosses object attributes/positions with visual context scale and perceptual difficulty. | Generic scale/difficulty interactions need a new clinical law and independent truth to remain novel. |

The only possible novelty in the screened candidate was the *signed* opacity
versus lucency interaction. The CPU result does not support it.

## 4. NO-GO tree

```text
same image, changed presentation -> changed answer
|
+-- Is the transform clinically equivalent?
|   +-- no / not independently verified
|   |   `-- ordinary perception change; cannot label hallucination
|   `-- yes
|       |
|       +-- Is there a known action on clinical claims?
|           |
|           +-- monotone tone x opacity/lucency sign
|           |   `-- CPU REJECT: -0.0054, CI crosses 0, p=0.821
|           |
|           +-- mirror/rotation x laterality
|           |   `-- COLLISION: MIRP; local VinDr spatial truth also inadequate
|           |
|           +-- resolution garbling x lesion scale
|           |   `-- HOLD/NO-GO: Perceptual Hallucination + MIOH collision;
|           |       VinDr box extent is not independent visibility truth
|           |
|           `-- render x prompt-equivalence composition
|               `-- already CECD; requires blinded transform/prompt admission,
|                   not a new source-style branch
|
`-- Is only raw disagreement left?
    `-- BASELINE ONLY: margin, LENS/VCD and generic robustness explain the object
```

## 5. What would reopen the branch

Do not reopen it by adding transforms or trying another style bank. Reopening
requires a new clinical law with all of the following available *before* model
scores:

1. an intervention whose clinical action is deterministic or independently
   reader-admitted;
2. at least two opposed claim classes with at least 20 cases in every
   truth/support cell;
3. a directional interaction of at least 0.05 logit units whose 95% clustered
   interval excludes zero after clean-margin adjustment;
4. replication in two model families;
5. a contribution not reducible to VCD/LENS-style view contrast, MIRP spatial
   bias, or Perceptual Hallucination's damaged-evidence propagation.

No current public/local substrate satisfies all five. The correct decision is
therefore to retain style transforms as nuisance controls and not spend GPU on
a renamed style mechanism.
