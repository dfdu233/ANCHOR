# ASCC-v2.1 blind analysis and collision lock

## Scientific ceiling

The primary result is a controlled reader-vote interaction screen, not a
hallucination, open-ended generation, patient-truth, or latent-uncertainty
result. VinDr 1/3 and 2/3 labels are called a **panel-undetermined proxy**.
They do not show that any reader explicitly selected an uncertain state.

The maximum claim before clinical admission is:

> In a fixed-prefix, restricted three-state assay, replacing *findings* with
> *abnormalities* changes definite-versus-panel-undetermined response odds more
> near both non-unanimous reader boundaries than at their matched unanimous
> endpoints, conditional on the fixed parent/child proposition.

## Blind repair record

The Huatuo 2x2 scorer completed 2,036/2,036 registered primary jobs. No score
value was inspected while the v2.1 analyzer was repaired. An independently
created v2 analysis and progression file were discovered after repair. Only
their provenance metadata was read; both are hash-sealed in
`INVALIDATED_PRE_BLIND_REPAIR_ANALYSIS_V2.json` and cannot support a scientific
claim.

The sole valid replacement is
`ascc-symmetric-factorial-analysis-v2.1-blind-locked`, writing
`primary_analysis_v2_1_blind_locked.json` once.

## Frozen computational gates

- independent image bootstrap within child-vote, parent-vote, and aspect
  strata;
- two local reader boundaries, 1/3 minus 0/3 and 2/3 minus 3/3;
- positive noun effect separately for `Describe` and `List`;
- 90% CI equivalence within +/-0.2 for local and absolute polarity shifts,
  separately by speech act;
- conditional three-marker uncertainty-mass loss and panel-state-proxy Brier
  worsening in both ambiguous bins;
- gauge-invariant cross-fold affine nuisance fit on clear bins with one common
  positive slope and endpoint-specific biases;
- affine refit inside every bootstrap draw, at least 99% valid draws, held-out
  endpoint R2 >=0.50, absolute clear commitment bias <=0.2, and RMSE <=0.5;
- raw and affine-residual DID at least log(1.5), with 95% CI above zero;
- full-vocabulary top-1 belongs to the restricted marker set in at least 90%
  of every final-layer prompt-by-vote cell;
- noun-effect direction agrees in parent-vote 2/3 and 3/3 sensitivities.

The last item is a sensitivity, not a repair for the fixed parent assertion.
A no-parent run remains mandatory.

## Construct gates still external to this run

1. At least three radiologists must judge the complete prefix and three
   continuations for polarity, certainty strength, uncertainty referent, and
   naturalness. Bare `Pneumonia is uncertain` is currently semantically
   ambiguous.
2. Independent radiologists must establish that explicit radiograph-only
   indeterminacy is enriched in official 1/3 and 2/3 bins relative to both
   unanimous endpoints.
3. A no-parent-prefix sensitivity must separate reader-support interaction from
   text-image conflict induced by `parent is present`.
4. Natural open-ended answers must reproduce the effect without shortening,
   deletion, refusal, polarity, or coverage exchange.

Until all four pass, `promotion_authorized` is always false regardless of the
computational screen.

## Mechanism-level collision map (retrieved 2026-08-03)

No mechanism-equivalent work was retrieved under searches spanning reader
disagreement, prompt framing, three-state commitment, medical VLMs, verbal
uncertainty, and causal intervention. Every individual axis is crowded:

| Work | Occupied mechanism/setting | Delta left for ASCC | Fatal collision condition |
|---|---|---|---|
| [Mind the Uncertainty in Human Disagreement, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32468) | VQA human-disagreement distributions and calibration | no fixed clinical proposition or framing intervention | only showing worse calibration on disagreement cases |
| [HaDola, ICLR 2026](https://openreview.net/forum?id=LuZjiUNuFL) | yes/maybe/no human uncertainty labels and selection | no framing-conditioned causal transition | only disagreement-aware training/calibration |
| [CheXthought](https://arxiv.org/abs/2604.26288) | multi-reader CXR disagreement prediction and communication | no controlled support-to-language framing transition | only adding reader disagreement supervision |
| [Prompt-Induced Hallucination, ACL 2026](https://aclanthology.org/2026.acl-long.1941/) | prompt pressure by visual difficulty and copying heads | no independent reader votes or fixed three-state proposition | generic prompt pressure explains the interaction |
| [Tinted Frames](https://arxiv.org/abs/2603.19203) | framing-to-attention-to-answer causal chain | no reader-disagreement-gated commitment readout | framing changes visual evidence/attention first |
| [PARC, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Schmalfuss_PARC_A_Quantitative_Framework_Uncovering_the_Symmetries_within_Vision_Language_CVPR_2025_paper.html) | prompt/vision reliability and calibration | no human-disagreement interaction | result is generic prompt sensitivity |
| [VUF, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.187/) | a generic verbal-uncertainty feature and intervention | no externally grounded framing interaction | one generic confidence direction explains all bins |
| [Calibrating Expressions of Certainty, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/66b35d2e8d524706f39cc21f5337b002-Abstract-Conference.html) | certainty phrase simplex and medical-report calibration | no multimodal framing mechanism | phrase calibration alone closes the effect |
| [CertainlyUncertain, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/21b5788d81f886ff81671379b4ff9453-Abstract-Conference.html) | answerability and epistemic/aleatoric uncertainty training | no reader-vote framing interaction | only a third-state benchmark or SFT contribution |
| [Antidote, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Antidote_A_Unified_Framework_for_Mitigating_LVLM_Hallucinations_in_Counterfactual_CVPR_2025_paper.html) | counterfactual presupposition and DPO mitigation | no graded independent reader evidence | diagnosis injection or generic DPO explains gains |
| [MM-R3, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.246/) | equivalent-prompt consistency | no reader-calibrated pragmatic commitment | consistency regularization is the whole method |

The remaining high-value mechanism is therefore not a new uncertainty
direction. It is an **evidence-dependent gain on the recruitment of an existing
verbal-confidence actuator**. This survives only if perception, proposition
identity, polarity, coverage, length, and global confidence stay invariant and
late cross-prompt patching changes commitment selectively in non-unanimous
reader bins.

## Positive and negative endpoints

- Strong positive: two models and most qualified findings reproduce the local
  interaction; natural OE and physician language distributions agree; visual
  evidence is prompt-invariant; a late intervention selectively changes
  commitment without changing content or polarity.
- Generic prior: noun effects are constant across vote bins or persist under
  text-only/image swap.
- Presupposition: claim identity, polarity, or content changes.
- Visual reframing: visual attention/evidence changes before commitment.
- Generic confidence circuit: VUF-like intervention explains all bins without
  a disagreement interaction.
- Perception-limited: independent reader support is not directionally encoded.

Only the strong-positive endpoint could motivate a reader-calibrated decoder.
All negative endpoints are publishable boundary results but terminate the
mitigation claim.
