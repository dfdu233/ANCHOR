# CECD 2026-08 collision update: what still survives

## Frozen question

CECD may test only the following narrow claim:

> After independent clinicians admit radiographic-render equivalence and clinical-wording equivalence, does the render-by-wording product-orbit interaction predict reader-grounded clinical polarity error beyond both marginal effects, generic two-axis instability, length/entropy, system-attention bias, and behavioral synergy controls?

The project must not claim that prompt sensitivity, visual-language imbalance, pre-generation hallucination prediction, confidence calibration, or generic activation steering are new.

## New closest-work collisions

1. **Medical prompt dependence is already explicit.** *Medical Context Distorts Decisions in Clinical Vision Language Models* manipulates image-text alignment, irrelevant history, and prompt formulations on MIMIC-CXR and reports text dominance and answer reversals under minor prompt changes ([arXiv:2605.17436](https://arxiv.org/abs/2605.17436)). *Benchmarking and Mitigating the Impact of Noisy User Prompts in Medical VLMs via Cross-Modal Reflection* contributes a medical prompt-noise benchmark and SFT mitigation ([EACL 2026](https://aclanthology.org/2026.eacl-industry.67/)). Therefore prompt robustness alone is occupied.

2. **Prompt-induced hallucination already has a head-level mechanism.** Rudman et al. identify a small, model-specific set of attention heads mediating prompt copying and reduce hallucination by at least 40% through ablation ([ACL 2026](https://aclanthology.org/2026.acl-long.1941/)). CECD must outperform a faithful prompt-copy/head or architecture-neutral causal control before making a mechanism claim.

3. **Yes-bias has a system-token explanation.** Chan et al. causally redistribute attention away from redundant system weights and suppress yes-bias ([Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1940/)). A CE-only gain without a system-attention control cannot be attributed uniquely to render-wording composition.

4. **Generic internal hallucination probes are occupied and architecture-dependent.** HALP predicts hallucination before generation with AUROC up to 0.93; the best layer/modality differs across architectures ([EACL 2026](https://aclanthology.org/2026.eacl-long.287/)). Thus no universal “early visual / late language” claim is allowed.

5. **Medical confidence calibration is occupied.** ConRad trains report- and sentence-level verbalized confidence using a proper logarithmic scoring reward ([arXiv:2603.29492](https://arxiv.org/abs/2603.29492)). Reader-grounding, frozen-model mechanism evidence, and content-preserving inference intervention—not claim-level confidence alone—must distinguish our work.

6. **Length and omission are fatal confounds, not auxiliary metrics.** HalluCXR reports response length alone predicting hallucination with AUROC up to 0.908 and an ensemble reducing fabrication while increasing omission ([arXiv:2605.20469](https://arxiv.org/abs/2605.20469)). Every OE result therefore requires matched length, fixed positive-claim count or matched claim coverage, omission recall, refusal rate, and an explicit no-deletion audit.

7. **Human perceptual equivalence is not optional.** Recent perceptual-hallucination work uses human adjudication and notes that visual recognition variability can masquerade as model error; its activation patching is limited to unanimously hallucinated samples ([Findings ACL 2026 paper](https://aclanthology.org/2026.findings-acl.1237.pdf)). This supports, but does not replace, our independent four-role admission.

8. **Evidence-bounded minimal editing is already an ACL method.** CEBC generates first and then minimally edits or suppresses visually unsupported object mentions under conformal constraints from an external detector ([ACL 2026](https://aclanthology.org/2026.acl-long.2142/)). Consequently, “commitment bounded by evidence” or minimal claim editing is not itself a contribution. Reader-distribution supervision, internal causal localization, and fixed-K/no-deletion behavior would all have to add something CEBC does not test.

9. **Source decomposition and adaptive steering are crowded.** HalluTrace separates visual-grounding failure, language-prior dominance, and cross-modal conflict through component ablations ([ALVR 2026](https://aclanthology.org/2026.alvr-main.29/)); VLI combines conflict diagnosis, visual-anchor localization, and instance-specific bi-causal steering ([ACL 2026](https://aclanthology.org/2026.acl-long.1784/)). A generic perception-vs-prior taxonomy or adaptive visual steering is therefore occupied.

10. **A “better VCD corruption” is not a sufficient pivot.** Object-aligned VCD already masks salient evidence to construct a more informative auxiliary view ([EACL SRW 2026](https://aclanthology.org/2026.eacl-srw.2/)). Any medical rendering intervention must be justified by clinically admitted equivalence and reader-grounded mechanism evidence, not merely stronger contrast.

11. **Model uncertainty cannot be treated as visual evidence.** A 2026 gastrointestinal VLM benchmark reports “confident confabulation”: hallucinations can remain consistent across samples and carry high token probability, defeating consistency- and uncertainty-based detectors ([arXiv:2606.24115](https://arxiv.org/abs/2606.24115)). Hence entropy, self-consistency, and verbal confidence are controls only; independent reader support and image-sensitive directionality must define evidence.

12. **Generic modality-conflict localization and latent steering are already occupied.** *Mechanistic Analysis and Inference-Time Control of Modality Conflict in VLMs* introduces a controlled image-versus-statement conflict substrate, reports architecture-dependent vision/text preference, causally characterizes the internal conflict mechanism, and applies inference-time representation control ([ICML 2026 Mechanistic Interpretability Workshop](https://mechinterpworkshop.com/posters/virtual/); [paper PDF](https://openreview.net/pdf/67007e9d31075e359c5a4ed1dc15a4ea829e6efd.pdf)). It therefore kills any generic claim that CECD is the first to localize a fusion-stage modality-preference direction or steer it. The remaining delta is conjunctive and narrower: both operations are independently clinician-admitted as meaning-preserving rather than deliberately conflicting; the target is reader-distribution clinical-loss orientation of their product-only residual; and causal rescue must preserve the interaction spectrum, activation norm, and both marginals. A conflict-direction score and matched steering arm are mandatory alternatives if the latent branch opens.

## Consequences for the experimental contract

- The 160-claim `pilot_screen` is engineering-only and can neither confirm nor reject the mechanism.
- `dev_fit` may fit scales and predictors but cannot authorize the method.
- Only the image-disjoint 960-claim `confirmation_locked` stage may test the frozen two-model gate.
- The confirmation baseline must include clean score, entropy, prompt/response length, both marginal effects, generic render/prompt/full-grid stability, and behavioral synergy. If hidden-state work opens, system-token attention redistribution and prompt-copy controls become mandatory.
- Any later mitigation comparison must include CEBC-style evidence-bounded editing and instance-specific conflict/steering controls where model compatibility permits. Otherwise the result is a narrow mechanism paper, not a mitigation paper.
- A layerwise orientation result must add value beyond a dev-fitted explicit-modality-conflict direction and its matched inference-time steering control. Otherwise the result is a medical task transfer of an occupied generic mechanism.
- VinDr listing is a closed 14-finding, open-cardinality transfer task—not unrestricted OE. Its primary result must be fixed-K or matched coverage, with positive hallucination and omission reported together.
- Any apparent improvement explained by shorter answers, fewer claims, more negatives, more hedges, or refusal is a failure.

## Current novelty verdict

**Conditional and narrow.** The product-orbit interaction with independent reader-vote truth remains potentially distinguishable, but only as an incremental clinical-error mechanism. The mitigation contribution is not yet established. Failure of the locked two-model confirmation should terminate CECD rather than trigger threshold repair.
