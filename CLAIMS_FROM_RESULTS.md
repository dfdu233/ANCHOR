# Claims from Center-Native Results

Verdict: **no**
Confidence: **high**
Review status: **provisional, same-family independent reviewer**

The completed development experiment does not support either pure inference
source-center calibration or a Center-Native training–inference benefit.

- Frozen vision: \(B-A=-3.65\)pp BA; \(D-C=+0.10\)pp.
- Early-vision fallback: \(B-A=+1.15\)pp with a confidence interval spanning
  zero; \(D-C=-4.48\)pp with a wholly negative patient-bootstrap interval.
- Closed-ended balanced accuracy is not sufficient evidence for a general
  clinical hallucination-reduction claim.

Supported statement:

> In this Qwen2.5-VL-3B PubMedVision-to-MIMIC development setting, fixed
> source-feature centering did not reliably improve closed-ended balanced
> accuracy. Center-aware early-vision training did not create a beneficial
> matching-center effect and could be harmful.

Final MIMIC, additional seeds, OE, center controls, and a success-framed model
release were correctly stopped after the preregistered continuation gate failed.

## Visual Evidence Chord / Style Contraction

Review verdict:

- **Claim A — medical training creates a stronger reusable
  acquisition-style prior: no.**
- **Claim B — the Huatuo checkpoint has lower normalized susceptibility to
  the tested style operator than its exact Qwen base: yes, narrowly.**

On 40 selected frontal MIMIC development images from 38 patients, six fixed
PubMedVision-derived Fourier transforms, six clinical concepts, and one
teacher-forced sentence contrast, the reusable style variance was 3.28% for
Huatuo and 2.06% for Qwen. The +1.22pp paired difference was uncertain
(patient-cluster-bootstrap 95% CI [-0.94, 3.66]pp).

Normalized style susceptibility was lower for Huatuo: median \(\kappa=.229\)
versus \(.329\), paired difference \(-.101\), 95% CI
\([-.144,-.026]\). The supported statement is:

> Under this fixed Fourier operator and exposed MIMIC development protocol,
> HuatuoGPT-Vision-7B shows lower normalized evidence drift than its exact
> Qwen2.5-VL-7B base, without a reliably larger reusable style main effect.

This is a checkpoint-pair mechanism diagnostic. It does **not** support:

- a causal effect of medical instruction tuning;
- natural-scanner or external-hospital robustness;
- accuracy, clinical correctness, or hallucination mitigation;
- generalization beyond the fixed styles, concepts, wording, and development
  images.

Review status: **provisional, same-family; medium confidence**. No independent
integrity audit or external replication is available.
