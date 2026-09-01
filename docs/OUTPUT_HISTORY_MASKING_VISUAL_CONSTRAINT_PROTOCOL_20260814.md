# Output-history masking for medical-VLM visual constraint

**Frozen on:** 2026-08-14  
**Scope:** white-box, inference-only, no training, no reference image pool, no disease-specific taxonomy

## Decision

There are close precedents for using prior output tokens to detect or mitigate
multimodal hallucination, but the checked literature does **not** establish direct
causal masking of output-history attention edges as a single-image estimator of
whether a medical image constrains the answer.

More importantly, output-history masking cannot be the primary visual-constraint
test: no generated history exists before the first answer token. In a long answer,
it measures autoregressive propagation/self-reinforcement, not the intrinsic
constraint supplied by the image. The primary intervention must therefore be a
visual-path intervention; output-history masking is a secondary source-separation
control for long-form answers.

## Closest literature and exact boundary

| Work | Relevant mechanism | Difference from the proposed test |
|---|---|---|
| [OPERA, CVPR 2024](https://arxiv.org/abs/2311.17911) | Penalizes over-trust in a few previous summary tokens and rolls decoding back; training-free | Uses attention pattern plus decoding penalty, not causal leave-history-out measurement of visual constraint |
| [PAS, CVPR 2026](https://arxiv.org/abs/2511.11502) | Detects hallucinated object tokens from attention to preliminary generated tokens; reference- and training-free | Observational attention score; no output-history edge intervention |
| [GACD, CVPR 2026](https://research.nvidia.com/labs/sil/projects/gacd/site/index.html) | Uses first-order gradients to partition influence among image, prompt, and past output tokens and rebalance decoding | Gradient approximation and guided decoding, not exact edge blocking; not medical or an image-constraint construct |
| [Hallucination Begins Where Saliency Drops, ICLR 2026](https://arxiv.org/abs/2601.20279) | Finds hallucination when preceding-output saliency drops and reinforces local coherence | Shows that low history dependence can also be harmful; invalidates a monotonic “less history is always safer” rule |
| [VIHD, 2026](https://arxiv.org/abs/2605.20772) | Medical VQA detection by intervening on visual tokens and using calibrated semantic entropy | Intervenes on visual tokens, not output history; detection rather than source-resolved token gating |
| [ACG, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Jo_Attention-Space_Contrastive_Guidance_for_Efficient_Hallucination_Mitigation_in_LVLMs_CVPRF_2026_paper.html) | Creates contrasted visual/text attention paths with attention masking in one pass | Close implementation precedent, but optimized as direct hallucination guidance rather than validating image constraint |
| [Same Attention, Different Truths, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html) | Shows attention mass alone is insufficient and tests representational/logit consistency | Supports using causal logit-margin changes rather than raw attention weights |

The novelty statement must remain bounded: after targeted searches for output-token
masking, response-history ablation, causal attention-edge masking, multimodal
hallucination, and medical VLMs, no exact primary precedent was found. This is not
proof that no unpublished or differently named work exists.

## Frozen construct

For the model's selected candidate answer token \(y_t\), use its margin over the
strongest alternative \(a_t\):

\[
m_t = \log p(y_t)-\log p(a_t).
\]

Run three teacher-forced passes with the **same image, prompt, response tokens,
positions, and token count**:

1. **Full:** normal model, margin \(m_t^F\).
2. **Visual-edge block:** current response position cannot attend to visual-token
   positions at every decoder layer, margin \(m_t^{-V}\).
3. **History-edge block:** current response position cannot attend to preceding
   generated-response positions, while visual and question positions remain
   visible, margin \(m_t^{-H}\).

Define

\[
V_t=m_t^F-m_t^{-V}, \qquad H_t=m_t^F-m_t^{-H}.
\]

- \(V_t>0\): the image causally supports the selected token.
- \(V_t\approx0\): the selected token does not need the image.
- \(V_t<0\): visual evidence opposes the selected token; this is the strongest
  candidate hallucination signal.
- \(H_t>0\): prior output supports the token, but this can represent either valid
  propagation of visual evidence or language-prior snowballing.

Therefore **\(H_t\) is never used alone**. The preregistered high-risk state is
\(V_t<0\) together with \(H_t>0\): history supports a token that vision opposes.
The continuous discovery analysis keeps \((V_t,H_t)\) separate; it does not tune
a weighted score on evaluation labels.

Do not replace response tokens with `[MASK]`, padding, or deletion. Those operations
change length, position, grammar, and move the model off-distribution. Preserve the
tokens and intervene only on attention edges. Do not use raw attention mass as the
primary score.

## Minimal experiment

### Experiment A — validate the image-constraint construct first

Use the existing 68 VQA-RAD instances from 34 exact-question, answer-changing
natural image pairs. Pair labels are used only to define the retrospective oracle;
the proposed score is computed per image without a pool.

For the first answer token, run only Full and Visual-edge block because there is no
output history. Test:

1. Spearman association between single-image \(V_t\) and the frozen natural-pair
   constraint oracle (between-image answer-distribution JS).
2. Error-detection AUROC with patient/image-pair bootstrap confidence intervals.
3. High- versus low-constraint quartile error rates.

Frozen baselines: full entropy, prompt-paraphrase JS, raw visual-attention mass,
no-image logit difference, and the existing prompt-robust margin. No threshold is
tuned in this experiment.

**Go criterion:** the sign is correct for the oracle association, AUROC exceeds
0.70, and the estimate is not inferior to prompt-robust margin. Failure means the
model's visual edge intervention is not a usable image-constraint estimator; do
not proceed to mitigation.

### Experiment B — test output-history masking only after A passes

Use frozen native open-ended medical-VQA/report outputs. Teacher-force the exact
same draft through Full, Visual-edge block, and History-edge block. Evaluate token
scores at factual content tokens and aggregate conservatively to each atomic claim
using its minimum \(V_t\) and maximum positive \(H_t\).

Primary test: grounded versus hallucinated claim AUROC for \(V\), \(H\), and the
two-dimensional high-risk state \(V<0,H>0\). Report results by response position to
distinguish initial visual under-constraint from later autoregressive snowballing.

Controls:

- block prompt-history rather than generated history;
- block a position-count-matched random set of permitted text edges;
- repeat with equal response length / claim-count strata;
- compare with OPERA, PAS, and gradient source attribution if their adapters are
  executable on the same model.

### Experiment C — inference-time mitigation

Only if B validates the state \(V<0,H>0\), apply online candidate rejection:

1. At each factual-token decision, score the original top-\(k\) candidates.
2. Set the logit of a candidate to \(-\infty\) only when vision opposes it
   (\(V<0\)) while output history supports it (\(H>0\)).
3. Select the highest remaining candidate; if all candidates fail, emit a calibrated
   uncertainty/abstention phrase rather than inventing content.

Primary mitigation endpoint: hallucinated-claim reduction at matched claim coverage.
Safety endpoints: omission, refusal, answer accuracy, clinical contradiction, length,
and latency. A shorter but less informative answer is not counted as success.

## Implementation notes

- Pilot offline with teacher forcing and disabled KV cache; this guarantees exact
  token alignment across interventions.
- Apply the edge mask in every decoder layer. Merely zeroing the final-layer
  attention or visual embeddings does not implement the causal question.
- In a decoder-only VLM, retain visual, question, and response embeddings. Change
  only the current response query's allowed key positions.
- Record both the selected-token probability and selected-versus-alternative margin;
  the latter is primary because it is less confounded by general fluency loss.
- The method is white-box but model-agnostic at the construct level. Each model
  still needs a small adapter that exposes visual and response token spans.

## Confirmed next action

Implement **Experiment A only** on HuatuoGPT-Vision-7B and the already frozen 68
instances. It is the fastest decisive test and directly answers the original
question: can one test image constraint from one image? Output-history masking is
then added only for long-form generation, where it has a well-defined causal role.
