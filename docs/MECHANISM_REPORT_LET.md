# LET Mechanism and Novelty Audit

## Locked decisions

- **Problem:** determine whether positive intermediate-layer evidence mixing
  improves full-sentence medical VLM answers under the exact RULE/MIMIC
  protocol, and why it behaves differently from DoLa/VCD.
- **Mechanism:** late decoder layers may suppress useful early evidence;
  positive mixing can recover it, but can also amplify an affirmative language
  prior. The discriminating prediction is improved balanced accuracy, not just
  increased Yes recall.
- **Substrate:** LLaVA-Med 7B, RULE/MIMIC 3,470 questions, `vicuna_v1`, RULE
  normalized generated-sentence parser. Frozen greedy cache accuracy is
  2,622/3,470 = 75.56196%.

## Collision audit

LET uses

\[
\ell=(1-\alpha)\ell_L+\alpha\ell_e.
\]

VISTA (ICML 2025) already implements Self-Logits Augmentation as
`logits = logits_alpha * aug_logits + (1-logits_alpha) * logits`; see the
vendored official code at
`third_party/VISTA/minigpt4/models/modeling_llama.py:86-97` and
`third_party/VISTA/llava/model/language_model/llava_llama.py:92-103`.
LET differs by using one final-normed expert layer and a reverse-KL barycenter
interpretation. These are currently insufficient to establish a distinct main
algorithm. Any paper must cite VISTA and describe LET as a medical
specialization/simplification unless a mechanism-level contribution is shown.

## Protocol audit and pilot evidence

The initial manual decoder was rejected: at 128 samples its `alpha=0` output
was 3.91 percentage points below the frozen baseline, so apparent LET gains
were confounded by decoder implementation. The corrected runner alters logits
inside the official `model.generate` path; `alpha=0` reproduced all 16 smoke
labels.

On the preregistered 256-question prefix, L-12/alpha=0.30 produced:

| Metric | Frozen greedy | LET | Change |
|---|---:|---:|---:|
| Accuracy | 75.78% | 79.69% | +3.91pp |
| Balanced accuracy | 74.74% | 76.00% | +1.26pp |
| Recall | 79.87% | 94.16% | +14.29pp |
| True negatives | 71 | 59 | -12 |

There were 27 rescues and 17 harms; exact McNemar p=.174 and the
patient-cluster bootstrap interval was [0.00, 9.52]pp. This is a promising
accuracy pilot, not confirmatory evidence. The sensitivity/specificity tradeoff
supports the affirmative-prior failure mode.

The frozen configuration was then evaluated on all 3,470 rows. Under the
exact joined cache it improved 75.45% to 80.75% (+5.30pp), with 353 rescues,
169 harms, exact McNemar p=5.76e-16, and a corrected 218-patient cluster
bootstrap interval of [+4.10,+6.55]pp. Under the user's common 3,466-row table
convention, the headline comparison is 75.56% to 80.84% (+5.28pp). Balanced
accuracy increased by about 3.17pp under that common convention. The gain is
therefore real at the task-metric level, but asymmetric: sensitivity rises to
91.46% while specificity falls to 63.90%. This safety tradeoff remains a
central mechanism result rather than a removable reporting detail.

## Mechanism scorecard

| Criterion | Score (0-2) | Rationale |
|---|---:|---|
| Sharpness | 2 | Evidence recovery versus affirmative-prior amplification |
| Grounding | 2 | Exact baseline, existing LET runs, DoLa/VCD and VISTA code |
| Identification | 1 | Balanced accuracy separates mechanisms; null-image control pending |
| Feasibility | 2 | One-pass, training-free, full run is practical |
| Falsifiability | 2 | No BA gain or VISTA parity falsifies the proposed distinction |
| Originality | 0 | Direct operator collision with VISTA SLA |

**Total: 9/12, but the originality hard gate fails.** LET should not currently
be claimed as a novel AAAI algorithm. A defensible contribution would require
an explicit VISTA-SLA comparator and a medical-domain mechanism result showing
when normalized single-layer mixing improves visual grounding rather than
merely shifting answer prevalence.
