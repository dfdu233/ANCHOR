# Visual Evidence Chord Audit

## Question

Does a source-derived acquisition style act as a reproducible clinical-prior
switch in a medical VLM, or does it perturb each patient differently?

This is a mechanism audit, not an accuracy experiment. It uses complete
clinical sentences under teacher forcing to measure evidence and never uses
label-token logits as a prediction rule.

## Three competing laws

For six clinical concepts, let \(e_x\in\mathbb R^6\) denote the difference
between the mean log likelihood of a complete positive sentence and that of
the corresponding complete negative sentence. Let \(e_\varnothing\) be the
same vector for a fixed null image and \(e_s\) the vector after applying
source style \(s\).

The audit distinguishes:

1. **Scalar visual attenuation**

   \[
   e_s=e_\varnothing+a_s(e_x-e_\varnothing).
   \]

2. **Style-conditioned prior rotation**

   \[
   e_s=e_\varnothing+a_s(e_x-e_\varnothing)+b_s.
   \]

3. **Concept-selective filtering**

   \[
   e_s=e_\varnothing+D_s(e_x-e_\varnothing),
   \]

   where \(D_s\) is diagonal.

All parameters are evaluated leave-one-image-out. A full linear filter is
reported only as a capacity diagnostic.

## Protocol

- Model: HuatuoGPT-Vision-7B based on Qwen2.5-VL.
- Target development images: 64 fixed, unique MIMIC-CXR images.
- Primary subset: 40 images classified as frontal radiographs by the local
  BiomedCLIP audit.
- Styles: six PubMedVision strict-CXR source clusters.
- Transformation: smooth low-frequency log-amplitude transfer, radius
  \(0.12\), strength \(0.65\), with phase and DC preserved.
- Concepts: pneumothorax, pleural effusion, opacity/consolidation,
  cardiomegaly, edema, and devices.
- Controls: real image and fixed null image.
- Statistics: image-cluster bootstrap and within-image style-label
  permutation.

The frontal subset is primary because the source style bank was constructed
from frontal CXR. The 64-image result is a sensitivity analysis.

## Result: none of the three laws passed

| Quantity | All 64 | Frontal 40 |
|---|---:|---:|
| Median pixel correlation | .959 | .971 |
| Median edge correlation | .976 | .976 |
| Style shift / real-null distance | .240 | .224 |
| \(\cos(e_s-e_x,e_\varnothing-e_x)\) | .185 | .156 |
| Identity LOIO MSE | .001274 | .001086 |
| Scalar chord LOIO MSE | .001248 | .001074 |
| Style-offset LOIO MSE | .001172 | .001002 |
| Diagonal-filter LOIO MSE | .001241 | .001090 |

The intervention is therefore nontrivial and preserves coarse structure, but
the drift is poorly aligned with loss of visual evidence. A style offset
improves mean MSE by 6.7% on frontal images, yet its image-bootstrap interval
crosses zero. The concept-selective filter is worse than the scalar chord.
Consequently, scalar attenuation, reproducible prior rotation, and
concept-selective filtering all fail their frozen gates.

![Frontal visual-evidence chord audit](../results_reference/visual_evidence_chord_probe_v1_n64/chord_frontal.png)

## A sharper result: style identity explains little of the drift

For patient \(i\) and style \(s\), define

\[
\Delta_{i,s}=e_{i,s}-e_{i,\mathrm{real}}.
\]

The balanced Euclidean two-way decomposition is

\[
\Delta_{i,s}
=\bar\Delta+A_i+B_s+\Gamma_{i,s},
\]

with mutually orthogonal patient, style, and interaction terms. On the 40
frontal images:

| Component | Fraction of centered evidence-drift variance |
|---|---:|
| Patient | 73.43% |
| Source style identity | 3.28% |
| Patient \(\times\) style | 23.29% |

Thus, the same source style does not induce a uniform disease-prior shift
across patients. The response is dominated by the patient state and its
interaction with the transformation.

### Least-squares ceiling for a global source correction

For a fixed style, any global additive correction solves

\[
\min_b\mathbb E_i\|\Delta_{i,s}-b\|^2.
\]

By the Pythagorean identity, the unique optimum is

\[
b_s^\star=\mathbb E_i[\Delta_{i,s}],
\]

and the maximum removable fraction of drift energy is

\[
\rho_s=
\frac{\|\mathbb E_i\Delta_{i,s}\|^2}
{\mathbb E_i\|\Delta_{i,s}\|^2}.
\]

**Proposition (global-correction ceiling).** For any square-integrable
style-induced evidence drift \(\Delta_s\), no patient-independent additive
correction removes more than a \(\rho_s\) fraction of its expected squared
energy.

**Proof.** Write \(m_s=\mathbb E[\Delta_s]\). For every constant vector \(b\),

\[
\mathbb E\|\Delta_s-b\|^2
=\mathbb E\|\Delta_s-m_s\|^2+\|m_s-b\|^2,
\]

because the centered cross term has expectation zero. Hence \(b=m_s\) is the
unique minimizer, and the removed energy is
\(\mathbb E\|\Delta_s\|^2-\mathbb E\|\Delta_s-m_s\|^2=\|m_s\|^2\).
Dividing by total drift energy yields \(\rho_s\). \(\square\)

For the six source styles, \(\rho_s\) ranges only from 2.2% to 14.8%. This is
an empirical upper bound for a shared additive style-offset under the measured
evidence geometry. It explains why one global source center, NBP, and
source-level offset corrections repeatedly fail: most of the induced change
is not a reusable style main effect.

### The bank is heterogeneous

The small aggregate style term does not mean every source style is pure
noise. In 2,000 random patient split-halves, the mean drift direction was
reproducible for three of six clusters:

| Style | Median split-half cosine | 95% interval |
|---|---:|---:|
| 0 | .366 | [-.257, .852] |
| 1 | .283 | [-.330, .815] |
| 2 | .732 | [.255, .949] |
| 3 | .779 | [.404, .960] |
| 4 | .017 | [-.575, .760] |
| 5 | .839 | [.546, .972] |

This supports a narrower hypothesis: a subset of discrete source styles may
induce a stable but small prior displacement, while averaging all source
styles into one center destroys that structure. It does not yet show that
subtracting such a direction improves generated answers.

## Refuted alternative explanation

The per-image chord-orthogonal fraction was not significantly associated with
the number of positive findings, uncertainty phrases, report length, or
normal/abnormal status. The effect therefore cannot currently be summarized
as “complex or ambiguous cases are more style-sensitive.”

## What this does and does not establish

Supported on this exposed development subset:

> PubMedVision-derived source styles produce measurable but strongly
> patient-conditional changes in complete-sentence clinical evidence.

Not supported:

- a single source style selects a stable clinical prior;
- style primarily attenuates visual evidence;
- a global source center can uniformly remove the shift;
- teacher-forced evidence changes imply an accuracy improvement.

## Exact-architecture training-lineage control

The exact base Qwen2.5-VL-7B model was run on the identical 40 frontal images
from 38 patients, six styles, six concepts, complete-sentence prompts, and
inference budget.

| Quantity | HuatuoGPT-Vision | Base Qwen2.5-VL |
|---|---:|---:|
| Image/study variance | 73.43% | 79.86% |
| Style variance | 3.28% | 2.06% |
| Image/study \(\times\) style | 23.29% | 18.09% |
| Certified split-half styles | 3 / 6 | 1 / 6 |
| Median normalized style susceptibility | .229 | .329 |

The medical-minus-base style-fraction difference is only \(+1.22\) percentage
points, with paired patient-cluster-bootstrap 95% interval
\([-0.94,+3.66]\) points. This checkpoint pair therefore does not pass the
lineage criterion for a stronger style-conditioned prior.

Moreover, style 3 and style 5 induce similar directions in the two models
(cross-model cosine .961 and .932). This favors an architectural or
transformation-level origin for those small shared offsets.

### Medical style contraction

Define the dimensionless patient susceptibility

\[
\kappa_\theta(x)=
\frac{
\sqrt{\mathbb E_s\|
e_\theta(T_sx)-e_\theta(x)
\|^2}
}{
\|e_\theta(x)-e_\theta(\varnothing)\|
}.
\]

Huatuo has median \(\kappa=.229\), compared with \(.329\) for the exact base
model. The paired median difference is \(-.101\), with
patient-cluster-bootstrap 95% interval \([-.144,-.026]\). Under the tested
operator, the medical checkpoint has lower normalized acquisition-style
sensitivity than its exact base; it does not exhibit a stronger reusable
style-prior switch. This paired observational control cannot isolate medical
instruction tuning as the causal source of the contraction.

![Training-lineage comparison](../results_reference/visual_evidence_chord_lineage_v1/comparison_frontal.png)

This result helps explain the earlier intervention failures in this model
pair. The medical checkpoint is less sensitive to the tested style operator,
while the residual drift is primarily image-conditional. Global source-center
adaptation therefore has both little reusable signal and the wrong
factorization under this protocol.

## Relation to the July 2026 frontier

The broad claim that a VLM retains visual evidence but uses it unreliably is
no longer novel. [Seeing or Knowing?](https://arxiv.org/abs/2607.26326)
studies controllability between visual context and internal knowledge, while
[Positive-Negative Decoding](https://arxiv.org/abs/2605.06679) and
[VCD](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)
construct visual counterfactual branches for decoding. The remaining
mechanism question here is narrower: whether *medical acquisition style*
creates a cross-patient reusable clinical-prior displacement. The variance
decomposition directly tests that prerequisite and currently answers “mostly
no” for the tested PubMedVision style operator.

## Claim ceiling

Evidence grade: **C-level mechanism diagnostic** under the unified evaluation
contract. MIMIC images were exposed during method development, only one
checkpoint per lineage was analyzed, the six Fourier styles are fixed rather
than sampled from a deployment population, and the primary quantity is a
teacher-forced diagnostic rather than generated-answer accuracy. The
patient-cluster bootstrap accounts for the 40 images belonging to 38
patients, but it does not provide style-population inference.
