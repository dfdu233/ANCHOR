# Virtual-Reader Sufficiency Protocol

Status: preregistered mechanism screen; no mitigation is authorized by dev data.

## Question

Does a medical VLM need a separate latent `Maybe` coordinate to represent
reader disagreement, or can the fixed reader panel be reconstructed from a
one-dimensional signed image-evidence score and reader-specific thresholds?

This screen follows the rejected Two-Plane Clarity Erasure hypothesis.  It
does not reinterpret that negative result as support for the new hypothesis.

For FP32 next-token logits `Y`, `N`, and `U` for the fixed verbalizers Yes, No,
and Maybe, define the shift-invariant coordinates

```text
e = Y - N
m = U - log(exp(Y) + exp(N))
```

`e` is signed clinical evidence. `m` is the decoder's direct uncertainty
coordinate.  Reader votes are kept individually for the fixed VinDr panel
R8/R9/R10; a majority label is never substituted for the panel.

The reconstructed distribution is a posterior for this fixed pseudonymous
panel under this dataset. With only three readers it is not an estimate of a
radiologist population and is not clinical truth.

## Data boundary

- Dataset: VinDr-CXR training images with the exact R8/R9/R10 panel.
- Findings: the eight findings passing all four vote-bin quotas.
- Dev: 20 images per finding and vote count, 640 claim rows per model.
- Confirmation: 60 images per finding and vote count, 1,920 claim rows per
  model.
- The global image-hash split, prompts, verbalizers, and decoder layers are
  frozen before confirmation.
- Bootstrap units are images, because one image may contribute several
  findings. Reader rows are never treated as independent bootstrap units.
- The experimental manifest is deliberately balanced across vote bins. Model
  slopes and equivalence tests are reported on that balanced design, while
  probability calibration, Brier, NLL, and ECE additionally use frozen
  inverse-sampling weights from `summary_v2.json` to recover each split's
  observed finding-by-vote prevalence. Unweighted balanced metrics are never
  presented as population calibration.

## Competing models

All feature transforms and regularization choices are selected on dev only.

1. `M0`: shared positive evidence slope with finding and reader intercepts.
2. `M1`: flexible evidence-only model using a dev-selected low-degree spline.
3. `M2`: flexible `e + m + e*m`; tests whether direct Maybe adds information
   after the evidence response is modeled adequately.
4. `M3`: unconstrained evidence-only three-count model; tests whether the
   conditional-independence assumption of the virtual panel is adequate.
5. `M4`: a nested-CV residual probe for panel unanimity using layerwise hidden
   features in addition to flexible `e`, finding, and prevalence controls.

For virtual reader `r`, the threshold family has

```text
p_r = sigmoid(a_f * e + b_f + t_r), with a_f > 0.
```

The fixed three-reader panel induces

```text
P0   = product_r (1 - p_r)
P3   = product_r p_r
Pmid = 1 - P0 - P3
```

corresponding to refuted, undetermined, and supported reader states.

## Baselines

- finding-only empirical reader distribution;
- direct softmax over No/Maybe/Yes;
- dev-fitted scalar temperature scaling of the direct logits;
- evidence-only calibration without reader effects;
- `M0` through `M3` above.

The main comparison for `Maybe` sufficiency is `M2` versus `M1`, not `M2`
versus a linear model. Otherwise unmodeled nonlinearity in `e` could be falsely
attributed to `m`.

## Verbalizer-prior falsification

`Maybe` losing under one prompt can be a token-frequency artifact. Before
confirmation, dev freezes a control matrix containing:

- permuted answer-option order;
- matched-token anonymous A/B/C labels with the semantic mapping permuted;
- uncertainty paraphrases (`Maybe`, `Uncertain`, `Cannot determine`) whenever
  tokenizer length permits a matched comparison;
- a content-free prompt used only for prior subtraction;
- a binary Yes/No evidence prompt that never mentions an uncertainty option.

The signed-evidence ordering and the conditional redundancy conclusion must be
stable across semantic mappings. If suppression follows the surface token or
option position rather than the uncertainty meaning, the mechanism is
rejected as verbalizer bias.

## Latent residual gate

An external reader-threshold calibrator alone is not a representation
mechanism. The paper path therefore additionally asks whether any hidden-state
family predicts fixed-panel unanimity after a flexible evidence-only model.

- Analyze `0/3 versus 1/3` and `3/3 versus 2/3` as separate polarity strata;
  the pooled model contains a spline-of-`e` by stratum interaction. This
  prevents a probe from recovering unanimity merely by recovering polarity.
  The target is never called within-reader uncertainty or clarity.
- Primary feature: per-dimension visual-token dispersion. Claim-token and
  visual-token mean features are secondary controls.
- Five-fold image-grouped outer cross-validation; PCA dimension and ridge
  strength are selected only in an inner grouped fold.
- Every hidden model contains the same spline of `e`, finding effects, and
  population weights as its evidence-only comparator.
- Hidden features are residualized against those nuisance variables inside
  each training fold. PCA is limited to {1, 2, 4, 8, 16} components and at
  most one component per 20 training rows; ridge and PCA dimension use nested
  image-grouped CV. A matched-dimension random projection is a required null.
- A usable residual requires at least 0.05 pooled AUROC gain with an image-
  cluster 95% interval above zero, at least 5% relative Brier improvement with
  its interval above zero, and the same Brier sign in at least six of eight
  findings.
- `Maybe` redundancy and hidden residual sufficiency are separate gates. A
  hidden residual without a causal language effect is only a probe result.
- Erasure requires a preregistered non-final layer to exceed the final layer by
  at least 0.05 AUROC with its interval above zero. Layer-stable residual
  information cannot be described as erasure.

A causal follow-up uses dev-matched pairs with the same finding and signed
evidence but different panel-unanimity states. The patched direction is
projected orthogonally to the signed-polarity direction and activation norm is
restored. It must change definite-versus-undetermined commitment without
changing claim polarity. Random, temperature, norm-only, and unmatched-pair
controls are mandatory.

If no hidden residual passes, the result is only external soft-label
calibration and is not promoted as the paper's mechanism contribution.

## Fail-closed gates

Dev only qualifies a frozen confirmation run. It cannot establish the claim.

- `M0` improves multiclass Brier over the finding prior by at least 5%.
- `M0` is within 2% Brier of flexible evidence-only `M1`.
- panel-constrained `M1` is within 2% Brier of unconstrained `M3`.
- the upper confidence bound on the relative Brier improvement of `M2` over
  `M1` is below 2%, and its NLL improvement is below 0.01 nat/claim.
- at least six of eight findings have the correct evidence direction.
- no reader-state calibration curve is systematically reversed.

On locked confirmation, each model must independently satisfy:

- at least 5% relative Brier improvement over finding prior, lower 95% image-
  cluster bootstrap bound above zero;
- `M0` versus `M1` excess Brier upper bound below 1%;
- `M1` versus `M3` excess Brier upper bound below 1%;
- `M2` over `M1` relative Brier improvement upper bound below 1%;
- `M2` over `M1` NLL improvement upper bound below 0.005 nat/claim;
- standardized conditional direct-Maybe effect entirely inside [-0.1, 0.1]
  log-odds;
- multiclass ECE at most 0.05;
- at least five of eight findings meet the same-direction non-inferiority
  criteria.

Failure of the panel constraint is not repaired post hoc with a beta-binomial
or extra latent variable. Failure of Maybe equivalence means the direct Maybe
coordinate is not redundant. Either result terminates Virtual-Reader
Sufficiency as the paper mechanism.

## Mechanism-to-method boundary

Only a confirmation-qualified model may use Virtual-Reader Commitment
Projection. For an already generated claim, the operation may change only its
certainty wording. It must preserve claim identity, anatomy, attributes,
polarity, and the number of positive claims.

For a positive claim the definite cap is `P3`; for a negative claim it is
`P0`. The projection must preserve `Y-N` exactly while changing the
definite-versus-undetermined coordinate.

This method targets reader-grounded certainty overcommitment. A hedged false
finding remains a fabricated finding and is counted as such.

An optional Evidence-Conserving Claim Exchange is evaluated separately and
only after per-finding support transport succeeds on holdout. It keeps the
positive claim budget `K` fixed and permits a weak draft claim to exchange
with a stronger omitted ontology claim. Failure at matched `K` deletes this
content-correction module without changing the certainty result.

## OE success criteria

- at least 20% relative reduction in false-definite overcommitment;
- at most 1 percentage point loss of definite retention on clear 0/3 or 3/3
  matched claims;
- 100% claim-identity and polarity invariance for certainty projection;
- unchanged positive claim count `K`;
- omission and fabricated-content hallucination non-inferior within 1 point;
- at least 5% relative improvement in reader-distribution Brier;
- physician-reviewed certainty appropriateness improves on matched claims;
- answer length, negative rate, rejection rate, and claim coverage cannot
  explain the gain.

The paper claim remains limited to image-grounded clinical-claim
overcommitment unless the separately gated fixed-`K` exchange also improves
fabrication and omission.
