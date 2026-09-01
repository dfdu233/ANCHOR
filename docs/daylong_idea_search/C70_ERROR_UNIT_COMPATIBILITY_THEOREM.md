# C70 — Frozen-VLM mitigation under error-unit replacement

Date: 2026-08-13  
Scope: formula and existing-artifact audit only; no GPU and no baseline changes.

## Verdict

Under the full C01--C69 constraints there is no sixth, error-unit-only
mitigation channel.  Replacing a token/claim by a set, measure, composite event,
attribution, or morphology has the following trilemma:

1. an invertible replacement is only a change of coordinates;
2. a non-invertible replacement coarsens the task and cannot be uniformly
   no-harm for the original clinical loss;
3. a set-valued replacement is abstention unless a selector maps it back to a
   singleton, in which case the selector is an ordinary decision intervention.

The only nontrivial loophole is not a new representation but an **externally
certified semantic law**.  A typed group-action compiler can correct a
reference-frame attribute in one forward pass while preserving finding content.
This is the already identified C58 exception, not a general fabricated-finding
method.  If deterministic output compilation is also excluded, the feasible
set is empty.

## 1. Setup

Let the frozen VLM induce a conditional sequence law

\[
P_\theta(Y\mid O), \qquad O=(X,Q,M),
\]

where `X` is the image, `Q` the question, and `M` any allowed external metadata.
A same-task direct mitigation must still return one output in the original
clinical output space `Y`, under the same loss and content budget.  Its complete
effect is another conditional law `Q_A(Y|O)`.

C65 already establishes an implementation fact: at the earliest point where
`Q_A` differs from `P_theta`, the wrapper has changed one of the input, an
internal representation/computation, a next-token distribution/support, the
search/trajectory selection, or the final output.  If none changes, the chain
rule gives `Q_A=P_theta`.

The question here is whether changing the *semantic error unit* can avoid this
fact.

## 2. Error-unit reparameterization trilemma

Let

\[
\phi:\mathcal Y\to\mathcal E
\]

map an ordinary report/claim to a proposed error unit, and let `T` operate on
`E`.  To return an ordinary report, the method also needs a realization kernel
`L: E -> Delta(Y)`.  The induced action is

\[
K=L\circ T\circ\phi.
\]

### Case A: `phi` is bijective

With the natural inverse realization, the method on the original space is

\[
\widetilde T=\phi^{-1}\circ T\circ\phi.
\]

Thus every output change, risk, fixed point, and constraint is exactly
conjugate to an ordinary operator on `Y`.  A signed measure, provenance vector,
or event code that losslessly represents the report is useful notation but
does not create a new intervention or correctness source.

### Case B: `phi` is non-injective

Choose `y1 != y2` with `phi(y1)=phi(y2)`.  Any rule whose only new input is that
unit must take the same action distribution in the two worlds.  For an original
loss that distinguishes `y1` and `y2` (including exact claim error), a single
action cannot be correct in both.  Under 0--1 loss it would have to equal both
`y1` and `y2`.

Therefore no non-injective error-unit-only rule can be both sometimes
corrective and uniformly no-harm on the original task.  Apparent guaranteed
improvement requires changing the target to the quotient/coarser task, adding
new truth-bearing information, or imposing a verified structural assumption.

### Case C: `phi` or `L` is set-valued

If the method returns a non-singleton set, it has changed point prediction into
set prediction/selective prediction.  The universal set obtains trivial
coverage, so hallucination reduction is inseparable from set size/abstention.
If a selector `sigma` returns one report, then

\[
K=\sigma\circ T\circ\phi
\]

is again an ordinary decision rule, implemented through one of the C65
channels.  Set-valued language does not remove the selection problem.

This is consistent with modern generative prediction-set work: its guarantee
is set coverage after calibration, not a lower singleton hallucination risk.

## 3. Candidate replacements do not escape the theorem

### Composite likelihood

A composite rule of the form

\[
S(y)=\sum_j w_j\log p_j(\phi_j(y)\mid O)
\]

changes generation only by ranking/sampling with `S`, hence it is an output
energy or search rule.  If all factors are deterministic views of the same
frozen state, they add no observation.  A correctness claim additionally needs
a valid factorization or independent evidence; otherwise composite likelihood
is a misspecified score, not a hallucination certificate.

### Signed measure

The Jordan decomposition `mu=mu+ - mu-` is either a lossless coordinate lift
(Case A) or a lossy aggregation (Case B).  Integrating, thresholding, or
transporting the measure to choose words is respectively score fusion,
calibration/constraint, or assignment/reranking.

### Causal attribution

An attribution `a=A(P_theta,X,Q)` is an observation about model dependence, not
about clinical truth.  If it is only displayed, the output law is unchanged.
If it changes the output, the action is a mask/input intervention, steering,
guidance, search, or editing.  Dependence on a pixel does not certify the claim
supported by that pixel.

### Morphological or topological event

Connected components, persistent components, paths, and shape events are
deterministic feature maps.  Treating them only as error units invokes the
trilemma.  Using them to alter pixels is an input transform; using them to
accept or suppress a claim is a verifier/veto; using their score in decoding is
guidance.  C66--C68 additionally show that path, persistence, and sparse-event
versions lack a truth-identifying local signal in the available cache.

## 4. The one legal loophole: certified semantic automorphism

The trilemma assumes that the new unit only re-expresses existing outputs.  It
does not forbid exact external side information.  Suppose:

1. the output is typed as `y=(a,s)`, where `a` is a frame-invariant clinical
   atom and `s` is a covariant attribute;
2. a known transformation `g` is supplied by the acquisition/viewer pipeline;
3. a group representation `rho_g` gives the exact action of `g` on the output
   semantics.

Then the deterministic compiler

\[
D_g(X)=\rho_{g^{-1}}D_{\rm native}(X)
\]

is a legal, one-forward operator.  If `pi_a rho_g=pi_a`, it obeys exact content
conservation

\[
\pi_aD_g(X)=\pi_aD_{\rm native}(X).
\]

If native-frame prediction is equivariant,

\[
D_{\rm native}(hX)=\rho_hD_{\rm native}(X),
\]

then compilation is invariant to re-rendering:

\[
D_{hg}(hX)=D_g(X).
\]

The useful ingredient is not the group formula; it is the externally certified
law connecting display and clinical semantics.  This avoids calibration,
abstention, reranking, veto, layer fusion, and attention modification.  It is
low latency and changes only the typed attribute.

The existing C58 cache is the concrete instance: Huatuo patient-frame
laterality was `1/13`, while display-frame prediction followed by a known
left/right compilation was `12/13`; non-frame text was preserved exactly.
However, VinDr orientation metadata is incomplete, the natural-OE and second-
model gates are pending, and the candidate registry correctly rejects this as
a general fabricated-finding solution.  It is at most a method for certified
reference-frame hallucinations.

## 5. Constraint-compatibility conclusion

| Proposed replacement | Same-task direct action | Strict verdict |
|---|---|---|
| Set-valued output | set prediction or singleton selector | abstention, or ordinary decision rule |
| Composite likelihood | sequence/event energy | guidance/search |
| Signed evidence measure | coordinate lift or coarsening | no new information/action |
| Causal attribution | measurement followed by an action | action returns to an exhausted channel |
| Morphological event | feature, input transform, or gate | representation only, visual transform, or veto |
| Certified group-typed attribute | exact semantic push-forward | **legal only for the certified subproblem** |

Consequently:

* If the objective remains general fabricated-positive mitigation and all five
  action channels remain excluded, **no legal operator exists**.
* If the objective allows one exact reference-frame subproblem and deterministic
  semantic compilation, C58 is the only current loophole, but it is not yet an
  ICLR-level general method.
* A genuinely new general method must relax at least one constraint: learn a
  small shared interface, admit a new independent observation, or permit a
  known action channel with a new empirically validated causal law.  Renaming
  the error unit cannot do this by itself.

## Primary references

- C65 channel exhaustion and C58 frame-covariant decoding (local reports).
- Shahrokhi et al., *Conformal Prediction Sets for Deep Generative Models via
  Reduction to Conformal Regression*, UAI 2025:
  <https://proceedings.mlr.press/v286/shahrokhi25a.html>.
- Equi-Tuning, AAAI 2023:
  <https://ojs.aaai.org/index.php/AAAI/article/view/25832>.
- DICOM patient-orientation modules:
  <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.html>.
- *Laterality: A Potential Pitfall for Large Language Models in Radiology*,
  Radiology 2024: <https://pubs.rsna.org/doi/10.1148/radiol.241421>.
