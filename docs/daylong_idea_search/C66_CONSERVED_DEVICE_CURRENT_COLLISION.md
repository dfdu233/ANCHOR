# C66 — Conserved device current: theorem/collision audit

**Decision: NO-GO as an ICLR mitigation primitive; no GPU authorization.**

The clinical subproblem is real: thin tubes and catheters are often hallucinated in
generated chest-X-ray reports.  The proposed physical analogy was to interpret a device as
an oriented one-dimensional current rather than as a bag of bright patches.  The audit below
asks the stricter question: does conservation create an output-changing operation that is
mathematically different from a segmentation mask, a shortest-path specialist, or a scalar
verifier?

## 1. Three possible operators and their exact reductions

Let an image lattice be a directed graph `G=(V,E)`, let `s,t` be an admissible entry and
terminal region, and let a specialist output nonnegative edge evidence `c_e(x)`.

### 1.1 Maximum flow does not certify one coherent path

The tempting score is

\[
F(c)=\max_{f\ge 0}\{F:Bf=F(e_s-e_t),\ 0\le f_e\le c_e\}.
\]

By max-flow/min-cut, `F(c)` is the minimum *sum* of capacities across a cut.  It is not the
quality of the best single path.

**Fatal counterexample.**  Take two parallel two-edge paths from `s` to `t`, with capacity
`1/2` on every edge.  Then `F(c)=1`, although every individual path has bottleneck only
`1/2`.  Hence two weak alternatives can add into a full certificate.  The claimed property
"one broken section bounds the whole witness" is false on any graph with parallel routes.
It is true only on a pre-specified chain, which already assumes the device path.

Replacing max-flow by

\[
W(c)=\max_{P:s\leadsto t}\min_{e\in P}c_e
\]

does enforce a single bottleneck path, but this is exactly the classical *widest path*
problem.

### 1.2 Unit min-cost current is ordinary shortest-path extraction

Consider the linear current program

\[
\min_{f\ge0}\sum_e d_e f_e,
\qquad Bf=e_s-e_t.
\]

The node-edge incidence matrix is totally unimodular.  Consequently an optimal extreme
point is the incidence vector of an `s`-to-`t` path (cycles can be removed when costs are
nonnegative), and the objective is exactly the shortest-path cost.  Capacity constraints
give the standard min-cost-flow problem; entropy regularization merely softens it into a
mixture of paths.  Medical minimal-path extraction for vessels, contours and other thin
structures predates modern VLMs, and curvature-aware shortest paths were already studied at
ICCV 2013.

Thus `max-flow` fails the desired single-path claim, while the repair is a mature
centerline/segmentation operator.

### 1.3 Flow attention is exactly a soft mask at the VLM interface

Suppose a decoder consumes visual values through

\[
o=\sum_e \alpha_e V_e.
\]

If a flow solver returns normalized weights `q_e`, "path attention" sets `alpha=q`.  An
ordinary additive attention mask can produce the identical weights.  If native logits are
`l_e`, choose

\[
b_e=\log q_e-l_e+C;
\qquad \operatorname{softmax}(l+b)=q.
\]

The two decoders then have token-exactly the same visual aggregate.  Conservation changes
how the mask was computed, not the intervention channel seen by the frozen VLM.  If the
flow is reduced to a scalar existence score instead, using it changes output only through a
standard energy, veto, reranking, or post-edit channel.

This gives an exact matched control: **the same final flow weights applied as an ordinary
soft mask must be identical**, not merely similar.  Therefore a gain cannot establish a new
decoder primitive; novelty can only lie in the upstream device extractor.

## 2. A 1-current boundary is not a semantic certificate

In geometric-measure language, a device curve may be represented by a one-current `T` with

\[
\partial T=\delta_t-\delta_s.
\]

This boundary condition expresses continuity, not clinical identity.

- Every curve from `s` to `t` has the same boundary, including a wrong catheter, a wire, or
  a high-contrast artifact.
- For every cycle `C`, `partial C=0`; hence `T+C` has exactly the same boundary as `T`.
- Minimum mass removes positive-cost cycles but selects the cheapest geometric curve, not
  necessarily the clinically correct device instance.

The missing information is device identity and admissible anatomy.  If a specialist already
supplies reliable instance identity, endpoints, and anatomical landmarks, it has solved the
clinically decisive part; the current solver only interpolates a path.

## 3. Collision audit

| Nearby work | What it already occupies | Consequence for C66 |
|---|---|---|
| Minimal/shortest paths for medical image analysis | Globally optimal extraction of thin elongated structures from image costs | The single-path repair is not new |
| Bottom-up catheter instance segmentation (Boccardi et al.) | Separates intersecting long thin catheter instances | Connectivity/identity is already a specialist vision task |
| Topology-aware catheter segmentation and misplacement prediction (Hwang et al., MIDL 2024) | Explicit topology loss, path highlighting, and position classification | "Use topology for device evidence" is directly occupied |
| Generalized Attention Flow (ACL 2025) | Maximum flow over Transformer information for attribution | Max-flow plus Transformer attention is already occupied |
| Flow-Attentional GNNs (TMLR 2025) | Kirchhoff-conserving attention | Conservation law inside attention is already occupied |
| FactCheXcker (CVPR 2025) | Specialist query/code/update correction for ETT claims and measurements | A specialist-derived scalar/edit baseline is mandatory and strong |

The untested conjunction "catheter flow used to bias a medical VLM" is narrower than these
components, but conjunction alone does not create a new mathematical operation.

## 4. Local feasibility and falsifiable residual claim

The repository contains a lexical screen over 1,388 MIMIC-CXR generated/reference report
pairs with frequent device phrase mismatches.  These are not clinical hallucination labels:
the reference can omit a stable device and the generated phrase can use a synonym.  No local
RANZCR images, polyline labels, or open centerline-capacity checkpoint were found.

One mechanism claim remains falsifiable without pretending it is a method:

> at matched local edge evidence and endpoint quality, false device claims have lower
> single-path bottleneck evidence than true device claims.

This would require expert/polyline ground truth and comparison against the final margin,
largest connected component, scalar existence, and shortest-path score.  Even a positive
result would establish a device-specific detector/analysis, not the requested universal
training-free mitigation primitive.

## 5. Verdict

C66 fails before GPU for three independent reasons:

1. max-flow admits additive weak-path superposition, contradicting its motivating claim;
2. enforcing one path reduces to classical widest/shortest-path extraction;
3. feeding the resulting current to a frozen VLM is exactly a soft mask, scalar verifier, or
   post-editor.

The idea may be useful engineering for catheter-specific verification, but it does not meet
the active goal's requirements of a general, elegant, novel hallucination-reduction
operation.  The recurring device errors should be retained as a valuable subproblem, while
the max-flow/current formulation is closed.

## Primary references

- Strandmark et al., *Shortest Paths with Curvature and Torsion*, ICCV 2013:
  https://openaccess.thecvf.com/content_iccv_2013/html/Strandmark_Shortest_Paths_with_2013_ICCV_paper.html
- Boccardi et al., *Bottom-Up Instance Segmentation of Catheters for Chest X-Rays*:
  https://arxiv.org/abs/2312.03368
- Hwang et al., *Semi-supervised Learning with Contrastive and Topology Losses for Catheter
  Segmentation and Misplacement Prediction*, MIDL 2024:
  https://proceedings.mlr.press/v227/hwang24a.html
- Azarkhalili and Libbrecht, *Generalized Attention Flow*, ACL 2025:
  https://aclanthology.org/2025.acl-long.980/
- Plettenberg et al., *Flow-Attentional Graph Neural Networks*, TMLR 2025:
  https://arxiv.org/abs/2506.06127
- Heiman et al., *FactCheXcker*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html
