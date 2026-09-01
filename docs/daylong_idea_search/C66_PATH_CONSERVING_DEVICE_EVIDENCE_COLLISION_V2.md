# C66 / PCDE collision and feasibility audit

**Decision: NO-GO as the main idea of a day-long ICLR search.**  Keep only a bounded
oracle falsification test if RANZCR data access becomes available.  No GPU experiment was
started, and no baseline process was touched.

## 1. What the seed was trying to solve

The natural target is useful and clinically concrete: a frozen medical VLM may mention an
endotracheal tube, central line, or other thin device that is not supported by the image.  C66
proposed to replace additive patch evidence with a source-to-target flow on an image graph.
Given nonnegative edge scores `c_e`, it used

\[
F_{\mathrm{flow}}(x)=
\max_f F\quad\text{s.t.}\quad 0\le f_e\le c_e,\qquad
Bf=F(e_s-e_t).
\]

Here, `B` is the graph incidence matrix, `s` and `t` are anatomical start/end regions, and
`F` is the amount of flow that can pass from `s` to `t`.  The intended intuition was simple:
disconnected bright fragments should not add up to a complete physical device.

The existing natural-report screen does show a real *candidate subproblem*: in 1,388 paired
MIMIC reports, model/reference phrase mismatches were 11.6% for central lines, 10.2% for ETT,
7.1% for pacemakers, 6.3% for enteric tubes, and 4.3% for chest tubes.  These are not clinical
error labels: a reference report may omit a stable device, and the lexical matcher may miss
synonyms.  They establish prevalence for follow-up sampling, not the proposed mechanism.

## 2. A mathematical counterexample to the operator

Maximum flow is the correct primitive for the *total capacity of many simultaneous routes*.
A device-presence claim usually asks whether at least one coherent physical path exists.  Those
are different questions.

Consider a graph with two parallel source-to-target paths.  Every edge on both paths has
capacity `0.4`.  Then

\[
F_{\mathrm{flow}}=0.4+0.4=0.8.
\]

Now consider one coherent path whose edges all have capacity `0.7`; its maximum flow is `0.7`.
C66 therefore ranks two weak parallel artifacts above one stronger coherent line.  With many
weak background bridges, this accumulation becomes more severe.  The max-flow/min-cut theorem
does not prevent it: the capacity of a cut is a **sum** over its crossing edges.

For a single-path witness, the mathematically aligned alternatives are instead:

\[
F_{\mathrm{wide}}=\max_{p:s\leadsto t}\min_{e\in p}c_e
\]

or, when edge scores are calibrated probabilities,

\[
F_{\mathrm{prod}}=\max_{p:s\leadsto t}\prod_{e\in p}c_e
=\exp\!\left[-\min_{p:s\leadsto t}\sum_{e\in p}-\log c_e\right].
\]

The first selects the path with the strongest weakest link (the widest path); the second is a
shortest-path problem after converting probabilities to negative log costs.  Both avoid summing
parallel weak paths, but neither is a new mathematical primitive.  More importantly, changing
max-flow to either one removes the purported flow-conservation contribution and leaves a
standard topology-aware specialist score.

## 3. The proposed decoder has no extra expressivity over a mask

Suppose the path solver produces desired normalized patch weights `q_i`, with `q_i >= 0` and
`sum_i q_i = 1`.  A normal cross-attention head has logits `a_i` and weights
`softmax(a_i + b_i)`.  Choose

\[
b_i=\log q_i-a_i+C,
\]

where `C` is any constant and a zero `q_i` is implemented by a large negative bias.  Then

\[
\operatorname{softmax}(a+b)_i=q_i.
\]

Thus a precomputed "flow mask" at the VLM aggregation point can be reproduced exactly by an
ordinary attention bias.  Any gain must come from how the specialist computes `q`, not from a
new decoding operator.  A genuinely recurrent flow network over the image graph could be a
different model class, but it would require a trained module and directly collide with FlowGNN.

This creates a strict baseline requirement: using the identical specialist output, PCDE must
beat (i) scalar device existence, (ii) largest connected component / widest path, and (iii) a
plain attention mask.  Otherwise it is a more complicated implementation of an existing tool
or mask.

## 4. Formula-level and problem-level collisions

### Direct mathematical collision

- **Flow-Attentional Graph Neural Networks**, TMLR 2025
  ([paper](https://arxiv.org/abs/2506.06127),
  [official code](https://github.com/pasplett/FlowGNN)).  It already identifies arbitrary
  message duplication as the defect of ordinary attention, normalizes over outgoing neighbors,
  proves that the induced absolute flow satisfies Kirchhoff's first law, and gives graph
  expressivity results.  C66's central mathematical primitive and its "evidence cannot be
  duplicated" story are therefore occupied.

- **Generalized Attention Flow**, ACL 2025
  ([paper](https://aclanthology.org/2025.acl-long.980/)).  It already treats Transformer
  attention/gradient tensors as capacities and solves a maximum-flow problem across the
  Transformer graph.  It is an attribution method rather than decoding, so it is not an exact
  task collision, but "Transformer attention + capacity + max-flow" cannot be claimed as new.

### Exact medical-prior collision

- **Semi-supervised Learning with Contrastive and Topology Losses for Catheter Segmentation and
  Misplacement Prediction**, MIDL 2024
  ([paper](https://proceedings.mlr.press/v227/hwang24a.html)).  It explicitly uses the known
  topological properties of catheters as a learning constraint and jointly predicts catheter
  masks and misplacement.  Therefore "a catheter is a physically connected object" is already
  an established prior in this exact subproblem.

- **Bottom-Up Instance Segmentation of Catheters for Chest X-rays**, MIDL 2024
  ([paper](https://arxiv.org/abs/2312.03368)).  It handles thin, crossing catheter instances via
  associative embeddings on 8,877 RANZCR images.  This shows that crossings and multiple
  devices require instance identity, not connectivity alone.

### Exact correction-channel collision

- **FactCheXcker**, CVPR 2025
  ([paper](https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html),
  [official code](https://github.com/rajpurkarlab/FactCheXcker)).  It queries specialist
  `exists`, `find`, and measurement tools and updates generated reports.  For ETT, it already
  reduces false mentions and improves presence/placement precision across 11 report generators.
  Its current report-triggered setting cannot recover reports that omit an ETT, which is a real
  limitation, but it already occupies specialist-assisted ETT hallucination correction.

Together, these papers make the likely reviewer summary unfavorable: "FlowGNN/topological
catheter evidence applied as a FactCheXcker-style specialist mask to a medical VLM."

## 5. Asset audit

| asset | what is actually available | can it instantiate PCDE? |
|---|---|---|
| RANZCR CLiP Kaggle data | 30,083 CXRs, image-level device labels, and polylines for roughly 9k images / 17,999 annotated line instances | Yes for an **oracle annotation** experiment after Kaggle authentication and rule acceptance; data are not present locally and no Kaggle credential/CLI is configured |
| `jarrelscy/cxr-lines-tube-model` | public code and two roughly 51 MB weights; a six-label DenseNet for ETT/CVC/NGT presence and malposition | No; it outputs labels, not a centerline or edge-capacity map |
| FactCheXcker CarinaNet | Apache-2.0 Hugging Face weight; outputs ETT-tip and carina coordinates | No; two points do not supply the path between them |
| FactCheXcker ETT ResNet | inference code references `checkpoints/ett-resnet-30epoch.ckpt`; checkpoint is absent from the GitHub tree, though cached predictions are provided | No; scalar existence only |
| `xinario/catheter_detection` | code plus a public OneDrive link to a 21 MB pretrained pediatric detector; clinical test set is withheld | Possible smoke heatmap only; pediatric/domain-mismatched and not a credible adult RANZCR capacity model |
| `pambros/CNN-2D-X-Ray-Catheter-Detection` | centerline extraction code for X-ray fluoroscopy, with synthetic/example training recipe | No reliable CXR formal baseline; wrong acquisition domain |
| Hwang topology model / MIDL bottom-up model | papers describe useful segmentation architectures | No public clinical checkpoint found |
| RadZero | open zero-shot CXR grounding/segmentation code and weights | It can provide a generic soft mask, but is not validated for device centerlines; using it makes the experiment mask + graph post-processing |

The practical bottleneck is therefore not a graph solver.  It is obtaining a reliable,
instance-aware, adult-CXR line probability field whose errors do not already decide the final
presence label.

## 6. The untested causal premise

C66 assumes a specific native failure:

> disconnected device-like fragments are additively pooled by the frozen VLM and cause a false
> device claim.

No current result tests that statement.  Phrase mismatches establish neither ground-truth error
nor fragment superposition.  Occlusion also makes the reverse implication unsafe: a true device
may have no visible continuous path.  Learned bridge probabilities can then fabricate exactly
the connection that the method is intended to certify.

The source and sink sets are additionally device-specific.  ETT, CVC, NGT, chest tubes,
pacemaker leads, vessels, and fracture lines have different admissible endpoints and topology.
This weakens the proposed generality before any result is obtained.

## 7. Only defensible residual experiment

This is a **falsification probe**, not the next paper goal.

1. After Kaggle access, sample patient-disjoint RANZCR cases with explicit device labels and
   annotated polylines.
2. From each positive polyline create paired maps/images: intact path, one cut segment,
   disconnected fragments with matched activated mass, spatial shuffle, and weak parallel
   paths.  Keep total mask mass and perturbation magnitude matched.
3. Measure the frozen VLM's device-token margin and final phrase under those counterfactuals.
   First test the phenomenon: do disconnected matched-mass fragments causally raise false
   device probability?
4. On the same edge field compare scalar existence, soft mask, largest connected component,
   widest/max-product path, and maximum flow.
5. Only if the phenomenon replicates in two VLMs should a decoder hook be attempted.

Go requires all of the following on a larger patient-level holdout, with paired bootstrap CIs:

- disconnected matched-mass fragments significantly increase native false device claims;
- structural correction lowers false-positive device claims by at least 20% with no more than
  1 percentage-point device recall loss;
- it strictly outperforms scalar `exists`, connected-component/widest-path, and ordinary mask
  baselines using the identical specialist field;
- method-off decoding is token-exact and the effect is not response shortening.

If the native fragment-superposition phenomenon is absent, a scalar specialist matches the
result, or max-flow loses to the single-path operators, close C66 permanently.

## 8. Final assessment

| criterion | assessment |
|---|---:|
| natural medical subproblem | useful, but current labels are only lexical proxies |
| mathematical fit | poor: max-flow sums parallel weak paths |
| mathematical novelty | low: FlowGNN and Generalized Attention Flow occupy the primitive |
| medical-prior novelty | low: topology-aware catheter work already exists |
| simple deployability | low: no credible open centerline-capacity checkpoint is ready |
| generality | low-to-medium: endpoint/topology semantics are device-specific |
| ICLR oral potential now | **NO-GO** |

The valuable lesson is not to discard physical structure.  It is that a successful structural
hallucination method must begin with a demonstrated **native structural computation error**, and
its decoder operator must have behavior that neither a scalar verifier nor an ordinary attention
mask can reproduce.  C66 currently satisfies neither condition.
