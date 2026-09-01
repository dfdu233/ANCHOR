# C66 — Path-Conserving Device Evidence (PCDE)

**Status: one concrete, untested seed; conceptually cleaner than label fusion, but not yet GPU-authorized.**

## Natural error substrate

A conservative lexical screen over 1,388 paired MIMIC-CXR native greedy reports
(694 Huatuo + 694 Hulu) found recurrent device mismatches:

| device family | paired phrase mismatch |
|---|---:|
| central line | 161 / 1,388 = 11.6% |
| endotracheal tube | 142 / 1,388 = 10.2% |
| pacemaker | 99 / 1,388 = 7.1% |
| enteric tube | 87 / 1,388 = 6.3% |
| chest tube | 60 / 1,388 = 4.3% |

This is not yet clinical ground truth: radiology references can omit stable devices and the
regex can miss synonyms.  It is nevertheless a real recurring report subproblem, not a
synthetic Yes/No construction.  The artifact is
`corrected_runs/daylong_idea_search_v1/natural_report_error_subproblems_v1/result.json`.

The raw reports also expose visually plausible fabrications.  For example, Huatuo describes
an ETT, central line, and pacemaker in case
`bc25fa99-0d3766cc-7704edb7-5c7a4a63-dc65480a`, while the paired report describes none;
55/694 Huatuo cases mention more device families than their references while mentioning at
least two device families.

## The computation error

Standard VLM attention treats image patches as an unordered additive evidence reservoir.
For thin medical devices, this is the wrong primitive: several disconnected radiopaque
fragments can be pooled into a plausible "line", although no physical device path exists.

The specialist should therefore not output a device label, posterior, ROI, or text.  It outputs
only a nonnegative edge-capacity field `c_e(x)` on an image lattice/graph.  A device witness is
an admissible source-to-sink flow:

\[
0\le f_e\le c_e(x),\qquad Bf=b_{s,t},
\]

where `B` is the node-edge incidence matrix and `b_{s,t}` injects one unit at an anatomical
entry region and removes it at an admissible terminal region.  The clinical evidence value is

\[
F(x)=\max_{f\ge0}\left\{\mathbf1^Tf_{out(s)}:
Bf=F(e_s-e_t),\;f\le c(x)\right\}.
\]

By max-flow/min-cut,

\[
F(x)=\min_{C\in\mathcal C(s,t)}\sum_{e\in C}c_e(x).
\]

Thus one weak broken section bounds the whole witness; disconnected bright fragments cannot
add into a positive device certificate.  This is a nontrivial structural property absent from
softmax pooling.

The intended decoder replacement is **path attention**: when a device word/phrase is being
formed, visual aggregation is allowed only through the normalized witness flow, not through
an arbitrary patch mask.  At internal nodes, incoming and outgoing attention mass must match
(Kirchhoff conservation).  Small model = edge capacities; large VLM = language and all
non-device findings.

## What is and is not novel

Not novel individually:

- catheter/tube segmentation and centerline extraction are mature;
- max-flow/min-cut and differentiable shortest paths are standard;
- Flow-Attentional GNNs (2025) enforce Kirchhoff conservation for physical graph flows;
- Generalized Attention Flow (ACL 2025) uses max flow for Transformer attribution;
- FactCheXcker (CVPR 2025) uses specialist tools to edit ETT existence/measurement.

Potentially new joint claim:

> thin-structure hallucination is caused by **fragment superposition** in additive visual
> attention; replacing patch aggregation with a path-conserving flow makes physical
> connectivity a generation invariant, without the specialist voting on the label.

This is not a generic medical hallucination solution.  It targets fabricated/misbinding
claims about one-dimensional devices (and, if the mechanism transfers, vessels, ducts,
fracture lines, and instruments).

## Fatal risks

1. If the specialist capacity map is itself a full device segmentation, the path solver adds
   little beyond standard specialist detection and the method is tool-based verification in
   disguise.
2. Device intersections and occlusions make strict connectivity false; the flow needs learned
   bridge capacities, which may hallucinate the very path it is supposed to certify.
3. A frozen decoder has no native path-attention hook.  Biasing its visual attention is still
   an intervention channel; an implementation must show that the conservation law, not simply
   the specialist mask, causes the gain.
4. FactCheXcker already improves ETT presence precision by querying an expert and removing
   unsupported text.  PCDE must beat its `exists()` baseline and a segmentation-mask attention
   baseline at matched device recall/claim count.

## Minimal experiment, only after a usable open capacity model is confirmed

1. Use 32 frontal MIMIC cases stratified as 8 reference ETT-present, 8 ETT-absent with VLM
   ETT mention, 8 other-device controls, 8 no-device controls.
2. Run one open ETT/line heatmap model once per image; no extra VLM pass.
3. Compare the *same* capacity map in three operators:
   - global/ROI soft mask;
   - largest connected component / scalar exists score;
   - path-conserving flow attention.
4. Mandatory canaries: method-off 32/32 token-exact; shuffled capacity map; spatially broken
   true path; disconnected fragments with matched total capacity; matched attention mass.
5. Go only if path flow removes at least 20% of phrase-confirmed device FP, loses <=1 TP in 32,
   and strictly beats both scalar existence and ordinary mask using the identical specialist
   output.

No GPU run is currently authorized: the readily available FactCheXcker checkpoint outputs ETT
and carina coordinates, not an edge-capacity/centerline map; substituting it would collapse PCDE
to the already-published `exists/find/update` pipeline.

## References

- Heiman et al., FactCheXcker, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.pdf
- Plettenberg et al., Flow-Attentional Graph Neural Networks, 2025:
  https://arxiv.org/abs/2506.06127
- Modarressi et al., Generalized Attention Flow, ACL 2025:
  https://aclanthology.org/2025.acl-long.980/
- Boccardi et al., Bottom-up Instance Segmentation of Catheters for Chest X-rays,
  MIDL 2024: https://arxiv.org/abs/2312.03368
- Hwang et al., topology-aware catheter segmentation/misplacement, MIDL 2024:
  https://proceedings.mlr.press/v227/hwang24a.html
