# C73 — Joint disease geometry does not yield a legal generation primitive

## Decision

**NO-GO.** The VLM's own seven-finding visual field shows a suggestive but
non-confirmed joint signal on Huatuo, and every proposed way of making that
signal change generation falls into an excluded, already occupied operation:
selection/reranking, an energy or logit tilt, verification, or hidden-state
editing. No GPU run is authorized.

## Cheapest cache-only test

The existing Huatuo artifact contains, for every image, 576 projected visual
tokens scored along all seven frozen finding directions. We compared:

```text
base  = finding fixed effects + final answer margin + target-finding patch mean
joint = base + all seven finding patch means
```

The logistic model was fitted on 840 development claims and applied unchanged
to 266 previously opened confirmation claims.

| quantity | result |
|---|---:|
| base macro AUROC | 0.7309 |
| joint macro AUROC | 0.7614 |
| delta | +0.0305 |
| image/stratum bootstrap 95% CI | [-0.0055, +0.0669] |
| NLL improvement CI | [+0.0006, +0.0516] |
| Brier improvement CI | [-0.0002, +0.0213] |

The predeclared gate required delta at least .02 and the AUROC, NLL, and Brier
lower confidence bounds all above zero. It fails. This is also only one model
and is based on supervised finding directions, so it cannot establish a
general phenomenon.

Artifact and executable:

- `anchor/corrected_sgta/screen_vlm_joint_disease_geometry_v1.py`
- `corrected_runs/daylong_idea_search_v1/vlm_joint_disease_geometry_huatuo_v1.json`

## Why DPP, submodular, and graph methods are not a new operator

Let `Y` be candidate clinical claim sets and let `p_theta(y|x)` be the native
VLM distribution.

1. A DPP or submodular objective returns
   `argmax_{S subset Y} quality(S)+diversity(S)`. This is candidate selection;
   if claim count is fixed it is reranking, and otherwise it also deletes
   claims. Diversity is not evidence and offers no truth guarantee.
2. A graph or hypergraph consistency energy produces
   `q(y|x) proportional to p_theta(y|x) exp(-lambda E(y))`. At each decoding
   step this is exactly an additive logit/sequence-energy correction. Calling
   it an energy-conserving operator does not change the computation.
3. Message passing on visual or claim states is feature fusion/editing. If the
   messages are allowed to alter the answer, it is steering; if they merely
   accept/reject the native answer, it is a verifier.
4. If claim identities, polarities, and count are all fixed, a graph operator
   can only change ordering or wording. It cannot correct a false-positive or
   false-negative clinical claim.

Thus there is a simple exhaustion result under the current constraints: a
joint-claim energy can affect clinical content only by changing candidate
probabilities, selecting candidates, or editing the causal state. Those are
precisely logit fusion, reranking/routing, or steering. There is no fourth
generation interface created by DPP terminology.

## Collision boundary

Knowledge/disease-graph report generation is already a crowded family,
including DDGIP (Findings NAACL 2025), graph-conditioned radiology generation,
and claim-decomposition verification such as Pelican (EMNLP 2024). The local
candidate would be weaker because it has neither an independent evidence
source nor a confirmed cross-model joint signal.

The only retained fact is modest: off-claim visual coordinates may improve
calibration/NLL for a weaker model. It does not currently define a direct,
novel hallucination mitigation method.
