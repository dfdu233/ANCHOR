# Specificity Ratchet mechanism protocol v1

> **Superseded before scientific scoring:** the isolated parent/child target
> runtime in this document fails F6 because automatically shortened parents can
> be incomplete or malformed standalone answers. The current contract is the
> full-visible-answer replay amendment in
> `docs/SPECIFICITY_RATCHET_FROZEN_RESEARCH_CONTRACT_20260802.md`. This file is
> retained as preregistration history and must not authorize a GPU run.

## Claim boundary

The only candidate claim is:

> On a physician-admitted nested clinical edge, a medical VLM can assign more
> commitment to the added child constraint than is warranted by the same
> image, even while the visually supported parent remains available.

This is **causal child-over-parent specificity escalation**. It is not a new
fine-grained hallucination detector, a new clinical ontology, or evidence that
all open-ended medical hallucinations arise autoregressively. The current
127-edge pack contains language-derived proposals only. No experiment may run
until two independent physician sheets and blinded adjudication pass the
fail-closed validator.

Model generations, VQA-RAD short references, cross-model recurrence, LLM
judges, and RadGraph are prohibited as clinical truth. They may not admit an
edge or choose its visual support state.

## Scientific unit and admission

The unit is one adjacent edge `parent -> child`, where the child is judged to
entail the parent plus one added constraint. The compiler admits a scientific
row only when all of the following hold:

1. both physician sheets are complete, use different reviewer IDs, preserve
   every blinded source field byte-for-byte, and use only schema values;
2. both reviewers attest that they are physicians, worked independently, and
   remained blind to private model provenance;
3. every reviewer field is copied exactly into `adjudication.csv` and every
   categorical disagreement has a reason;
4. every final state, rationale, adjudicator ID, and physician adjudicator
   attestation is present;
5. the final edge is admitted and preserves logical scope;
6. the final parent is visually `supported`;
7. the child/source pair is internally coherent.
8. the scored child is a unique exact UTF-8 substring of the frozen OE
   generation. This proves spontaneous occurrence only and never supplies a
   clinical support label.

Eligible rows receive one of three roles:

- `supported_specificity_control`: parent and child are supported on the
  supplied image;
- `causal_escalation_error`: parent is supported, the image-observable child
  constraint is refuted or undetermined;
- `evidence_source_boundary`: parent is supported but the child requires a
  missing view, sequence, history, laboratory, pathology, prior, or nonvisual
  knowledge.

The third role tests the boundary and is never pooled into the primary visual
error endpoint. Edges with uncertain source observability or unsupported
parents are excluded with a recorded reason.

## Frozen paired estimands

For image (I), question (q), admitted parent (P), child (C), and the
exact added-constraint token union (T_\Delta\), teacher forcing uses the same
image and question for both sequences.

At decoder layer (l):

\[
g_l = \operatorname{mean}_{t\in T_\Delta}
      \log p_l(C_t\mid I,q,C_{<t}),\qquad
b_l = \operatorname{mean}_{t\in T_P}
      \log p_l(P_t\mid I,q,P_{<t}).
\]

The paired specificity contrast is (r_l=g_l-b_l). The mechanism estimand is
the error-versus-supported-control difference in the late-minus-early change of
(r_l), with image/case clustering. This is a nested-sequence intervention:
the image and question are fixed and only the admitted parent/child target is
changed. It supports a causal claim only about this child-over-parent contrast,
not about the origin of every free-generation error.

The runtime must tokenize the full target and map offsets back to each frozen
UTF-8 character span. It scores the union of those spans. A tokenizer boundary
failure excludes the row before analysis; it must never fall back to the whole
sentence. Repeated additions such as two occurrences of `right` remain two
frozen spans.

### Mandatory nuisance controls

- exact scored-token count under each model tokenizer;
- full parent- and child-target token counts;
- mean text-only NLL of the exact parent and constraint tokens as the frozen
  lexical-frequency proxy;
- edge type, modality, anatomy, whether the prompt requested the increment,
  and case cluster.

The residualizer and any layer choice are fit on `dev` only. The compiler
creates deterministic, image-disjoint grouped dev/test splits while balancing
mechanism role, edge type, modality, and anatomy. The test split is evaluated
once.

Image-null and same-support image-swap contrasts are secondary sensitivity
analyses only. Arbitrary image nulls can be out of distribution and therefore
cannot admit the mechanism. The primary comparison is the same-image parent
control.

## Causal controls and falsification

The screen requires all of the following on the held-out split:

1. error edges show a positive error-versus-supported difference in the
   late-minus-early child-over-parent contrast, with image-cluster bootstrap
   95% CI excluding zero;
2. the result remains after token-count and text-only-NLL controls;
3. it is not reproduced by random equal-norm directions, sequence-length
   permutations, or shuffled parent-child pairings;
4. a parent-directed activation patch or projection selectively reduces the
   unsupported child constraint without materially reducing parent support;
5. the direction and intervention replicate in at least two model families and
   at least three admitted edge types.

If only final-layer confidence separates errors, this is a detector result, not
a specificity-ratchet mechanism. If image-null supplies the only signal, the
mechanism fails. If supported children fall as much as unsupported children,
the intervention is nonspecific. If the physician-admitted pack yields too few
image-disjoint error/control units for stable inference, the experiment remains
underpowered and no threshold is relaxed.

## Fixed-K nearest-ancestor mitigation

Mitigation is allowed only after the mechanism passes. For an exactly mapped
unsupported child, replace the child one-for-one with its physician-admitted
nearest supported parent. Clinical claim count (K) is unchanged:

\[
K_{after}=K_{before},\qquad \Delta K=0.
\]

No deletion, refusal, added hedge, negative default, or ontology insertion
receives credit. Unmatched claims are untouched. Evaluation reports mapped
coverage, positive-content precision, parent retention, clinical usefulness,
answer length, and refusal rate. The comparison set includes an equal-count
random ancestor replacement and a lexical-frequency-matched replacement. Any
gain that disappears at matched coverage or comes from fewer claims fails.

## Closest-collision audit (retrieved 2026-08-02)

| Work | What it covers | Why it is close | Remaining delta |
|---|---|---|---|
| [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) | Conformally calibrated external-detector constraints and minimal editing of unsupported object mentions | Evidence-bounded editing while preserving output quality | No physician-admitted clinical parent/child edge and no paired teacher-forced child-over-parent causal estimand |
| [CounterVHD, arXiv:2606.28520](https://arxiv.org/abs/2606.28520) | Medical entity extraction, factual/counterfactual grounding, and entity-level uncertainty | Medical fine-grained support and counterfactual evidence | Detects entity hallucination through an external grounding verifier; does not test whether adding an admitted clinical constraint causally escalates commitment relative to its same-image parent |
| [ZINA, CVPR 2026](https://arxiv.org/abs/2506.13130) | Human-annotated hallucinated spans, six error types, and span editing | Fine-grained localization and correction | The unit is an erroneous span, not an independently physician-admitted nested clinical edge; no parent-controlled layerwise mechanism |
| [FINER, arXiv:2603.17662](https://arxiv.org/abs/2603.17662) | Fine-grained negative queries and DPO data for multi-object, attribute, relation, and “what” errors | Shows that present coarse content can induce fine-grained false commitments | Query benchmark/training contribution; no clinical reader adjudication or causal child-over-parent teacher-forcing contrast |

No mechanism-equivalent work was retrieved under searches for nested claims,
specificity escalation, parent-child teacher forcing, fine-grained clinical
hallucination editing, and counterfactual grounding. This is not proof of
novelty. The defensible delta is only the physician-grounded causal
child-over-parent escalation estimand.

## Execution

The current pack is intentionally blank, so both commands must refuse:

```bash
python anchor/corrected_sgta/validate_specificity_ratchet_adjudication_v1.py \
  --pack corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2

python anchor/corrected_sgta/compile_specificity_ratchet_mechanism_manifest_v1.py \
  --pack corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2
```

After independent review, create `physician_attestations.json` in the pack:

```json
{
  "protocol_id": "specificity-ratchet-physician-pack-v2",
  "reviewers": [
    {
      "reviewer_id": "<sheet-1 ID>",
      "role": "physician",
      "independent_review": true,
      "blinded_to_private_provenance": true,
      "completed_at_utc": "<ISO-8601>"
    },
    {
      "reviewer_id": "<sheet-2 ID>",
      "role": "physician",
      "independent_review": true,
      "blinded_to_private_provenance": true,
      "completed_at_utc": "<ISO-8601>"
    }
  ],
  "adjudicator": {
    "adjudicator_id": "<adjudication.csv ID>",
    "role": "physician",
    "blinded_to_private_provenance": true,
    "completed_at_utc": "<ISO-8601>"
  }
}
```

Only a successful compile may produce
`corrected_runs/specificity_ratchet/mechanism_manifest_v1/samples.jsonl` and
its inseparable `metadata.json` contract. No GPU job is authorized before that
artifact exists.
