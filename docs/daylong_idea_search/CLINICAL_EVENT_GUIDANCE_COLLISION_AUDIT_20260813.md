# Clinical-event guidance: collision and cache audit (2026-08-13)

## Verdict

**Strict NO-GO as an ICLR-level method.**  Giving a complete clinical event one
expert weight, rather than repeating that weight on every subtoken, is the
correct way to define a semantic sequence energy.  It can be a useful repair to
CCD-like implementations, but its exact update is already posterior
regularization / a Doob `h`-transform (FUDGE), and its finite-state
implementation is standard WFST phrase biasing plus weight pushing.  No update
law remains that is materially different from constrained decoding or
sequence-level reranking.

This audit did not use a GPU and did not touch the running baseline queue.

## 1. Exact mathematical reduction

Let `p(y|x)` be the frozen VLM distribution over complete token sequences and
let `phi_j(y)` be a binary indicator that complete clinical event `j` occurs in
the decoded report.  A clinical event contains finding, polarity, uncertainty,
location and attributes; it is not an individual token.  If the small expert
supplies desired event marginals `mu_j`, the proposed minimum-change update is

```text
min_q KL(q || p)  subject to  E_q[phi_j] = mu_j.
```

The Lagrangian has the unique exponential-family solution

```text
q_lambda(y|x) = p(y|x) exp(lambda^T phi(y)) / Z(lambda).
```

This is exactly an I-projection / posterior-regularization update.  It is not a
new clinical decoding law.

For online autoregressive decoding, define the future potential

```text
h(u) = E_p[exp(lambda^T phi(Y)) | Y has prefix u].
```

Then the exact next-token law is

```text
q(t | u, x) = p(t | u, x) h(ut) / h(u).
```

This is the Doob `h`-transform form of future-discriminator guidance.  FUDGE
implements the same Bayesian prefix/future decomposition; GeDi is the nearby
case in which a smaller generative discriminator provides the class likelihood
used to alter next-token probabilities.

If clinical events are recognized by a DFA/WFST, placing `lambda_j` only on the
accepting transition does settle the evidence once.  Moving some or all of that
weight to earlier subword arcs while preserving every complete path score is
exactly WFST **weight pushing**.  “Pay at event completion” and “distribute the
same total weight across the event path” are two representations of the same
sequence energy.

The problematic CCD-style rule instead pays the full event weight at every
matching subtoken:

```text
S_token(y) = log p(y|x) + sum_j lambda_j L_j(y) phi_j(y),
```

where `L_j` is the number of biased subtokens.  The event-correct rule is

```text
S_event(y) = log p(y|x) + sum_j lambda_j phi_j(y).
```

The former changes the objective by multiplying evidence by lexical length; it
is not an approximation to the latter unless the per-token weights are
normalized to sum to one.  Removing this multiplicity is an implementation
correction, not a new inference principle.

## 2. Exact collision matrix

| Work | Same object | Same update/intervention | Collision verdict |
|---|---|---|---|
| Ganchev et al., *Posterior Regularization for Structured Latent Variable Models*, JMLR 2010 | posterior moments of structured events | minimum-KL projection gives an exponential tilt | **Direct mathematical collision** |
| Yang & Klein, *FUDGE*, NAACL 2021 | complete-sequence attribute evaluated from prefixes | Bayesian future potential adjusts each next-token distribution | **Direct online-law collision** |
| Krause et al., *GeDi*, Findings EMNLP 2021 | small model supplies desired/undesired sequence evidence | Bayes-rule next-token guidance | Strong collision for small-expert guidance |
| Mohri, Pereira & Riley, *Weighted Finite-State Transducers in Speech Recognition*, 2002; Mohri & Riley, weight pushing, 2001 | phrase/event automata and path weights | apply or redistribute one phrase weight while preserving path score | **Direct implementation collision** |
| Jinnai et al., *Model-Based MBR Decoding*, ICML 2024 | complete hypotheses and semantic utility | model-probability-aware candidate reranking | Direct collision if implemented after generation |
| Zhang et al., *CCD*, 2025/ACL 2026 | radiology-expert probabilities and pathology terms | converts expert probabilities to token-level log-odds biases | Closest medical baseline; event settlement only repairs multiplicity |
| *AEGCD*, ACL ARR 2026 | multiple experts and semantic token classes | adaptive token-level expert bias and routing | Remaining delta is only event-path accounting |
| Li et al., *Visual Evidence Prompting*, ACL 2025 | detector/scene-graph objects, attributes and relations are complete semantic evidence units | expert output is symbolized and inserted as a prompt | Covers the event-level expert-prompt branch; not an I-projection |
| Guo & Terzopoulos 2024 | a cheap medical weak learner emits pathology judgments to a medical VLM | expert output inserted as explicit textual guidance | Covers the medical prompt-level small/large collaboration branch |

Primary sources:

- <https://jmlr.csail.mit.edu/papers/volume11/ganchev10a/ganchev10a.pdf>
- <https://arxiv.org/abs/2104.05218>
- <https://aclanthology.org/2021.findings-emnlp.424/>
- <https://www.sciencedirect.com/science/article/pii/S0885230801901846>
- <https://proceedings.mlr.press/v235/jinnai24a.html>
- <https://arxiv.org/abs/2509.23379>
- <https://openreview.net/forum?id=gsAgvQ2T8T>
- <https://aclanthology.org/2025.acl-long.205/>
- <https://arxiv.org/abs/2407.21368>

## 3. Why fixed claim count and length do not create a new method

With fixed output length and a fixed number `K` of claims:

- if candidate reports contain the same clinical events, `S_event` adds the
  same constant and cannot change their ranking; `S_token` can change the
  ranking only because surface forms have different token lengths, which is a
  spurious tokenizer effect;
- if candidates contain different event identities, `S_event` reranks by the
  sum of expert event scores, which is ordinary expert-energy reranking / MBR;
- if the event constraint is enforced during generation, the exact rule is the
  FUDGE/Doob transform above, and a finite-state approximation is standard
  constrained decoding.

Thus fixed `K` and fixed length are important anti-confounding controls, but
they do not produce a mathematically new update law.

## 4. Local cache feasibility audit

Inspected artifacts:

- `corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl`:
  694 images, one greedy plus one sampled candidate per image; only candidate
  text, token count and sequence-level uncertainty are retained.
- `corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl`:
  694 images, one greedy plus one sampled candidate per image; same limitation.
- Native Huatuo/Hulu MIMIC beam caches under
  `corrected_runs/paper_baselines_v1/full_matrix_v1/report_scores/native/`:
  694 outputs each; generated token IDs and mean token NLL are present, but
  per-step vocabulary logits, alternative beam states, beam indices and
  expert-conditioned traces are absent.
- `corrected_runs/xrv_visual_increment_v1/xrv_logits.npz`: genuine XRV expert
  outputs exist for 1,003 **VinDr** images, not for the MIMIC report IDs in the
  candidate caches.

The available greedy/sample candidate pairs contain exact equal-token-count and
equal-regex-claim-count cases, but only 8 nontrivial pairs for Hulu.  The 118
such pairs in the other cache are dominated by identical normal templates and
zero matched ontology claims.  More importantly, final token IDs plus one mean
NLL cannot reconstruct how tokenwise guidance would have changed the beam at
each step.  Offline selection between finished reports would test MBR/reranking,
not the proposed online event-settlement law.

Therefore a real L0 cannot be obtained from current report/beam caches without
a fresh instrumented generation pass.  Running such a pass is not justified
after the direct collision and would compete with the active baseline GPU.

## 5. Existing empirical evidence against the motivating mechanism

The repository already tested the necessary surface-fragmentation phenomenon
on 694 MIMIC-CXR reports for each of Huatuo and Hulu across 13 non-`No Finding`
CheXbert concepts (candidate registry C42):

- Huatuo: concept token length vs FPR Spearman `-0.014 (p=.964)`; vs FNR
  `.166 (p=.587)`.
- Hulu: concept token length vs FPR `.576 (p=.039)`; vs FNR `-.377 (p=.204)`.

The direction is not shared across models and is not the predicted “short
events are hallucinated, long events are omitted” pattern.  So even the
engineering motivation for event settlement lacks a cross-model local signal.

## 6. Claim ceiling

The defensible conclusion is narrow:

> Expert evidence should be defined on clinical events and conserved across
> their surface realization; repeatedly adding the same evidence to every
> subtoken changes the intended sequence objective.

That is a useful implementation principle or ablation for CCD-like baselines.
It is **not** an ICLR-level mitigation method, because the I-projection, online
future-potential update, finite-state realization and candidate-level version
are all established methods, and the local token-fragmentation mechanism did
not reproduce across models.
