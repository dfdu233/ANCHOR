# Query-conditioned causal topology audit

Date: 2026-08-13  
Decision: **scientifically meaningful, method-level NO-GO**

## Frozen question

For a decoder-only VLM with prompt order `[V, Q, A]`, the native causal mask
forces every visual-token state to be independent of the later question:

`V_l = f_l(V_{l-1})`.

The tested `visual_read` topology changes only visual-token query rows so that
they can read all observed question columns:

`V_l = f_l(V_{l-1}, Q_{l-1})`.

Pixels, input embeddings, token order, position IDs, sequence length and
Yes/No/Maybe verbalizers remain identical. `full_prefix`, where all observed
prompt tokens attend bidirectionally, is an upper control rather than a method.

## Huatuo frozen n=16 result

| Topology | BAcc | FP | FN | Mean Yes-No margin |
|---|---:|---:|---:|---:|
| Native causal | 62.50% | 5 | 1 | +0.727 |
| Visual-read only | 68.75% | 4 | 1 | +0.563 |
| Full prefix | 56.25% | 1 | 6 | -0.410 |

`visual_read - native` BAcc was `+6.25pp`, but its stratified bootstrap 95% CI
was `[0, +18.75pp]`; the preregistered strict-positive lower-bound gate failed.
Exactly one prediction changed: a unanimous no-finding image queried for lung
opacity moved from margin `+1.125` to `-0.250`. No new error was introduced in
this panel. The margin change was more negative in vote-0 than vote-3 examples
(`-0.250` versus `-0.078`), but the difference CI `[-0.313, +0.688]` was
uninformative.

Full-prefix attention caused a large global negative shift and five additional
false negatives. It cannot be interpreted as better visual grounding.

Artifacts:

- `corrected_runs/daylong_idea_search_v1/prompt_closure_huatuo_n16/analysis.json`
- `corrected_runs/daylong_idea_search_v1/prompt_closure_huatuo_n16/raw.jsonl`
- `anchor/corrected_sgta/run_huatuo_prompt_closure_probe_v1.py`

## Collision audit

The exact intervention family is already occupied:

1. Sony AI, *Seeing is Understanding: Unlocking Causal Attention into
   Modality-Mutual Attention for Multimodal LLMs* (arXiv:2503.02597; the
   official repository currently labels the venue ICML 2026), makes the same
   image-token-to-later-question visibility edit without new parameters:
   <https://arxiv.org/abs/2503.02597>.
2. Pei et al., *Rethinking Causal Mask Attention for Vision-Language
   Inference*, ICLR 2026, explicitly evaluate future-aware masks in which
   visual-token queries preview later textual tokens and develop pooled
   future-aware attention: <https://arxiv.org/abs/2505.18605>.
3. Abhinandan et al., *Ask Twice, Look Twice* (2026) use `[Q,V,Q]` question
   echoing so the first copy steers perception and the second supports answer
   readout: <https://arxiv.org/abs/2607.15565>.
4. InViC (2026) learns question-conditioned visual cue tokens for Med-VQA:
   <https://arxiv.org/abs/2603.16372>.
5. AIF, CVPR 2026, modulates text-to-visual causal-mask connectivity during
   inference: <https://cxliu0.github.io/AIF/>.

The tested edge is also not guaranteed to add clinical evidence. It writes a
question-derived value into visual slots, so question priors can travel through
the new `Q -> V -> A` path as easily as image evidence. It changes an inductive
bias, not the information present in the image, and has no monotone error
correction property.

## Boundary

The result supports the architectural diagnosis that one-way prompt topology
can matter, but it does not support a new mitigation method. Do not enlarge,
tune layers, interpolate masks, or rebrand question echoing as a medical
contribution. The implementation may be retained only as an existing-method
control.
