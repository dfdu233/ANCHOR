# C72 — VLM-native path objects: information gate and action degeneration

Date: 2026-08-13  
Scope: cached, CPU-only secondary fatal audit. No GPU or baseline process was
used. The three candidate mathematical objects are exchangeable conformal/test
martingales, persistent/discrete-topological summaries, and optimal transport.

## Verdict

**NO-GO.** None yields a new direct-generation primitive when restricted to a
frozen VLM's native observables:

* a martingale can affect generation only by stopping, filtering, or reweighting
  token probabilities;
* a topological functional can affect generation only after becoming a score
  that selects, masks, or guides tokens;
* optimal transport can affect generation only by moving probability mass,
  attention mass, or token representations, which is respectively assignment/
  reranking, attention bias, or feature mixing.

Moreover, their cheapest common premise fails empirically. On image-disjoint
VinDr development/confirmation claims in two medical VLMs, simple native
four-layer trajectory objects add at most `+0.0038` macro AUROC beyond the final
claim margin, with confidence intervals crossing zero. The frozen `+0.02`
two-model information gate fails before any generation experiment.

## 1. The only allowed information source

Let a frozen VLM expose, for claim `c`, a sequence of native margins

\[
m_{1:L}(x,c)=(m_1,\ldots,m_L),
\]

with no external expert, retrieved case, alternate image, verifier, or learned
adapter. A proposed object `F(m_{1:L})` must first contain held-out case
information not already present in `m_L`. Otherwise using it in generation can
only re-express the final decision or shift its operating point.

The available cache records four evenly spaced decoder layers for Huatuo and
Hulu. We evaluated:

* the full four-margin path;
* early maximum and minimum;
* total variation `sum |m_l-m_{l-1}|`;
* range and maximum drawdown from an earlier layer to the final layer;
* final increment, path mean, and sign-change count;
* linear logistic and nonlinear histogram-gradient-boosting readouts.

Finding identity was controlled in every readout. Development contained 280
clear `0/3` or `3/3` claims per model and confirmation contained 840 claims on
746 images. Labels had already been opened for earlier endpoints, so this is a
secondary fatal audit, not blind confirmation.

## 2. CPU fatal result

| Native object | Huatuo macro AUROC | delta over final | 95% CI | Hulu macro AUROC | delta over final | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| final margin | 0.7667 | -- | -- | 0.8606 | -- | -- |
| full path, linear | 0.7662 | -0.0005 | [-0.0057, 0.0045] | 0.8612 | +0.0006 | [-0.0031, 0.0041] |
| path shape, linear | 0.7704 | +0.0038 | [-0.0026, 0.0100] | 0.8581 | -0.0025 | [-0.0069, 0.0021] |
| full path, nonlinear | 0.7355 | -0.0312 | [-0.0546, -0.0082] | 0.8277 | -0.0330 | [-0.0509, -0.0137] |
| path shape, nonlinear | 0.7448 | -0.0218 | [-0.0448, 0.0011] | 0.8322 | -0.0284 | [-0.0482, -0.0100] |

The result agrees with the earlier blind admission audit: Huatuo's selected
non-final visual readout was `-0.1091` macro AUROC below the matched final
comparator; Hulu's advantage was only `+0.0005`. A fancier scalar functional of
the same coarse path has no empirical authorization.

This does not prove that every conceivable nonlinear functional is useless. It
does prove that the proposed families lack the required minimal positive signal
and should not consume a generation run.

## 3. Exchangeable conformal martingale

Given sequential conformity p-values `p_t`, a standard betting process is

\[
M_t=\prod_{j\le t} f_j(p_j),\qquad
\mathbb E_0[M_t\mid\mathcal F_{t-1}]\le M_{t-1}.
\]

Ville's inequality then controls the probability that `sup_t M_t` crosses a
threshold under a valid null. This is a statement about sequential evidence or
type-I risk, not a token-generation operator.

To change a report, the method must use `M_t` to:

1. stop or abstain;
2. delete/withhold a claim;
3. gate or reweight next-token probabilities;
4. acquire another observation.

These are selective generation, verifier/filtering, guidance, or active
acquisition. The first two are explicitly outside the target; the third is an
old decoding channel; the fourth is no longer a same-observation one-pass
method. Valid p-values additionally require exchangeability or a calibrated
null, so a martingale cannot be constructed from arbitrary correlated layer
margins merely by multiplying ranks.

The collision is direct. [ConfLVLM (EMNLP
2025)](https://aclanthology.org/2025.emnlp-main.576/) treats generated details as
hypotheses and conformally filters unreliable claims. [CEBC (ACL
2026)](https://aclanthology.org/2026.acl-long.2142/) conformally calibrates
external visual evidence and then minimally revises/suppresses unsupported
mentions. [Look Again Before You Abstain
(2026)](https://arxiv.org/abs/2606.16667) combines conformal reliability with
budgeted evidence acquisition. A test martingale changes the time index, not
the intervention class.

**Decision: formula collision plus failed native-path information gate.**

## 4. Persistent or discrete-Morse trajectory

A layer trajectory may be viewed as a one-dimensional filtration. Local extrema,
merge-tree persistence, total variation, and maximum drawdown are stable or
interpretable summaries of shape. They do not identify truth: an confidently
wrong trajectory can be perfectly persistent, while a correct late correction
can have high variation.

Once a persistence number is computed, it can change generation only by:

* selecting a layer/token (dynamic layer choice or reranking);
* masking/downweighting an unstable component;
* adding a topology-derived logit or hidden-state bias;
* refusing a low-persistence claim.

Those are layer selection, masking, guidance/steering, or abstention. Calling
the score a Morse critical value does not alter the action.

There are two independent local failures. First, the path-shape gate above is
near zero in both models. Second, persistent H0 summaries of the Huatuo
claim-conditioned patch field previously added only `+0.0091` AUROC over the
strong final/mean/max/top-5%/scan base, CI `[-0.0048,+0.0241]`; it failed the
same `+0.02` gate. Persistence therefore has neither a native trajectory nor a
spatial-field positive premise here.

Nearby work already includes PHG-Net (WACV 2024) for persistent-homology-guided
medical classification and topology-based hallucination analysis such as TOHA;
using topology as a scalar input to a frozen generator would be an ordinary
feature/guidance attachment.

**Decision: failed cached premise and action degeneration.**

## 5. Optimal transport

OT needs two measures and a ground cost. With only one frozen VLM path, any
second measure must be manufactured from another layer, token subset, or
reference distribution. The possible generated actions are exhaustive:

1. transport vocabulary probability mass: linear assignment/reranking or
   auxiliary guidance;
2. transport attention mass: Sinkhorn/IPFP scaling, exactly a patchwise
   attention-logit bias;
3. transport visual/token embeddings: feature mixing or token merging;
4. transport a full output sequence: minimum-Bayes-risk selection/reranking.

For example, projecting native attention `A` to a target column marginal `c`
via

\[
B^*=\arg\min_{B\ge0}\mathrm{KL}(B\|A),\quad
B\mathbf1=r,\ B^\top\mathbf1=c
\]

gives `B*=diag(u)A diag(v)`. If `A_tp` is a softmax of logits `ell_tp`, this is
exactly `ell_tp -> ell_tp+log v_p`, a soft attention mask. Likewise, transporting
the sorted vocabulary mass to an evidence ranking makes the greedy winner the
evidence ranker's top token, i.e. hard reranking. These identities were already
established in local C59/C63 audits.

Without an external truth-bearing measure, the transport cost only says which
native states are geometrically near, not which clinical claim is correct. With
an external measure it returns to expert fusion or model stitching, already
closed by C68/C71.

**Decision: exact algebraic reduction; no distinct L0 experiment exists.**

## 6. Unified degeneration proposition

Let `S=F(O_theta)` be any deterministic statistic of a frozen model's native
observables and let a same-task method return a different autoregressive law.
At the first time its conditional token law differs from the frozen law, one of
the following must have occurred:

* support was reduced or generation stopped;
* token probabilities were changed;
* hidden/visual states or attention were changed;
* multiple continuations were compared and one selected.

Thus `F` can be a new measurement, but its action is respectively selective
generation, guidance, representation/attention editing, or search/reranking.
Conformal, topology, and OT do not form a fifth direct-generation channel. A
new paper would need a new causal law showing why one permitted channel becomes
clinically specific, not a new name for `F`.

## 7. Reproducibility

* script: `anchor/corrected_sgta/audit_native_path_objects_v1.py`
* result: `corrected_runs/daylong_idea_search_v1/native_path_objects_v1/result.json`
* Huatuo/Hulu development and confirmation metadata under
  `corrected_runs/vindr_v2/hidden_*`
* related local audits: C59, C63, C67, C70, C71

No baseline state or GPU process was modified.
