# Treble–CECD closest-work collision protocol

**Freeze:** 2026-08-02; source/venue revalidated 2026-08-03; dual-source
envelope frozen 2026-08-03. **Scope:** source audit and implementation
contract; no GPU result. **Decision:** exact paper-native Treble reproduction
is blocked. A non-paper-native dual-semantics common-protocol envelope is now
the only admissible fallback comparison, but is not yet authorized to run.

This protocol exists to stop a particularly damaging category error: CECD's
render marginal, prompt marginal, or two-way centered interaction must never be
renamed a Treble Natural Direct Effect (NDE). Treble is a **global,
representation-level, method intervention**. CECD Stage 1 is a **per-claim
behavioral factorial diagnostic**. They can be compared only through paired
method outcomes after a faithful Treble adapter exists.

## 1. Verified sources and audit boundary

The authoritative paper is Li Li, Jiashu Qu, Linxin Song, Yuxiao Zhou, Yuehan
Qin, Tiankai Yang, and Yue Zhao, *Treble Counterfactual VLMs: A Causal Approach
to Hallucination*, Findings of EMNLP 2025, pages 18423--18434, DOI
10.18653/v1/2025.findings-emnlp.1000
([ACL Anthology record](https://aclanthology.org/2025.findings-emnlp.1000/);
[paper](https://aclanthology.org/2025.findings-emnlp.1000.pdf)). This supersedes
the earlier audit's arXiv-only metadata; the corresponding preprint remains
arXiv:2503.06169v2 ([record](https://arxiv.org/abs/2503.06169)). The official repository is
[`TREE985/Treble-Counterfactual-VLMs`](https://github.com/TREE985/Treble-Counterfactual-VLMs),
audited at commit
[`f52197e`](https://github.com/TREE985/Treble-Counterfactual-VLMs/tree/f52197e48bd34a54508afbb49da25a26cb74be3f).
Remote `HEAD` and `main` still resolve to that same commit on 2026-08-03, with
no tags. The repository page still reports one commit and exposes no release,
environment lockfile, requirements file, or explicit root license. Its README
instructs the user to populate LAVIS, LLaVA, POPE, and COCO manually
([README](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/README.md)).

Two nearby medical baselines were also checked. SPCD is currently retrievable
only as a January 2026 ResearchGate manuscript record describing original vs
proximity-constrained counterfactual logit contrast; no official arXiv,
proceedings page, or source repository was retrieved
([record](https://www.researchgate.net/publication/399488093_Semantics_Preserving_Contrastive_Decoding_for_Hallucination_Robust_Medical_Vision-Language_Models)).
LENS is an anonymous submission available through OpenReview and describes
attribution-guided semantics-preserving views plus stability-aware decoding,
but no official implementation was retrieved
([OpenReview PDF](https://openreview.net/pdf?id=oh3c2ieVab)). Consequently,
neither SPCD nor LENS is paper-native runnable here; an implementation from
their prose would be a local surrogate and is inadmissible as an "official"
baseline.

## 2. What the Treble paper states

Let `F(t,v)` denote fused multimodal knowledge and `Y(t,v,F)` the answer. The
paper labels the following output contrasts as NDEs:

\[
\begin{aligned}
E_V &= Y(t,v,F(t,v))-Y(t,v_*,F(t,v_*)),\\
E_T &= Y(t,v,F(t,v))-Y(t_*,v,F(t_*,v)),\\
E_{VT} &=Y(t,v_*,F(t,v_*))-Y(t,v_{null},F(t,v_{null})).
\end{aligned}
\]

Operationally, its representation estimators are:

\[
\begin{aligned}
D^V_{i,k}(I)&=V^I_{i,k}-\frac1m\sum_{j=1}^m
V^{M_j(I)}_{i,k},\\
D^T_i(C)&=T_i^C-T_i^{C^h},\\
D^{VT}_i(I)&=H_i(I_{black},C)-H_i(I_{null},C).
\end{aligned}
\]

Across `N` demonstrations, the first PCA direction is used as a global
direction. The paper then states an additive intervention at every layer and
token:

\[
V'_{i,k}=V_{i,k}+a d^V_{i,k},\qquad
T'_i=T_i+b d^{VT}_i+c d^T_i,
\]

with `N=50`, PCA rank 1, and `a=b=c=0.9`. These definitions are reported as
the authors' operational estimands. They should not be upgraded to a standard
mediation-identification claim: the displayed output contrasts change
`F(t,v)` together with the treated input rather than holding the mediator at a
cross-world reference value.

## 3. What the released code actually computes

The public implementation is materially different from the prose.

### 3.1 Counterfactual inputs and direction order

| Component | Paper | Released source before PCA | Verdict |
|---|---|---|---|
| Vision | original minus average of `m` randomly masked views | average of `m` Gaussian diffusion-noise-step-500 tensors minus original; sampled `mask_index` is unused | different perturbation and opposite order |
| Text | factual-caption last-token state minus hallucinated-caption state | factual-caption state minus hallucinated-caption state | same order in proceedings and source |
| Cross-modal | black-image state minus no-image state | no-image state minus mean Gaussian-noise-step-200 image state | different inputs and opposite order |

The relevant released functions are
[`visual_shift`, `text_shift`, and `cross_modal_shift`](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/experiments/Treble_Counterfactual_utils/Representation_Shift.py#L292-L345),
the Gaussian sampler
([lines 518–580](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/experiments/Treble_Counterfactual_utils/Representation_Shift.py#L518-L580)),
and the nominal blank sampler
([lines 648–674](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/experiments/Treble_Counterfactual_utils/Representation_Shift.py#L648-L674)).

The direction called "PCA" in the code is not simply `PC1`. For rank `r`, it
constructs `mean + sum_{q=1}^r component_q`, then reshapes that vector to all
layers (and all visual tokens). This source behavior must be reproduced exactly
if the code, rather than the prose, becomes the resolved reference.

### 3.2 Test-time intervention

The source wraps the MLP of every vision layer and then wraps the MLP of every
decoder layer twice, first for text and then for cross-modal shift. For an MLP
output vector `x`, directions `d_j`, and weights `lambda_j`, the exact source
arithmetic is

\[
x'=\lVert x\rVert_2\operatorname{normalize}\left(
  \operatorname{normalize}(x)+0.1\frac1J\sum_j
  \lambda_j\operatorname{normalize}(d_j)\right),
\]

followed by FP16 casting. Thus the nominal coefficient `0.9` enters through a
hard-coded inner step `0.1`, making the single-direction angular step `0.09`
before renormalization. The vision, text, and cross-modal modules are additive
residual shifts after each MLP, not a per-example NDE scalar
([source](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/experiments/Treble_Counterfactual_utils/Test_Time_Intervention.py#L9-L27)).

### 3.3 Released-entry-point blockers

The audited LLaVA entry point cannot run unchanged on a case-sensitive Linux
filesystem:

1. it imports lowercase `representation_shift`, while the tracked file is
   `Representation_Shift.py`;
2. the parser defines `sample_num`, while the sampler reads `num_demos`;
3. the runner reads `rankk`, while the parser defines only `rank`;
4. the cross-modal sampler passes an extra `mask_index` argument to
   `add_gaussian_noise`, whose signature does not accept it;
5. there is no exact model/checkpoint revision, dependency version set, or
   executable command in the README;
6. no license grants reuse of the repository code or its 100 generated
   demonstration records.

The runner and its parser are visible in the
[`Test_Time_Intervention_llava.py`](https://github.com/TREE985/Treble-Counterfactual-VLMs/blob/f52197e48bd34a54508afbb49da25a26cb74be3f/experiments/test/Test_Time_Intervention_llava.py)
source. These are not ordinary plumbing defects we may silently repair because
the apparent repairs require choosing between contradictory scientific
semantics.

## 4. Compute ledger

For the released defaults `N=50` demonstrations and `m=50` perturbation trials,
direction estimation requires:

| Resource | Calls |
|---|---:|
| Gaussian-step-500 vision-encoder forwards | 2,500 |
| Original vision-encoder forwards | 50 |
| Factual/hallucinated full multimodal forwards | 100 |
| Gaussian-step-200 full multimodal forwards | 2,500 |
| No-image language-only forwards | 50 |
| Total image-bearing calibration forwards | 5,150 |
| Target generation after fitting | 1 per example |

This `5,150` ledger is specific to the released-source variant. The
proceedings-faithful variant uses 2,500 masked vision forwards, 50 original
vision forwards, 100 factual/hallucinated multimodal forwards, 50 black-image
multimodal forwards and 50 no-image language forwards: 2,700 image-bearing
calibration forwards, not 5,150. The dual-semantics envelope freezes these as
two different heterogeneous ledgers; copying the released Gaussian-200 cost
onto the paper's one-black-image-per-demo estimator is prohibited.

Vision-only, multimodal, and language-only calls must remain separate in every
runtime table; converting them to one invented FLOP number is forbidden. The
paper-native method is cheap only after amortizing its global calibration.
For `B` target examples its source-path ledger is `5,150 + B` heterogeneous
image-bearing calls, whereas the proposed `2 x 2` CECD method is `4B` target
calls and the current diagnostic factorial is `15B` science calls. These are
not matched at the 160-claim pilot scale.

Matched comparisons therefore have two axes:

- **Calibration-matched:** Treble full versus identical counterfactual
  collection/PCA/hooks with norm-matched random directions and paired-label
  randomization. All use the same `N`, `m`, source images, and hook operations.
- **Online-matched:** CECD `2 x 2` correction versus the four-pass full-orbit
  ensemble, render-only and prompt-only four-pass padded controls, and a
  random interaction projection. Treble remains a one-pass amortized method
  and is reported on the latency/quality Pareto frontier rather than falsely
  called compute-matched.
- **Component controls:** Treble `V`, `T`, and `VT` individually, all pairwise
  combinations, and full `V+T+VT`; fitting cost is shared and target-pass count
  is identical. This is necessary because the paper reports no modality
  ablation.

## 5. Huatuo and Hulu adapter plan

### 5.1 Shared frozen inputs

If the paper/code conflict and license are resolved, the exact closest-work
baseline first uses the official 100 demonstration identities, captions, and
hallucinated captions, selecting `N=50` with seed 42 and acquiring the
corresponding COCO train2014 images. This preserves the Treble source
distribution. A VinDr-derived direction would be labelled
`Treble-medical-adapted`, never exact Treble, because VinDr has reader labels
but no paired factual/hallucinated captions.

Calibration is dev-only; model weights, processors, demo ordering, random
noise tensors, all counterfactual inputs, PCA inputs, directions, and hook
paths are hash-bound. Evaluation is on a locked image-disjoint VinDr test set.
One-token Yes/No/Maybe is retained only as the shared CE instrument; exact
Treble also receives the fixed-`K` OE/report evaluation if CECD reaches that
stage.

### 5.2 HuatuoGPT-Vision-7B

Huatuo has a fixed CLIP-ViT-L/14@336 visual grid and a 28-layer Qwen2 decoder,
so the released tokenwise direction shape can be represented.

1. Reuse the exact image and conversation construction already validated by
   `HuatuoScorer` in `run_cecd_factorial_v1.py`.
2. Collect every CLIP hidden state from
   `bot.model.get_vision_tower().vision_tower.vision_model.encoder.layers` for
   the original and each resolved visual counterfactual.
3. Collect the last text-position state at the embedding output and after all
   28 decoder blocks for the resolved factual/hallucinated and
   degraded/no-image pairs.
4. Fit source-exact tokenwise vision directions and last-token decoder
   directions on the frozen demonstrations; discard the embedding direction
   exactly as the released runner does.
5. Wrap each CLIP layer MLP with the vision shift and each
   `bot.model.model.layers[l].mlp` sequentially with text then cross-modal
   shifts. Assert 24 vision directions and 28 decoder directions at runtime;
   record nesting order.
6. Conformance requires zero-shift token-logit identity, norm preservation per
   token, exact hook counts, deterministic replay, and one-example equality to
   the independently implemented pure shift equation before any batch run.

This is implementation-ready only after the semantic and license blockers are
resolved. Until then it is deliberately not coded as a runnable model adapter.

### 5.3 Hulu-Med-4B

Hulu has a 27-layer vision encoder, a 36-layer Qwen3 decoder, and adaptive
image grids. Its processor preserves aspect ratio and therefore emits a
variable number of visual tokens. Treble's released visual estimator stacks
the same token index across demonstrations and its layer intervention expects
one direction per token. That contract is undefined for Hulu's variable grids.

The decoder portions have clear hook paths:

- vision: `runtime.model.get_vision_encoder().encoder.layers[l].mlp`;
- text: `runtime.model.model.layers[l].mlp`.

But a full exact adapter remains blocked unless the authors specify a
variable-token transport. Resizing every image to a fixed square, interpolating
directions, or pooling tokens changes the method and must be labelled a Hulu
surrogate. A text-only or cross-modal-only port is also not the full Treble
baseline. Therefore the two-model exact collision gate cannot claim Hulu until
this architecture boundary is resolved; CECD must not lower the gate to a
surrogate merely to obtain two models.

## 6. External method-outcome contract

Exact Treble produces globally fitted directions and intervened model outputs;
it does **not** produce a paper-defined per-claim scalar. The CECD analyzer must
ignore legacy `treble_nde_score` fields. Any scalar projection onto a direction
is diagnostic and must be named
`crossmodal_direct_effect_scalar_surrogate`.

A future independent validator consumes
`cecd-treble-method-collision-v1` with these mandatory fields:

```text
source_repo_commit
reproduction_fidelity
model_fingerprint
calibration_split
evaluation_split
record_keys_sha256
compute_ledger
paired_method_metrics
paired_cluster_bootstrap
collision_verdict
```

`reproduction_fidelity` must be `paper_and_code_semantics_resolved`, calibration
must be `dev`, evaluation must be `locked_test`, and the comparison must use the
same record-key hash. The only verdicts are `cecd_survives` and
`direct_collision`.

The collision thresholds are frozen before either method output is opened:

1. both Huatuo and Hulu must first pass the existing CECD Stage-1 behavioral
   gate; a Treble surrogate cannot substitute for a blocked Hulu adapter;
2. on CE, CECD must reduce reader-grounded overcommitment by at least 20%
   relative to the unmitigated model, retain clear-case accuracy within 1
   percentage point, and have a paired image-cluster-bootstrap advantage over
   full Treble whose 95% interval is strictly above zero;
3. on fixed-`K` OE/report generation, CECD must reduce positive-content
   hallucination by at least 20% relative to the unmitigated model, not increase
   omission, and again beat full Treble with a paired 95% interval strictly
   above zero at identical claim coverage;
4. reader-distribution Brier must improve at least 5% relative to the
   unmitigated model; CECD may not lose that improvement relative to Treble;
5. all tests use the same locked records, decoding settings, claim count,
   answer-length accounting, and no-refusal rule. Hyperparameters remain the
   paper defaults (`N=50`, rank 1, `a=b=c=0.9`) unless a deviation is labelled
   adapted and excluded from the exact gate.

Failure of any absolute quality/coverage gate, a CECD–Treble interval touching
zero, one-model-only fidelity, or an unresolved compute/fidelity audit yields
`direct_collision` or `blocked`; it never authorizes hidden-state work.
Hidden-state work remains unauthorized until two models meet all rules through
an independent external validator.

### 6.1 Repairing the circular gate without inventing an “exact” Treble

The preceding v1 wording contains a sequencing problem: it requires CECD to
beat Treble before authorizing the causal comparison that would produce that
result. It also lets a future artifact self-assert
`paper_and_code_semantics_resolved`, although the proceedings and released
source are mutually inconsistent. The corrected state machine is:

```text
two-model CECD Stage 1 passes
        -> freeze outcome-blind method-comparison preflight
        -> independently bind preflight to the reconstructed Stage-1 files
        -> run one locked controlled comparison
        -> validate outcomes and decide collision
```

No output-dependent statistic is required at preflight. The runtime binder
permits CECD hidden intervention only inside that exact comparison and keeps
general hidden-state/GPU and paper authorization false.

Because there is no unique exact Treble to choose, the common-protocol
comparison must bracket both primary-source semantics:

| Variant | Vision | Text | Cross-modal | Direction/intervention |
|---|---|---|---|---|
| `treble_proceedings` | original − random-mask mean | factual − hallucinated | black-image − no-image | PC1; paper additive `a=b=c=0.9` |
| `treble_released` | Gaussian-500 mean − original | factual − hallucinated | no-image − Gaussian-200 mean | released mean+PC construction; norm-preserving `0.1×0.9`, FP16 |

Both are independently implemented from the public equations and audited
arithmetic. Neither is called paper-native or exact, no official source/demo
content is redistributed, and both retain separate heterogeneous compute
ledgers. Huatuo and Hulu each bind checkpoint, processor, template, generation,
hook, and visual-token-transport hashes; the Hulu transport is therefore an
explicit architecture adaptation rather than a concealed exact port.

The preflight schema is `cecd-treble-dual-semantics-preflight-v1`. It freezes
the two source variants, ten-method closure, full-orbit/render/prompt/random/
sign/main-effect controls, all CE/OE no-exchange metrics and thresholds,
10,000 cluster bootstraps, dev/locked-test manifests and online/calibration
compute. The runtime binder
`authorize_cecd_dual_semantics_preflight_v1.py` additionally reconstructs the
existing two-model Stage-1 result, verifies all hashes, and rejects a preflight
if any method output already exists.

The post-run schema is `cecd-treble-dual-semantics-envelope-v1`. CECD survives
only if both model families independently meet the absolute 20% CE and OE
reductions, ≤1 pp clear-case loss, no omission/refusal increase, fixed claim
count, matched coverage and length, ≥5% reader-Brier improvement, and paired
CI advantage over **both** Treble variants and full-orbit averaging. The
validator recomputes the verdict; it rejects a self-declared positive verdict
when any CI touches zero or any exchange control fails. Even a valid surviving
envelope sets `paper_claim_authorized=false`; physician-grounded downstream
evidence remains required.

This envelope does not repair the exact paper-native baseline. Exact Treble
remains `blocked, not reproduced`; the bracket instead prevents the scientific
conclusion from depending on an arbitrary choice between contradictory sources.

## 7. Collision decision

The current exact-paper-native verdict is **blocked, not reproduced**. The
dual-semantics common-protocol path is frozen but has no Stage-1 authorization
or model output. This is stronger and more useful than a guessed port:

- generic vision/text/cross-modal counterfactual steering is already occupied
  by Treble and cannot be claimed by CECD;
- CECD retains novelty only if its clinically admitted composition defect adds
  error information beyond marginals and later beats faithful Treble as a
  method;
- official Treble cannot currently be used to satisfy that gate because its
  public paper and code disagree, the entry point is broken, licensing is
  absent, and Hulu lacks the fixed visual-token axis its estimator assumes;
- SPCD and LENS remain non-admissible paper-native baselines until official
  runnable implementations and licenses are retrieved.

The executable audit primitives and tests are in
`anchor/corrected_sgta/treble_collision_contract.py` and
`tests/test_treble_collision_contract.py`. The outcome-blind runtime binder is
`anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py`. Together
they encode the two competing delta definitions, the exact unambiguous
norm-preserving shift, separate resource ledgers, frozen no-exchange controls,
and fail-closed pre/post-run validators. They do not fabricate a scientific run.

The 2026-08-03 proceedings/repository revalidation corrects the earlier
arXiv-only sign audit: proceedings Eq. (7) and the source both use factual minus
hallucinated text. Proceedings Eq. (5), however, uses original minus masked
vision while the source uses Gaussian-degraded minus original, and Eq. (8)
still differs in both counterfactual inputs and order. This correction changes
one blocker but not the execution verdict: publication in Findings confirms
Treble as mandatory closest work, while the unchanged one-commit source leaves
the remaining paper/code conflicts, broken entry point, absent reuse license,
and Hulu variable-token ambiguity unresolved. Therefore no local
implementation can be reported as `paper_native` or satisfy the two-model
collision gate without an author clarification or a new licensed official
release.
