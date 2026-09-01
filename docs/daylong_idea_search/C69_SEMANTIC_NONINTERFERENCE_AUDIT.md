# C69 — Semantic invariants for frozen medical VLMs: formula-level audit

Date: 2026-08-13  
Scope: CPU/source/literature audit only; no baseline process or GPU was touched.

## Executive verdict

The security-language intuition is attractive:

> treat pixels as high-integrity evidence and language priors as lower-integrity
> context; do not allow the latter to overwrite the former.

But four candidate invariants—read-only visual memory, exact noninterference,
taint/provenance propagation, and secure multi-execution—do not yield a new
training-free mitigation operator for the present frozen decoder-only VLMs.

The reasons are structural rather than empirical:

1. future generated tokens already cannot write into an earlier visual prefix;
2. exact independence from the question also removes the information specifying
   which clinical claim to answer;
3. exact provenance through dense softmax attention taints every downstream
   token, while relaxed provenance becomes attribution;
4. a noninterfering merger of trusted and untrusted executions cannot use the
   untrusted execution, while a merger that does use it is fusion, verification,
   or reranking.

The useful conceptual correction is that the failure is **read-path competition**,
not visual-memory corruption.  That problem is real, but its intervention space
is already occupied by visual reinjection, attention routing, latent steering,
and contrastive decoding.

## 1. Read-only visual memory is already native

Let visual-prefix positions precede generated answer positions.  In a causal
decoder, the hidden state at position `i` and layer `l` is

\[
h_i^{(l)}=F_l(h_1^{(l-1)},\ldots,h_i^{(l-1)}).
\]

For a visual position `v` and a future answer position `t>v`, therefore,

\[
\frac{\partial h_v^{(l)}}{\partial e_t}=0
\quad\text{for every layer }l.
\]

With KV caching, visual keys and values are computed during prefill and later
steps append new KV entries rather than overwrite the prefix.  Consequently, an
operator that merely sets

\[
\widetilde K_V^{(l)}=K_V^{(l)},\qquad
\widetilde V_V^{(l)}=V_V^{(l)}
\]

is token-exactly identical to native decoding.  This has already been checked
numerically in the local C68 audit.

If the operator instead prevents visual updates *across depth*, it changes the
architecture.  [ShortV (ICCV 2025)](https://arxiv.org/abs/2504.00502) already
freezes visual tokens in selected ineffective layers, while [ViCA
(2026)](https://arxiv.org/abs/2602.07574) explicitly treats visual tokens as
read-only cross-attention memory.  If it reinjects the saved memory later,
[MemVR (ICML 2025)](https://arxiv.org/abs/2410.03577) already does so via FFN
key-value memory.

**Verdict:** pure write protection is an identity; non-identity versions are
directly occupied architecture or reinjection methods.

## 2. Exact prompt-to-claim noninterference is too strong

Write a question as `(c,r)`, where `c` identifies the requested clinical claim
and `r` contains wording, framing, or a possible presupposition.  The desired
integrity rule appears to be

\[
P(Y\mid X,c,r)=P(Y\mid X,c,r')\quad\forall r,r'.
\]

This is meaningful only if the system already has a function that separates `c`
from `r`.  Without such a declassifier, enforcing invariance to arbitrary
question changes gives

\[
P(Y\mid X,Q)=P(Y\mid X),
\]

so two different questions about the same image must receive the same output
distribution.  That is incompatible with closed-ended VQA and most targeted
open-ended questions.

With a declassifier `d(Q)=c`, the system can be useful, but the novelty moves to
the semantic parser/sanitizer `d`.  Applying `d` before the model is prompt/input
rewriting; projecting its effect from hidden states is feature steering; using
it to restrict tokens is constrained decoding.  Security terminology does not
create a sixth intervention channel.

There is also a direct behavioural collision: [Mechanisms of Prompt-Induced
Hallucination (ACL 2026)](https://arxiv.org/abs/2601.05201) studies prompts that
override vision and causally ablates model-specific prompt-copying heads.

**Verdict:** exact noninterference removes task conditioning; selective
noninterference presupposes the very semantic boundary that must be learned or
verified.

## 3. Exact taint propagation collapses under dense attention

Suppose each input value carries a provenance label and labels combine by join.
One attention output is

\[
o_t=\sum_{i\le t}a_{ti}v_i,
\qquad a_{ti}=\operatorname{softmax}(s_t)_i.
\]

For finite logits, every unmasked `a_ti` is strictly positive.  A conservative
dataflow semantics must therefore assign

\[
\operatorname{prov}(o_t)
=\bigvee_{i\le t}\operatorname{prov}(v_i)
 \vee \operatorname{prov}(q_t)
 \vee \bigvee_{i\le t}\operatorname{prov}(k_i).
\]

After one cross-modal mixing layer, every answer state is tainted by essentially
all accessible image and text sources.  Requiring that an image label merely be
present is vacuous—almost every answer passes.  Requiring image-only provenance
rejects useful question-conditioned answers.

Cancellation-aware or influence-weighted taint must replace the exact join by a
thresholded attention, ablation, or Jacobian measure.  This is attribution, not
formal provenance.  It also collides with [GIF
(2026)](https://arxiv.org/abs/2606.23277), which uses the LLM Jacobian and output
geometry to upper-bound local information flow, and with [Permissive
Information-Flow Analysis (TMLR 2025)](https://arxiv.org/abs/2410.03055), which
propagates only influential input labels.

Most importantly, nonzero image influence is not clinical correctness.  A false
claim may strongly depend on the image, while a correct answer may use language
to name a visually observed pattern.  Provenance measures dependence, not truth.

**Verdict:** exact propagation has taint explosion; useful relaxations are
attribution/audit signals and do not themselves mitigate hallucination.

## 4. Secure multi-execution cannot provide a new merger

Run the model once with trusted visual/neutralized context and once with full
context:

\[
Y_T=f(X,d(Q)),\qquad Y_F=f(X,Q).
\]

Let `G(Y_T,Y_F)` be the delivered output.  If it must be noninterfering with
respect to arbitrary changes of untrusted context while `Y_T` stays fixed, then
for all `y_F,y_F'`,

\[
G(y_T,y_F)=G(y_T,y_F').
\]

Hence `G(y_T,y_F)=g(y_T)`: the full execution is unusable.  If the merger does
read `Y_F` to choose, reject, average, or repair, it is respectively reranking,
veto/verification, distribution fusion, or editing.  At the token-distribution
level this also follows from the density-ratio identity: every changed
distribution `q` with the same support can be written as

\[
q(v)\propto p(v)\exp r(v),
\]

which is ordinary energy/guidance.

The system-security use of multi-execution is legitimate because its objective
is confidentiality/integrity, not improving a fixed predictor's clinical risk.
It gives no theorem that the trusted execution is the more accurate one.

**Verdict:** exact secure multi-execution discards the extra branch; useful
mergers fall into already excluded intervention classes.

## 5. Protected evidence subspace also reduces to steering

One may posit a visual evidence subspace `U` and enforce

\[
P_U h_{l+1}=P_U h_l,
\]

or remove the text update inside it:

\[
h' = h-P_U\Delta_{text}.
\]

If `U` is known, this is orthogonal activation projection/feature steering.  If
`U` is not known, it requires a probe, contrast pairs, labels, or an external
expert.  Preserving `P_U h` is sufficient for a downstream claim only under the
additional assumption that all later claim-relevant computation factors through
that coordinate—an assumption false for a general nonlinear Transformer.

[VTI (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4008025c2182bfe16fcc8566ee14d64-Abstract-Conference.html)
already performs visual/textual latent steering for hallucination; dynamic and
orthogonal/subspace steering is now a dense literature.  Locally, the strongest
incremental visual-evidence gate failed on two medical models, so there is no
admitted case-specific `U` to protect.

**Verdict:** standard projection plus an unverified evidence-subspace premise.

## 6. The only valid semantic lesson

The high-level distinction remains useful:

* **state protection** asks whether visual memory is overwritten; in these
  causal-prefix architectures it is already protected;
* **read integrity** asks whether a current text query still uses the fixed
  visual memory rather than a language prior.

The second is the real problem.  But changing it requires modifying attention,
hidden states, logits, search, or text.  Existing methods such as M3ID, VISTA,
SPIN, MemVR, and prompt-copy-head ablation already occupy those channels, and
local interventions repeatedly produced common-mode answer shifts without
case-specific evidence gain.

Therefore no candidate in this audit is authorized for a GPU efficacy test.
A future candidate must provide a *new observable with a sound semantic type*,
not merely attach an information-flow label to the same uncertain internal
response.

