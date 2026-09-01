# Clinical Correction Alignment Is Not a New Steering Mechanism

**Decision:** 2026-08-03  
**Status:** hard NO-GO as a paper mechanism or new decoder; retain only as an
exploratory analysis after blinded clinical truth exists.

## Grounded observation

The unified 32-question VQA-RAD T2 outputs show that “the baselines do not
activate” is false:

| Method | Outputs changed vs greedy | lexical token-F1 delta vs greedy |
|---|---:|---:|
| PAI | 37.5% | -0.0092 |
| VCD | 87.5% | -0.0383 |
| AvisC | 90.6% | -0.0173 |
| Beam | 68.8% | -0.0226 |
| OPERA | 65.6% | -0.0083 |
| VISTA combined | 9.4% | not a clinical endpoint |
| VISTA SLA only | 15.6% | not a clinical endpoint |
| VISTA VSV only | 15.6% | not a clinical endpoint |

These are source-activation and lexical diagnostics, not clinical efficacy.
They show heterogeneous behavioral reach with no positive lexical evidence;
they do not show that an intervention is geometrically orthogonal to a true
clinical correction.

## Candidate and distinguishing prediction

The candidate proposed that a decoding method works only when its induced
logit/hidden update aligns with a sequence-level direction from the erroneous
clinical claim to the clinically correct claim. Subthreshold updates would
stay in the same semantic basin; large misaligned updates would change wording,
length, or unrelated claims.

The intended test would compare intervention norm with projection onto the
correct-claim direction at the first semantic divergence token. That test is
not authorized for two independent reasons.

## Direct mechanism collisions

The mechanism space is already densely occupied:

- [Activation Steering Decoding, ACL 2025](https://aclanthology.org/2025.acl-long.634/)
  identifies directional hallucination patterns in LVLM hidden states and
  contrasts positive and negative steering predictions.
- [VTI, ICLR 2025 Spotlight](https://openreview.net/forum?id=LBl7Hez0fF)
  attributes hallucination to visual/textual representation misalignment and
  intervenes in latent space to stabilize vision features.
- [SchröMind, ICASSP 2026](https://arxiv.org/abs/2602.09528) explicitly learns
  token-level minimal-cost mappings between hallucinatory and truthful
  activations rather than assuming one global shift.
- [MESA](https://arxiv.org/abs/2604.07914) directly targets entangled steering
  whose hallucination suppression shortens outputs or shifts token
  distributions, then performs selective intervention to preserve generation
  behavior.
- [TLVS](https://arxiv.org/abs/2606.07647) finds that visual conditioning is
  sparse over decoding steps and applies token-specific, sensitivity-adaptive
  steering only where the visual signal is strong.

Consequently, “measure whether the intervention direction points toward the
truthful answer, then steer selectively” is not an open mechanism. Replacing
generic truthful/hallucinatory examples with medical answers is a setting
change, not a new causal account.

## Missing truth in current OE data

For open medical answers, the benchmark reference string is not a unique
correct sequence. A correct response can use different concepts, granularity,
uncertainty, or clinically equivalent wording. Defining a “clinical correction
direction” from one reference would therefore make the proposed mechanism
depend on lexical realization rather than clinical truth.

The frozen physician review can establish which atomic claims are supported,
refuted, omitted, or inappropriately certain. Until those labels exist, neither
the endpoint nor the correction vector is admitted. Automatic token-F1 and an
LLM judge may not create it.

## Frozen decision

Do not implement a new correction-alignment decoder, activation direction, or
GPU probe. Do not claim that the current baselines fail because their updates
are orthogonal to clinical truth.

After clean blinded consensus only, the following may be reported as an
exploratory diagnostic, not the preregistered primary endpoint:

1. source activation rate versus physician-verified clinical correction rate;
2. transition taxonomy: identity, correct claim repair, new fabrication,
   omission, location/attribute exchange, or certainty-only rewrite;
3. clinical benefit per changed answer at matched claim coverage and length.

This analysis can explain why a published method fails to transfer to medical
VQA. It cannot be promoted into a new steering-method contribution without a
causal variable and intervention not already covered by the works above.

