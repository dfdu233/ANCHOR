# OE brevity and mitigation control audit

**Date:** 2026-08-02  
**Dataset:** VQA-RAD official open-ended test split  
**Decision:** lexical scores are confounded by answer length and cannot promote
any current mitigation method; T3 remains blocked on the existing blinded
clinical review pack.

## Question and scope

This audit asks whether an apparent OE gain can be reproduced by a reference-
blind transformation that simply removes answer suffixes. It does **not** ask
whether shorter answers are clinically safer. Token-F1, ROUGE-L and exact
reference-phrase coverage are transparent lexical diagnostics, not clinical
truth or claim coverage.

Every control is deterministic and image/reference blind after generation:
keep the first sentence or the first 8--64 words. Uncertainty uses 5,000
image-cluster bootstrap replicates over 120 unique images for the 200-question
full runs and 24 images for the 32-question T2 mitigation screen.

The evaluator now records its exact command, source SHA-256, input hashes,
bootstrap count and seed. Runs used an isolated Python cache because multiple
concurrent sessions share the worktree; this changes no metric or row.

## Full qualified native outputs

The three input artifacts had already passed their generation qualification
contracts: Huatuo uses the admissible 512-token output; Hulu and LLaVA-Med use
the admissible 256-token outputs.

| Model | Original token-F1 | Coverage-matched control | Token-F1 delta [95% CI] | Reference-phrase coverage delta [95% CI] | Token delta [95% CI] |
|---|---:|---|---:|---:|---:|
| HuatuoGPT-Vision-7B | 0.03761 | first 48 words | +0.01558 [0.01245, 0.01886] | -0.010 [-0.02564, 0] | -40.06 [-46.27, -34.34] |
| Hulu-Med-4B | 0.12937 | first 48 words | +0.00272 [0.00112, 0.00426] | -0.010 [-0.02564, 0] | -7.47 [-9.39, -5.71] |
| LLaVA-Med-v1.5-7B | 0.10726 | first 48 words | +0.00012 [0, 0.00032] | 0 [0, 0] | -0.63 [-1.06, -0.27] |

For Huatuo, a mechanical 48-word truncation raises token-F1 by 41.4% relative
while the point estimate of exact reference-phrase coverage remains at the
predeclared -1 percentage-point tolerance. The same direction appears in Hulu;
LLaVA is already short and changes little. This is direct evidence that lexical
overlap can reward saying less even without better visual reasoning. The
coverage quantity here is still lexical and must not be called omission recall.

Primary artifacts:

- `corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v3_512/brevity_controls_v1.json`
  (`7378592f...43d04`);
- `corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_greedy256_v1/brevity_controls_v1.json`
  (`c5b1e621...b96e1`);
- `corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_greedy256_v1/brevity_controls_v1.json`
  (`72d2b6bf...20f1`).

## Mitigation T2 against canonical greedy

All six T2 paths passed output qualification; the canonical greedy backend is
32/32 generated-token exact against the common adapter. Method activation is
therefore real, but activation is not efficacy.

| Method | Token-F1 delta vs greedy [95% CI] | Reference-phrase coverage | Median tokens |
|---|---:|---:|---:|
| beam | -0.02262 [-0.07099, 0.01124] | 0.1875 | 15.0 |
| VCD | -0.03833 [-0.08549, 0.00255] | 0.0938 | 11.5 |
| OPERA | -0.00831 [-0.05920, 0.03614] | 0.1875 | 14.5 |
| PAI | -0.00919 [-0.03217, 0.00164] | 0.2188 | 11.0 |
| AvisC | -0.01732 [-0.07513, 0.03077] | 0.1562 | 16.5 |
| canonical greedy | reference | 0.2188 | 10.0 |

No method has a positive lexical point estimate. More importantly, first-
sentence shortening raises token-F1 for VCD by +0.01241 [0.00239, 0.02484]
and AvisC by +0.01240 [0.00328, 0.02451] while exact reference-phrase coverage
does not change at all; it removes 10.69 and 11.22 tokens on average. Thus even
within a method, a larger lexical score can be manufactured without touching
the image, decoder state or answer-bearing reference phrase.

## Scientific decision

1. Token-F1, ROUGE-L and reference-phrase coverage remain auxiliary only.
   None may authorize T3 or support a medical-hallucination claim.
2. VCD, OPERA, PAI, AvisC and beam remain at T2. Their point estimates do not
   justify full generation, and the 32-case screen lacks clinical labels.
3. The already frozen blinded pack at
   `corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t2_v1/`
   is the only admissible promotion route. It contains 24 image groups and all
   six methods; reviewers must evaluate direct correctness, visual claim
   support, omission, commitment, harm, length and claim count before unblinding.
4. These results do not establish that every mitigation method is clinically
   harmful. They establish a narrower, decisive boundary: current automatic
   OE overlap scores cannot distinguish a real mitigation from reference-blind
   suffix deletion, and no present T2 output provides positive evidence that
   warrants additional GPU cost.

This audit therefore strengthens the unified evaluation framework and explains
why the current mitigation baselines are not promoted, but it is not itself an
ICLR mechanism claim.
