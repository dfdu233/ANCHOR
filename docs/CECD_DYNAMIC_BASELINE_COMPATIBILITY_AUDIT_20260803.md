# CECD dynamic-baseline compatibility audit

Date: 2026-08-03 UTC  
Decision: **official implementations are not directly admissible on both CECD
models; the present ten-method envelope is sufficient only for the phenomenon
gate, not for an oral-level mitigation comparison.**

## Frozen target architectures

- Huatuo Stage-1 uses `/home/dbw/models/HuatuoGPT-Vision-7B`, whose frozen
  architecture is `LlavaQwen2ForCausalLM` (`model_type=llava_qwen2`, 28 decoder
  layers).
- Hulu uses `/home/dbw/models/Hulu-Med-4B`, whose checkpoint-owned runtime is
  `HulumedQwen3ForCausalLM` (`model_type=hulumed_qwen3`, 36 decoder layers) with
  a custom vision encoder and token-compression/multimodal preparation path.

The comparison must cover both families. Support for only one model cannot
satisfy the frozen cross-family gate.

## Official-source audit

| Method | Audited commit | Released execution path | Exact Huatuo support | Exact Hulu support | Decision |
|---|---|---|---:|---:|---|
| HulluEdit | `ac0beeba40021578d8fa01543024338ba8138c3e` | engines exist only for LLaVA-1.5/Vicuna, MiniGPT-4 and mPLUG-Owl2; the README lists Qwen/Qwen2.5-VL but the released `hulluedit/engines` tree contains no Qwen implementation | no | no | paper-native run inadmissible |
| CAI | `ed59bc8e314e75df2935a826788f6c9ec7d3fae3` | bundled LLaVA path registers only `LlavaLlamaForCausalLM`; scripts require the merged LLaVA-1.5-7B checkpoint | no | no | paper-native run inadmissible |
| ONLY | `109bffeafddf761ded65806b6707d3c124f6473c` | LLaVA path registers `LlavaLlamaForCausalLM`; the separate Qwen path is legacy `QWenLMHeadModel`/Qwen-VL-Chat, not Qwen2/2.5 or Qwen3 | no | no | paper-native run inadmissible |

The repositories are preserved under `third_party/HulluEdit`,
`third_party/CAI`, and `third_party/ONLY`. This audit does not infer
compatibility from a paper or README claim when no matching released engine is
present. CAI and ONLY carry MIT license files at the audited commits; HulluEdit
contains no repository-level license file, so its source is inspection-only and
must not be redistributed or copied into the project.

## Consequence for CECD

The existing ten-method dual-semantics envelope may answer the narrow
phenomenon question: does a physician-admitted product interaction add
reader-grounded error information beyond marginals, orbit averaging, random
controls and both Treble semantics? Treble remains a transparent clean-room
common-protocol envelope because its proceedings and released source disagree.

It may **not** support the stronger statement that CECD is a superior general
mitigation algorithm. Before such a claim, one of the following must be frozen
before method outputs:

1. an independently implemented, architecture-neutral dynamic control whose
   equations and deviations from an official method are explicit, evaluated on
   both Huatuo and Hulu; or
2. a third, officially supported model used only for paper-native dynamic
   baseline comparison, while keeping Huatuo/Hulu for the two-family CECD
   mechanism gate.

Option 1 is a common-protocol control, not a paper-native reproduction. Option
2 cannot substitute a third-model result for the required two-model clinical
mechanism evidence. In either case the preflight method list, metric closure,
compute ledger and authorization hash must be updated before any method output;
an after-the-fact appended baseline is prohibited.

## Current action

Do not expand the ten-method closure merely by renaming a port as HulluEdit,
CAI or ONLY. First run the physician-admitted behavioral gate. If it fails,
terminate CECD without spending compute on mitigation adapters. If it passes,
freeze an architecture-neutral dynamic-control contract or add an officially
supported third-model comparison before producing mitigation outputs.
