# Third-Party Method Sources

The current T0 source audit is hash-bound at
`results_reference/baseline_t0_source_audit_20260802.json`.

This document records external mitigation repositories discovered during ANCHOR development. It is intended for migration and reproducibility; do not treat these entries as claims that the method has been audited on ANCHOR.

| Method | Local path | Upstream | Commit | Migration status |
|---|---|---|---|---|
| AGLA | not vendored | `https://github.com/Lackel/AGLA` | `efa126347c41631152a70d7db0a6ac0708bd9d00` | Official HEAD and entry point audited 2026-08-02, but the repository has no root method-code license; `not_admissible`. |
| ClearSight | not vendored | `https://github.com/ustc-hyin/ClearSight` | `5466c945be8cbd69ecc08b09455d8ed11f37ce67` | Official HEAD audited 2026-08-02. The embedded LLaVA subtree is Apache-2.0, but no root license covers ClearSight method code; `not_admissible`. |
| MedVR | not vendored | `https://github.com/alibaba-damo-academy/MedVR.git` | `4fdd671e29487f455c0b88ef9f73d96ca88ff298` | ICLR 2026 paper-native model/training candidate. Apache-2.0 training code is public, but no MedVR checkpoint is released and README defers the tool-enabled evaluation code; `not_admissible`, and never a model-agnostic common-protocol decoder. |
| VHR | `third_party/baselines/VHR/` | `https://github.com/jinghan1he/VHR.git` | `f0db54a7eae62b4b8d1d585636a446ed40799512` | Apache-2.0 source is pinned. Official VHR replaces HF `LlamaSdpaAttention` under Transformers 4.45; the certified medical LLaVA backend is custom Mistral/4.36. Current medical T1/T2 are therefore closed, and a local Mistral rewrite cannot be reported as official VHR. |
| VISTA | `third_party/baselines/VISTA/` | `https://github.com/LzVv123456/VISTA.git` | `efcf499919e066755e7c33778fbfd864c204329c` | MIT licensed; audited core files are byte-identical to official HEAD. At the unified 256-token OE budget, method-off is 32/32 token-identical; VSV-only and SLA-only each change 5/32 outputs and combined changes 3/32, with all arms response-qualified. A 24-image/101-unique-answer blinded multiarm clinical pack is frozen; T3 efficacy remains unauthorized. |

`configs/unified_eval/method_ladder_v1.json` is the sole paper-qualification
authority. Presence in this table or in the runtime registry never promotes a
method. When a T0-passing source is needed, vendor a minimal source-only copy
without `.git`, downloaded datasets, generated outputs, caches, or unresolved
LFS pointer files, and then pass method-off token identity before functional
testing.
