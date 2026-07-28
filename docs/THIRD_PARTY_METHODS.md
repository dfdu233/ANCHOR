# Third-Party Method Sources

This document records external mitigation repositories discovered during ANCHOR development. It is intended for migration and reproducibility; do not treat these entries as claims that the method has been audited on ANCHOR.

| Method | Local path | Upstream | Commit | Migration status |
|---|---|---|---|---|
| AGLA | `third_party/AGLA/` | `https://github.com/Lackel/AGLA` | `efa126347c41631152a70d7db0a6ac0708bd9d00` | Local clone present; not vendored in this commit because it contains a nested `.git`. Reclone or vendor without `.git` when needed. |
| ClearSight | `third_party/ClearSight/` | `https://github.com/ustc-hyin/ClearSight` | `5466c945be8cbd69ecc08b09455d8ed11f37ce67` | Local clone present; contains outputs/data directories. Reclone or curate before committing. |
| VHR | `third_party/VHR/` | `https://github.com/jinghan1he/VHR.git` | `f0db54a7eae62b4b8d1d585636a446ed40799512` | Small local clone present; may be vendored later after license/protocol review. |
| VISTA | `third_party/VISTA/` | `https://github.com/LzVv123456/VISTA.git` | `efcf499919e066755e7c33778fbfd864c204329c` | Local clone present; large assets and unresolved LFS-style files require curation. Use as the closest LET comparator. |

Recommended migration rule: either add these as Git submodules pinned to the commit above, or vendor a minimal source-only copy without `.git`, downloaded datasets, generated outputs, caches, or unresolved LFS pointer files.
