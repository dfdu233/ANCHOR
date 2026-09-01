# Huatuo / Qwen2.5 checkpoint lineage audit — 2026-08-03

## Decision

**All three local checkouts are byte-complete and usable, but the released
Huatuo checkpoint is admitted only as a Qwen2.5-VL family-level descendant,
not as a reproducible exact child of either audited public checkpoint.**

The current weights strongly favor the raw Qwen2.5-VL release over the
Qwen2.5 text release as the relevant reference: a frozen numerical sample is
about 51 times closer to raw VL (median relative L2 0.00858 versus 0.43972).
This is relationship evidence, not historical proof of the exact initializer.

## Repository and content integrity

| checkpoint | detached clone | Git HEAD | LFS fsck | pointer files | index/header | tensors / tensor bytes |
|---|---:|---|---:|---:|---:|---:|
| Huatuo medical | exit 0 | `451ac32400e36cfd07b41b62cbe63e6894895b38` | pass | 0 | pass, 4/4 shards | 729 / 16,584,333,312 |
| Qwen2.5-VL-7B-Instruct | exit 0 | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | pass | 0 | pass, 5/5 shards | 729 / 16,584,333,312 |
| Qwen2.5-7B-Instruct | exit 0 | `a09a35458c702b33eeacc393d103063234e8bc28` | pass | 0 | pass, 4/4 shards | 339 / 15,231,233,024 |

For every repository, the declared index keys equal the union of Safetensors
header keys, every key is mapped to the correct shard, shapes and dtypes are
readable, and the tensor-byte sum exactly matches `metadata.total_size`. Git
working trees are clean. The audit did not instantiate a model or use a GPU.

## Complete tensor equality

Every same-schema tensor was streamed and assigned a BLAKE2b content digest;
the rates below are therefore exhaustive exact-byte comparisons, not samples.

| comparison / component | exact tensors | exact tensor bytes | interpretation |
|---|---:|---:|---|
| medical vs raw VL — vision encoder | 55 / 385 | 140,800 / 1,263,951,360 (0.0111%) | nearly all substantive vision weights changed |
| medical vs raw VL — merger/projector | 0 / 5 | 0 / 89,148,928 | all changed |
| medical vs raw VL — language model | 3 / 339 | 21,504 / 15,231,233,024 (0.000141%) | all substantive LM weights changed |
| medical vs text — language model | 0 / 339 | 0 / 15,231,233,024 | no exact text-model transplant |
| raw VL vs text — language model | 0 / 339 | 0 / 15,231,233,024 | distinct released LMs |

The 55 retained vision tensors occupy only 140.8 KB and are predominantly
normalization weights. The three retained LM tensors occupy 21.5 KB. Thus a
tensor-count rate such as 55/385 must not be presented without the byte rate.

After the exhaustive equality test, the audit selected up to four changed
tensors per component by a frozen SHA-256 ordering (only tensors at most 40 MB)
and computed numerical distances:

| comparison | sampled changed tensors | median relative L2 |
|---|---:|---:|
| medical vs raw VL | 12 | 0.008582 |
| medical vs text | 4 common LM tensors | 0.439719 |
| raw VL vs text | 4 common LM tensors | 0.439959 |

The close medical/raw distance is consistent across the sampled vision,
merger, and LM components. The complete per-tensor rows and component totals
are retained in the JSON artifact.

## README and configuration evidence

The medical model card front matter names
`Qwen/Qwen2.5-VL-7B-Instruct` as `base_model`, and the checkpoint has the same
729-key architecture and tensor shapes as that release. However:

1. it does not pin a base revision;
2. it does not provide a reproducible conversion/training command or manifest;
3. it still contains a conflicting legacy sentence saying the 7B model was
   trained from Qwen2-7B with LLaVA-v1.5;
4. all substantive weights in vision, merger, and language components differ
   from the audited raw-VL release.

The vocabulary file is identical across all three checkouts, but the Huatuo
`tokenizer.json`, tokenizer configuration, merges file, chat template, and
model configuration are not byte-identical to raw VL. These compatibility
facts do not recover historical ancestry.

## Claims admitted and prohibited

Admitted:

- “The released Huatuo checkpoint uses the Qwen2.5-VL architecture and its
  current weights are materially closer to the audited raw Qwen2.5-VL release
  than to the audited Qwen2.5 text release.”
- “Medical adaptation changed substantive tensors in the vision encoder,
  merger, and language model.”

Prohibited:

- “Commit `cc594898…` is the exact Huatuo parent.”
- “Huatuo was produced by a documented raw-VL plus text-model conversion.”
- “The vision tower was frozen or copied unchanged.”
- Any causal before/after attribution treating the released Huatuo checkpoint
  and either public release as a controlled parent/child pair.

For controlled children, use the pinned raw-VL commit as the explicitly chosen
public parent and record the new training manifest. Use released Huatuo only as
external validation unless the authors provide the exact base revision and
reproducible recipe.

## Reproduction and artifacts

```bash
cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m pytest -q \
  tests/test_audit_checkpoint_lineage_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  .venv-full/bin/python \
  anchor/corrected_sgta/audit_checkpoint_lineage_v1.py \
  --output corrected_runs/checkpoint_lineage_audit_v1/audit.json
```

- Audit result: `corrected_runs/checkpoint_lineage_audit_v1/audit.json`
- Audit script SHA-256: `6528975b3d64bdcec82d5d2722c2b6d3504e0c1e4b59879ad626da1f934f51d5`
- Result SHA-256: `c530301d0d435c7cf704fcefe6a1b57b30a8d97007373ec4a6fc9e3bfc5885fe`
- Focused tests: 3 passed
- Final free disk at handoff: approximately 238 GB, above the 100 GB floor

The numerical-distance rows are a deterministic stratified sample; only the
exact-equality statistics cover every tensor. Neither statistic by itself can
prove an undocumented historical initialization path.
