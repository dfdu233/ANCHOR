# Specificity Ratchet teacher-forcing runtime v1

> **F6-rejected before scientific scoring.** Isolated automatic parent/child
> targets confound specificity with grammar and discourse boundaries. The CLI
> and default Python runtime now refuse execution. This document is retained
> only as historical contract evidence; the active path replays the complete
> visible Huatuo answer as frozen in
> `docs/SPECIFICITY_RATCHET_FROZEN_RESEARCH_CONTRACT_20260802.md`.

## What this runtime admits

The runtime measures the frozen physician-admitted child-over-parent estimand.
It does not admit clinical edges, infer missing labels, or use a model/LLM as
truth.  It exits before model construction unless both `samples.jsonl` and its
inseparable `metadata.json` certify:

- `specificity-ratchet-mechanism-v1`;
- `status = physician_admitted`;
- image-disjoint dev/test grouping;
- an exact manifest SHA-256 and row count.

The current blank reviewer sheets therefore remain a deliberate hard stop.

## Exact contextual-offset contract

A production adapter implements `TeacherForcingAdapter` from
`specificity_ratchet_teacher_forcing_v1.py`.  For each same-image parent/child
target and its text-only counterpart it returns:

1. response-content token IDs produced by the exact serialized chat path;
2. each response token's offset relative to the raw target, with an explicit
   `unicode_character` or `utf8_byte` unit;
3. gold-token log probability at every declared decoder layer;
4. prompt, target, image, serialized-input, tokenizer/template, and model
   fingerprints;
5. `contextual_offsets_certified = true` only after exact-template alignment is
   checked.

Standalone target tokenization is not evidence of alignment.  Assistant
delimiters, EOS, and template suffixes are excluded from the response-content
trace.  Every non-whitespace target byte must be covered.  Every frozen
constraint occurrence is checked independently, then unioned.  Leading/trailing
whitespace BPE spill is allowed; a token that also covers another
non-whitespace character makes the row unidentifiable and fails.  There is no
whole-sentence fallback.

Image and text-only traces must have identical response IDs, offsets, layer
IDs, and template ID.  Text-only final-layer NLL is recorded only as a lexical
frequency nuisance; it cannot define visual support.

For intermediate-layer probabilities, the adapter must apply the production
model's final decoder norm and production LM head to each layer residual before
selecting the teacher-forced gold token.  Raw residual dot products are not
admissible probabilities.

## Frozen output signals

Each atomic sample shard contains, per layer:

- exact constraint-token mean log probability `g_l`;
- full same-image parent mean log probability `b_l`;
- full child sequence mean log probability;
- an exact-count, relative-position-matched parent mean;
- an exact-count, relative-position-matched child non-constraint mean;
- constraint-minus-parent, constraint-minus-matched-parent,
  constraint-minus-matched-child, and child-minus-parent contrasts.

It also records exact parent, child, and constraint counts and final-layer
text-only NLL for those same token sets.  Matching never duplicates tokens; a
row with too few comparison tokens fails rather than changing K.

## Negative-control plan

After all shards are complete, `controls.json` freezes two deterministic
controls:

- shuffled parent pairing only within exact `(split, edge_type,
  parent-token-count, constraint-token-count)` bins and across cases;
- role permutation only within exact `(split, parent-token-count,
  child-token-count, constraint-token-count)` bins, across cases and roles.

Both use seeded cyclic derangements.  Singleton or structurally impossible bins
are marked ineligible and lower reported coverage.  They are never widened by a
post-hoc length caliper.  Equal-norm random activation directions remain an
intervention-stage adapter responsibility and are not simulated by this
teacher-forcing collector.

## Atomicity, fingerprint, and resume

`config.json` locks the manifest and metadata hashes, image root, adapter/model/
tokenizer/template fingerprint, source-code hash, split, seed, exact command,
and offset/text-only policies. `ordered_keys.json` locks sample order. Each
sample is written to a temporary file, fsynced, and atomically renamed. Resume
validates the config, row hash, and payload checksum before reusing a shard;
any drift or corruption aborts the run. `COMPLETE.json` is written only after
all selected rows and the control plan exist.

## Huatuo and Hulu hook audit (no GPU run)

No production adapter is claimed yet.

| Family | Existing useful path | Unresolved conformance requirement |
|---|---|---|
| HuatuoGPT-Vision-7B | `/home/dbw/HuatuoGPT-Vision/cli.py` serializes a prompt as `<|user|>...<|assistant|>\n`; decoder blocks are `bot.model.model.layers`, final norm is `bot.model.model.norm`, and the multimodal expansion path is `prepare_inputs_labels_for_multimodal_new`. | The Specificity bridge now reuses the exact-context implementation in `huatuo_lockin_adapter_v1.py`, scoring the complete target as an empty-prefix continuation with Huatuo's `" \n"` suffix. CPU contract tests pass; the mandatory real-model conformance canary remains blocked by physician admission. |
| Hulu-Med-4B | `processing_hulumed.py::_process_conversation_with_label` tokenizes each rendered message, masks the prefix, and exposes supervised labels; decoder blocks are `runtime.model.model.layers`. | Processor-only inspection proved that the native slice includes `<|im_end|>`. Exact raw-target selection is implemented as reusable plumbing, but its factory is intentionally disabled: the current pack contains Huatuo outputs and cannot establish spontaneous Hulu generation. A separate Hulu full-answer substrate is required only after the Huatuo mechanism gate. |

Required adapter canary before any full job:

1. ASCII, UTF-8, punctuation-adjacent, leading-space, and repeated-constraint
   targets;
2. response content IDs reconstructed from offsets exactly equal the selected
   expanded labels for image and text-only paths;
3. final declared-layer probabilities numerically match the model's ordinary
   final logits on the same selected tokens;
4. intermediate layer uses final norm + LM head and all probabilities are
   finite and non-positive;
5. image SHA is identical for parent/child and absent for text-only;
6. one intentionally merged boundary (for example `left` inside
   `left-sided`) is rejected.

Until these pass, `--adapter-factory` is intentionally absent and the runtime
refuses rather than routing through common eval or guessing a hook.

## Command protocol

After physician review and successful manifest compilation, the first command
is CPU-only preflight:

```bash
/opt/miniconda3/envs/huatuo/bin/python \
  anchor/corrected_sgta/specificity_ratchet_teacher_forcing_v1.py \
  --manifest corrected_runs/specificity_ratchet/mechanism_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/mechanism_manifest_v1/metadata.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/teacher_forcing_v1/preflight \
  --split dev \
  --preflight-only
```

The audited production invocation has the same inputs plus a model-specific
factory and immutable JSON config:

```bash
/opt/miniconda3/envs/huatuo/bin/python \
  anchor/corrected_sgta/specificity_ratchet_teacher_forcing_v1.py \
  --manifest corrected_runs/specificity_ratchet/mechanism_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/mechanism_manifest_v1/metadata.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/teacher_forcing_v1/<model>/dev \
  --split dev \
  --adapter-factory 'corrected_sgta.huatuo_specificity_ratchet_adapter_v1:build_adapter' \
  --adapter-config configs/specificity_ratchet/<model>.json
```

Dev is collected and analyzed before test. A test command uses a new output
directory and `--split test`; it must not be used to tune layer choice,
residualization, or thresholds.
