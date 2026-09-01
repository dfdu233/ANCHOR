# Specificity Ratchet full-visible-answer runtime v1

This is the only active mechanism runtime. The old isolated parent/child
teacher-forcing path is F6-rejected and now hard-refuses execution. No GPU run
is authorized until two independent physician reviews, blinded physician
adjudication, attestations and the CPU compiler all pass.

The current 70-case v2 pack additionally fails the label-blind confirmatory
lexical-overlap ceiling (dev at most 8 blocks, test at most 5, required 10).
Thus this runtime is ready for a bounded pilot but does not authorize a broad
confirmatory claim. A targeted higher-overlap pack must be frozen first.

## Frozen sequence

1. Preserve both returned physician sheets independently and merge them using
   `SPECIFICITY_RATCHET_REVIEW_RETURN_WORKFLOW_V1.md`.
2. Complete blinded adjudication and attestations in a disposable adjudicated
   pack. Never overwrite the blank source pack.
3. Compile the physician-admitted full-answer manifest. This localizes the
   exact constraint inside the complete Huatuo OE answer, excludes the two
   frozen tokenizer-boundary failures, creates image-disjoint splits and
   freezes same-split modality/anatomy swap pools.
4. Run CPU preflight for the complete frozen manifest.
5. Run one engineering native-generation identity canary. It uses the frozen
   greedy-512 source contract and directly captures `output.sequences`. A
   failed canary exits nonzero and its shard is immutable; do not select a
   replacement.
6. Run the complete native capture for every selected case. Scientific replay
   additionally requires exact equality between terminal-EOS/PAD-trimmed
   native IDs and contextual teacher-forcing IDs, plus equal native visual-token
   length for the own image and both frozen swaps. A one-case canary cannot
   authorize scientific replay.
7. Only after the complete capture passes, launch the Huatuo replay. Every row
   uses the full answer under the own image, exactly two different-case swaps
   with equal visual-token length, and a secondary text-only trace. The
   label-blind dev/test assignments and analyzer source are already frozen in
   manifest metadata, so one `all` run avoids duplicate model work without
   permitting post-data layer or gate selection.

## Commands after human admission

```bash
python anchor/corrected_sgta/compile_specificity_ratchet_replay_manifest_v1.py \
  --pack /work/specificity_ratchet/adjudicated_pack_v1 \
  --attestations /work/specificity_ratchet/adjudicated_pack_v1/physician_attestations.json \
  --output corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl \
  --metadata-output corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json

python anchor/corrected_sgta/specificity_ratchet_visible_replay_v1.py \
  --manifest corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json \
  --native-capture corrected_runs/specificity_ratchet/native_capture_huatuo_all_v1/native_capture.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/visible_replay_huatuo_all_v1 \
  --split all --preflight-only

python anchor/corrected_sgta/capture_huatuo_specificity_native_v1.py \
  --manifest corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/native_capture_huatuo_dev_canary_v1 \
  --split dev --limit-cases 1 \
  --adapter-factory corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter \
  --adapter-config configs/specificity_ratchet/huatuo_full_replay_v1.json

python anchor/corrected_sgta/capture_huatuo_specificity_native_v1.py \
  --manifest corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/native_capture_huatuo_all_v1 \
  --split all \
  --adapter-factory corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter \
  --adapter-config configs/specificity_ratchet/huatuo_full_replay_v1.json

python anchor/corrected_sgta/specificity_ratchet_visible_replay_v1.py \
  --manifest corrected_runs/specificity_ratchet/replay_manifest_v1/samples.jsonl \
  --metadata corrected_runs/specificity_ratchet/replay_manifest_v1/metadata.json \
  --native-capture corrected_runs/specificity_ratchet/native_capture_huatuo_all_v1/native_capture.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf \
  --output-dir corrected_runs/specificity_ratchet/visible_replay_huatuo_all_v1 \
  --split all \
  --adapter-factory corrected_sgta.huatuo_specificity_ratchet_adapter_v1:create_adapter \
  --adapter-config configs/specificity_ratchet/huatuo_full_replay_v1.json

python anchor/corrected_sgta/analyze_specificity_ratchet_visible_replay_v1.py \
  --run-dir corrected_runs/specificity_ratchet/visible_replay_huatuo_all_v1 \
  --output corrected_runs/specificity_ratchet/visible_replay_huatuo_all_v1/analysis.json \
  --bootstrap-replicates 5000 --seed 7319
```

The active analyzer is `analyze_specificity_ratchet_visible_replay_v1.py`.
The older one-case/full-replay analyzer is retained only as engineering
history and must not produce a paper result. The active analysis requires at
least eight distinct cases per primary role and three edge types in each
label-blind split. At least 95% of 5,000 case-cluster bootstrap replicates must
be valid. Analyzer source, gates, lexical fixed effect, bootstrap count and
seed are all bound before traces exist; output is write-once.

Run GPU commands through the repository's detached-job supervisor so VSCode or
SSH disconnection does not terminate them. The research watchdog monitors
registered jobs; it does not bypass scientific gates. The persistent clinical
monitor owns the frozen successor chain and advances only after validating the
preceding write-once artifact:

```text
specificity-ratchet-native-canary-v1
  -> specificity-ratchet-native-full-capture-v1
  -> specificity-ratchet-visible-replay-v1
  -> specificity-ratchet-visible-analysis-v1
```

All three GPU stages acquire the shared blocking
`corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock`, so a simultaneously
admitted CECD job is serialized rather than failed or co-scheduled. Any canary,
capture, replay, or contract failure is terminal and is never retried. Analyzer
exit code 1 is a scientific `failed`, `underpowered`, or `pilot_only` result
when a valid `analysis.json` exists; it is not treated as an operational retry.
No result from this 70-case pack authorizes a confirmatory claim or an automatic
second-model launch.

## Primary observable

For each layer, compute the mean log-probability of exact constraint tokens
minus relative-position-matched non-constraint tokens in the same full answer.
The frozen analysis uses the first recorded decoder layer and final decoder
layer; it does not choose a layer after seeing data. The mechanism signature is
conjunctive: error cases have weaker early own-minus-swap evidence, a positive
error-selective own-image late shift, a positive late shift that survives the
mean of both swaps after frozen nuisance adjustment, and the lower bootstrap
bound for the swap-surviving fraction exceeds 0.50. Text-only transition is a
nuisance control and secondary sensitivity, never primary visual evidence.
The nuisance design also contains an exact normalized added-constraint lexical
fixed effect; if scientific role is not identifiable after that control, the
split is underpowered rather than silently fit with a pseudoinverse. Each split
requires at least eight independent cases per primary role and three edge types.

No second model or mitigation begins before the frozen Huatuo signal gate.
Hulu needs its own spontaneous outputs and physician-admitted substrate; the
current Hulu helper factory intentionally refuses scientific use.
