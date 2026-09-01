# CECD Huatuo + Hulu Stage-1 outcome-blind readiness audit

Date: 2026-08-03 04:20 UTC

## Verdict

**Engineering-ready, scientifically human-gated.**  The exact two-model
Stage-1 path is runnable and crash-resumable, but it is correctly not authorized
today: all eight independent review/attestation return files are absent.  No
Stage-1 state, admitted model output, sealed outcome, or GPU job was opened by
this audit.

## Readiness evidence

| Boundary | Outcome-blind check | Result |
|---|---|---|
| Data | Frozen manifest and box hashes; selected DICOM presence | manifest `23dec244...5254fa`, boxes `4e2c06e3...10b5a`; 160/160 claims resolve, 0 missing DICOMs |
| Balance | Four findings x four reader-vote bins x ten claims | exact 16 bins, ten per bin; 160 claims on 154 images |
| Orbit | Five scientific renders x three prompts plus 3+1 exact controls | 19 unique cells/claim; full model gate independently requires 3,040 rows, 160 complete orbits, 40 orbits/finding |
| Token/readout | Model-specific tokenizer and standard generation path | both environments map Yes/No/Maybe to the same three distinct one-token IDs; direct FP32 final-hidden readout must pass a hash-bound ordinary `generate(max_new_tokens=1)` check |
| Models | Local full content identity | Huatuo 4 shards / 15,871,393,048 bytes; Hulu 2 shards / 9,666,263,512 bytes |
| Runtime code | Executable provenance | patched to content-hash every checkpoint runtime asset and all 18 Huatuo external runtime Python files; Stage-1 join re-hashes them after scoring |
| Hardware | Sequential single-GPU execution under one `flock` | RTX 4090 reports 48,493 MiB free, 18 MiB used, 0% utilization; both model families have already completed real local probes in these environments |
| Storage | Current free space and projected Stage-1 growth | about 230 GiB free, above the 100 GiB reserve; JSON-only Stage-1 output is projected well below 0.1 GiB |
| Persistence | VS Code/SSH independence | admission and transition monitors are PID-1-adopted; research status reports both `alive` |

Stage-1 is a behavioral final-hidden readout, not a hidden-layer hook run.  The
config and downstream gates explicitly keep general hidden-state authorization
false.  Hidden intervention can only be enabled later inside the separately
frozen dual-semantics comparison.

## Runtime estimate

The exact Huatuo 19-cell engineering canary took 5.02 seconds after model load.
Existing 640-case real probes measured 315.6 seconds for Huatuo and 417.2
seconds for Hulu.  With 3,040 scored cells/model, two one-claim canaries, four
model loads/content hashes, packing, and 5,000 bootstrap draws, the conservative
wall-time estimate is **45--75 minutes** on the current GPU.  This is an
operational estimate, not a scientific result.

## Fail-closed repairs made in this audit

1. The two-model input gate now requires and hash-binds the real next-token
   conformance artifact.  A missing, failed, non-finite, wrong-model, wrong-cell
   or tolerance-drifted artifact is rejected.
2. Formal model identity now includes non-weight runtime assets.  Huatuo's
   external `cli.py`/`llava` Python tree and Hulu's local custom model,
   processor and image-processing code can no longer drift behind an unchanged
   weight hash.  The input gate independently re-hashes current assets.
3. The dual-semantics monitor now validates and reuses an existing write-once
   authorization.  It no longer turns into a false transition error merely
   because a later runner has begun writing inside the separately authorized
   method-output root.

Focused regression: **70 passed** across the complete CECD admission,
factorial, two-model input-gate, Treble-collision, and transition test set.  The transition monitor
was safely restarted onto the repaired code as supervisor/child
`557275/557278`; supervisor PPID is 1, heartbeat is fresh, and the recovery
watchdog reports it alive.

## Resume and handoff contract

- Every scored cell is an atomic JSON shard.  A same-config restart skips only
  fully validated shards; a changed manifest, admission, source, tokenizer,
  checkpoint/runtime asset, transform, prompt, or model config is rejected.
- Each canary/full directory is resumed automatically when its `config.json`
  exists.  Packing occurs only after every required shard validates and the
  invocation has zero claim errors.
- A detached Stage-1 nonzero exit is intentionally fail-stop, not silently
  auto-retried.  The shards remain resumable, but an operator must first audit
  the log and then re-launch the identical job.  This preserves the frozen
  scientific contract rather than converting an unknown failure into an
  unattended retry loop.
- On success, the separate live transition monitor revalidates admission,
  both raw-run hashes, the exact two-model contract, analysis provenance, and
  the behavioral gate.  A NO-GO terminates.  A pass waits for the independent
  outcome-blind dual-semantics preflight and then emits only the narrow
  controlled-comparison authorization.  The controlled-comparison runner is a
  separate downstream deliverable; Stage-1 does not grant general GPU or paper
  authority.

## Verifiable commands

```bash
cd /home/dbw/ANCHOR

PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_factorial_v1.py \
  tests/test_verify_cecd_two_model_stage1_v2.py \
  tests/test_monitor_cecd_admission_pipeline.py \
  tests/test_monitor_cecd_dual_semantics_transition_v1.py

bash -n scripts/run_cecd_two_model_stage1_v2.sh
python scripts/research_status.py
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
df -BG --output=avail,target /home/dbw/ANCHOR /workspace
```

The launch command remains owned by
`scripts/monitor_cecd_admission_pipeline.py`; running
`scripts/run_cecd_two_model_stage1_v2.sh` directly before the admitted analysis
exists fails before model loading.
