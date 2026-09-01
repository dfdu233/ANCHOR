# CECD system/PIH native-eager canary factory handoff v1

Status: **source ready; explicit serial canary pending**.  No checkpoint was
loaded, no GPU API was queried, no model forward was executed, and no canary
artifact was produced while preparing this handoff.

## What is now executable

`cecd_system_pih_canary_factories_v1.py` provides separate model and input
factories for HuatuoGPT-Vision and Hulu-Med.  Model imports are local to the
two model factories, so importing the module or calling `factory_description`
cannot initialize either model.

Each input factory uses the model's existing native adapter:

- Huatuo uses `HuatuoOEAdapter._inputs`, then
  `prepare_inputs_labels_for_multimodal_new`.  The single signed image
  placeholder is replaced by the observed number of projected visual tokens.
- Hulu uses `HuluAdapter._inputs`, then
  `prepare_inputs_labels_for_multimodal`.  Exact processor-emitted image-token
  positions must form one nonempty contiguous run, and visual compression is
  rejected if it changes the sequence length.

Both return exactly `provenance`, `forward_kwargs`, and `input_identity`.
`forward_kwargs` contain already-expanded batch-one `inputs_embeds`, matching
attention and position tensors, `input_ids=None`, and `use_cache=False`.
Pixel tensors and raw images do not enter the canary forward.  Every non-image
template token is conservatively labelled `user_text`; no prefix is relabelled
as a true system role without provenance.

## Frozen input

- Anonymous record: `vindr_train_canary_d925309691e7929d`
- Finding: `aortic_enlargement`
- DICOM: `/workspace/vinbigdata/train/d925309691e7929d905eaa42f081833f.dicom`
- DICOM SHA-256: `2893814198d6126656311fe55c08515a86eeb1917535ba273e11d5429a18bec5`
- Prompt SHA-256: `70bf1620f6c79ccd2c007f56b4d47cc4f4c68baa993054e59e0381edf8c7db20`

The DICOM identity is checked before image decoding, and the runtime artifact
binds signed input-ID identity, expanded length, dtype, and exact provenance
fingerprint without storing model logits in the input identity.

## Validation completed without models

The fake/monkeypatch suite covers native-adapter reuse, exact Huatuo visual
expansion, exact Hulu image-token roles, cache-free kwargs, import-safe
description, unbound-model rejection, noncontiguous Hulu image-token rejection,
length-changing visual-compression rejection, and source-file identity for all
four public factories.  Together with the existing runtime integration tests,
the focused result is `22 passed` after adding the true `python -m` regression.
The factory-only suite also passes independently inside the frozen Hulu
environment (`8 passed`), without loading a checkpoint.

The two model runtimes are deliberately isolated even though both interpreter
paths are present on this host.  Huatuo is frozen to its conda interpreter,
PyTorch `2.0.1+cu117`, Transformers `4.37.2`, and the Huatuo source root. Hulu
is frozen to `/home/dbw/.venvs/hulumed/bin/python`, PyTorch `2.4.0+cu118`,
Transformers `4.51.2`, and `PYTHONPATH=/home/dbw/ANCHOR` only.  CPU-only import
preflights resolved every model/input callable back to the factory source file;
the Hulu process does not inherit the Huatuo source root.

The first Huatuo launch exposed a Python module-identity issue before any model
forward: the `-m` runtime lived under `__main__`, while the factory re-imported
the same source canonically, so the provenance dataclass identities differed.
The run failed closed, wrote no artifact, and never loaded Hulu.  The runtime
now binds its canonical `__spec__.name` to `__main__` before defining classes.
A subprocess regression executes the real `-m ... canary` route in both frozen
interpreters and proves that canonical factory imports return the identical
provenance class (`2 passed` in the isolated Hulu run).  GPU relaunch remains
pending and must continue to wait on the shared lock.

The serial launcher passes `bash -n`.  It is inert unless the exact explicit
environment phrase is supplied.  It acquires the existing shared GPU-0 lock in
blocking mode and holds it across Huatuo followed by Hulu, so a future explicit
launch waits for the current unified evaluation rather than competing with it.
It also fails closed if either write-once output already exists.

## Explicit future launch (not executed)

```bash
CECD_EXPLICIT_CANARY_LAUNCH=run-native-eager-canaries-v1 \
  bash /home/dbw/ANCHOR/scripts/run_cecd_system_pih_native_eager_canaries_v1.sh
```

The machine-readable handoff is
`configs/cecd_system_pih_native_eager_canary_handoff_v1.json`.  The existing
system/PIH preflight intentionally remains false.  Passing native/eager
artifacts must be audited and source-bound before changing any readiness flag;
head selection and both interventions remain outside this canary.
