# CECD admission → Huatuo + Hulu Stage-1 fail-closed audit

Date: 2026-08-03 UTC  
Scope: priority-2 CECD four-role admission through the two-model behavioral Stage 1  
Constraints honored: no sealed outcomes inspected, no scientific threshold changed, no GPU job run

## Verdict

The pre-audit chain was directionally careful but was not fully fail-closed. The initial audit reproduced three authorization-relevant engineering holes:

1. An existing `analysis.json` was trusted when its reviewer-path/hash provenance matched. A modified `passed=true` payload with unchanged provenance could therefore authorize the next stage.
2. The independent two-model input gate accepted a self-declared `3040 rows / 160 complete orbits` manifest without reconstructing the exact 19-cell orbit. Duplicate cells could hide a missing interaction cell while preserving the row count and manifest hash.
3. Huatuo and Hulu each bound an admission, but the join did not require one identical frozen transform/prompt/scientific contract. Stage analysis also compared only the *set* of input hashes, losing path-to-hash identity.

An independent post-audit recheck reproduced two further repository-boundary holes: the factorial runner permitted admission-free model scoring under an engineering label, and the retired Huatuo-only wrapper gated admission without binding it into the produced factorial artifacts.

All five holes are now closed. The chain is fail-closed under the tested admission-omission, legacy-entrypoint, file-drift, missing-cell, wrong-role, wrong-model, contract-drift, stale-artifact and provenance-swap faults. This is an engineering-integrity conclusion only; it is not evidence that CECD passes its clinical or behavioral scientific gates.

## Threat-model results and repairs

| Threat | Pre-audit state | Repair / current invariant |
|---|---|---|
| Four completed sheets and four attestations change between polls | Each file was copied to a hash-named path, but no first-complete-bundle lock joined all eight files | `human_return_bundle_lock.json` is write-once and binds all four role sheets plus all four attestations. A later byte change creates a write-once collision and cannot replace the admitted bundle. |
| Reviewer roles or identities collide | v3 validator already enforced role-specific CSV schemas, role-specific professional attestations, blinded independence and four distinct nonempty reviewer IDs | Runtime authorization now also requires the eight-file bundle to equal the v3 validation records. Exact role isolation remains in the v3 validator; no relaxation was made. |
| Sealed mapping is opened before returns are frozen | Analyzer opened it only after copy, but there was no explicit eight-file ordering invariant | The eight-file bundle lock is written first. Only then may source-pack closure code hash/open `sealed_mapping.json`, followed by deterministic admission analysis. |
| Source pack or existing working artifacts drift | Manifest/sealed mapping existed, but the complete critical pack was not joined in a write-once closure | `pack_source_lock.json` binds the integrity artifact and every critical top-level protocol/review/mapping file. Existing lock/validation/analysis artifacts are revalidated by hash and write-once equality. |
| Existing admission analysis is modified | Existing analysis with matching provenance was loaded and augmented in place | The analyzer is rerun deterministically into a temporary directory on every admission transition. The reconstructed complete payload must be byte-equal to an existing `analysis.json`; otherwise authorization terminates on a write-once collision. |
| Formal runner is called directly without admission | `--admission-result` was optional, so a direct call could load a GPU model while labelling the output engineering-only | Formal execution now fails before model paths, tokenizer metadata, output creation or CUDA are touched. Only explicit `--engineering-render-audit` (legacy alias `--render-audit-only`) may omit admission, and every resulting config explicitly sets model-scoring and scientific-artifact authorization to false. |
| Legacy one-model Huatuo wrapper is invoked | It gated before GPU but failed to propagate the admission into factorial artifacts | The wrapper now exits `64` immediately and directs operators to the admission-bound two-model v2 wrapper. It cannot acquire a GPU lock, load a model or create an artifact. |
| Pixel similarity substitutes for clinical equivalence | Runner comments said pixel guard was engineering-only, but runtime admission did not assert the evidence basis | Admission payload and runtime gate now require the frozen clinical/language human-evidence basis and explicitly prohibit pixel similarity as clinical-admission evidence. Thresholds remain exactly 0.05 change and 0.10 unable. |
| A missing interaction cell is hidden behind `rows=3040` | Independent gate checked only counts, hashes, model and config fingerprint | Gate now groups raw rows by `(image_id, finding)`, rejects duplicate cell IDs, derives the exact 19 cells from `cell_specs`, and checks render, prompt, role, reference cell, prompt-text hash, `status=ok`, 160 complete orbits and 40 orbits per frozen finding. |
| One model is absent or two aliases name one model | Family set was checked but distinct model identity was not explicit | Exactly two gate records, families `{huatuo,hulu}`, and two distinct model IDs are mandatory. Both must be present before analysis. |
| Models use different transforms/prompts/substrates | Each config was internally fingerprinted, but no cross-model scientific join existed | Each run emits a verifier-derived scientific-contract hash over manifest/bbox hashes, split, findings/votes/seed, selection hashes, render and prompt specs, cells, readout and source hashes. Huatuo and Hulu hashes must be identical. |
| Stage analysis swaps path→hash bindings | Only the set of hash values was compared | Provenance paths are canonicalized and the exact path→SHA-256 mapping must equal the two admitted raw runs; duplicate path aliases are rejected. |
| A single passing model authorizes the method stage | Authorization was computed from list length and subset membership | Passing model IDs must be unique, drawn from the two exact expected models, and method authorization is true iff the passing set equals both expected models. Hidden-state authorization remains false. |
| Failure silently retries expensive work | Failed admission/stage was already terminal; wrapper uses `set -e` | Scientific terminal states remain terminal: a valid analyzed `passed=false` admission and a failed detached Stage-1 job are not retried. Malformed, incomplete or drifting inputs instead produce `input_or_transition_error` and are polled again as a recoverable operational condition; this path performs no model scoring and cannot authorize GPU work. |
| Huatuo and Hulu contend for GPU or interleave | Wrapper already acquired one `flock` before all four canary/full calls | Preserved. One `gpu0-vindr-v2.lock` is acquired once before Huatuo canary/full and remains held through Hulu canary/full, input verification and analysis. |

## Fault injection

All injections used temporary directories and synthetic labels/artifacts. No real returned decision, sealed mapping outcome, model score or image was read.

The suite now covers:

- one completed role sheet changing after the first eight-file freeze;
- a critical source-pack review sheet changing after its closure lock;
- an existing failed analysis being changed to `passed=true` while retaining its provenance;
- duplicate raw interaction cells with a simultaneously missing cell while keeping 3040 rows and updating the manifest hash;
- two models reporting different scientific-contract hashes;
- swapped raw input hash values under the correct analysis input paths;
- admission bytes changing after model-run binding;
- missing Stage input gate and a failed detached job remaining terminal;
- admission threshold/evidence-basis drift and stale reviewer-sheet bytes;
- a direct formal runner invocation with no admission, proving rejection occurs before model-path access or output creation;
- an admission-free engineering render-audit contract, proving all scientific/model-scoring authorization flags remain false;
- invocation of the retired Huatuo-only wrapper, proving it exits before any command or artifact path is reached.

Regression command:

```bash
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  $(rg --files tests | rg -i 'cecd|clinical_equivalence' | tr '\n' ' ')
```

Result: **44 passed in 12.11 seconds**.

Python compilation also passed for the monitor, factorial runner, admission analyzer/gate and two-model verifier; both active and retired shell wrappers passed `bash -n`.

## Live-state audit and operational action

At the final read-only check, detached supervisor PID `499758` and monitor child PID `499760` were alive. The heartbeat was current and safely at:

```text
waiting_for_four_independent_returns
```

All eight expected returned files were still absent. Therefore no admission analysis, Stage-1 GPU launch, or sealed-outcome access occurred during this audit.

The running child imported the pre-repair Python code at 02:57 UTC, whereas the monitor source changed afterward. It **must be cleanly restarted before any return files are delivered** so that the first complete bundle is processed by the repaired fail-closed code. Restarting now is safe because the monitor is only waiting and no human-return lock or Stage-1 state exists.

### Post-audit operational closure (03:18 UTC)

The required restart was completed before any return file arrived. The repaired
monitor is now running as supervisor/child PIDs `513549/513552`; its fresh
heartbeat remains `waiting_for_four_independent_returns`, lists all eight files
as absent, and records all three safety flags as false. The research watchdog
reports the job alive, and GPU utilization remains zero. Consequently the
restart requirement above is closed; the scientific admission gate remains
open and no behavioral result is implied.

## Scientific boundary

This audit deliberately did not assess whether the four transforms are clinically equivalent, whether CECD exists in either model, or whether the downstream effect reaches the preregistered threshold. Pixel checks remain engineering corruption guards only. Clinical equivalence is authorized exclusively by the frozen four-role human protocol, and method-level work remains forbidden unless both distinct models pass the later behavioral gate.
