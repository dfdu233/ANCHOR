# CECD product-estimand P0 fail-open closure

**Date:** 2026-08-03  
**Scope:** outcome-blind contract, source, synthetic tests, and CPU verification only. No sealed outcome, human return, admission decision, model output, or scientific result was opened. No GPU/model process was started. Frozen data selections and the `0.25 RE` / `0.03 delta-AUROC` thresholds were not changed.

## Outcome

Both P0 fail-open paths identified in `CECD_PRODUCT_ESTIMAND_AUDIT_20260803.md` are closed.

### 1. Exact science-grid admission

`analyze_cecd_admission_reviews_v1.py` now authorizes only when:

- admitted nonbaseline renders exactly equal `SCIENCE_RENDER_NAMES - BASELINE_VIEW`;
- admitted candidate prompts exactly equal the two nonbaseline `PROMPT_TEMPLATES`;
- the exact identity-render family is present and passes;
- every required render and prompt family passes the unchanged human-review thresholds.

The emitted `science_grid_contract` records the expected names and three exact-closure booleans. `cecd_admission_gate.py` independently rechecks the exact lists and booleans against runner constants before accepting an otherwise valid top-level `passed=True` artifact.

New admission schema:

```text
cecd-human-admission-analysis-v2-exact-science-grid
```

The legacy `cecd-human-admission-analysis-v1` JSON schema is still ordinary readable JSON but is explicitly rejected by every authorizing gate.

### 2. Independent raw-input scientific recomputation

`verify_cecd_three_stage_v3.py` no longer treats stored confirmation booleans as scientific authority. It now:

1. verifies the raw dev and confirmation JSONL paths/hashes against the six model-stage runs;
2. reloads the bound dev JSONL and reruns `fit_dev_stage(folds=5, draws=5000, seed=42)`;
3. compares the complete deterministic dev-fit core against the stored artifact;
4. reloads the bound confirmation JSONL and reruns `apply_confirmation_stage(draws=5000, seed=42)` with the independently rebuilt frozen dev fit;
5. compares the complete deterministic confirmation core against the stored artifact;
6. independently reconstructs, from numeric metrics rather than booleans, pooled delta-AUROC, harmful alignment, interaction RMS, identity ratio, all four reader slopes, four-finding heterogeneity, each model pass, and the two-model conjunction;
7. emits authorization solely from the recomputed decision.

New schemas:

```text
clinical-equivalence-composition-defect-dev-fit-v2-recomputable
clinical-equivalence-composition-defect-confirmation-locked-v2-recomputable
cecd-three-stage-input-gate-v4-independent-recomputation
```

Legacy dev-fit v1, confirmation v1, and input-gate v3 artifacts cannot authorize downstream work. Listing validation and the persistent admission monitor import the new schema constants rather than maintaining independent string copies.

## Adversarial tests

Added/updated tests cover:

- only two admitted nonbaseline render families despite top-level pass intent;
- a missing admitted candidate prompt;
- runtime exact-grid mismatch under asserted `passed=True`;
- legacy admission v1 rejection;
- empty per-model metrics with asserted `both_models_pass=True`;
- a gate component that disagrees with its numeric metrics;
- a tampered stored confirmation that differs from independent recomputation;
- proof that the recomputation path consumes both models' bound raw dev and confirmation inputs with the frozen folds/draws/seed;
- legacy dev-fit v1 rejection.

Verification:

```text
focused affected chain: 80 passed
complete CECD / clinical-equivalence / Treble-collision suite: 240 passed
post-addition verifier suite: 9 passed
Python compilation: passed
git diff --check: passed
```

## Runtime and hash ripple

Changing the persistent clinical monitor source made its pre-existing process stale by design. The old child was terminated, and the same registered PPID-1 job was relaunched from the new source:

```text
supervisor PID 709453
child PID 709456
state running
stage waiting_for_four_independent_returns
labels/attestations synthesized false
sealed mapping exposed before returns locked false
```

The dual-semantics transition monitor imports the clinical monitor's schema
constants, so it was also restarted to avoid retaining v3 expectations in
memory (`supervisor 712639`, `child 712640`, PPID-1, running). It still creates
no GPU/model work and remains waiting on a genuine recomputed CE gate.

No human file was opened during restart. The v2 outcome-blind DAG audit was regenerated after process-identity closure:

```text
status static_handoffs_ready_waiting_genuine_inputs
passed true
blockers 0
fingerprint 6a12bbe7b0decc01d61ef5719f24eb336c96fe39f8a434cccc3cf5b8cf8f9364
artifact SHA-256 d396c3bed779e72f7de2b8281c7bb9f2274bc66bd58fd3ce75b38f6f0ed8b014
```

The analyzer source change also required outcome-blind provenance refreshes:

- `configs/cecd_reader_threshold_alias_sensitivity_v1.json` analyzer binding;
- `corrected_runs/vindr_v2/cecd_stage1_power_audit_v1/power_audit.json` analyzer binding and fingerprint.

The refreshed power audit changed only analyzer SHA-256 and its own fingerprint; selections, power values, thresholds, and scientific decisions did not change.

## Scientific boundary: axis admission is not a human product null

This repair guarantees that every model-scored render and wording belongs to the independently reviewed axis sets. It does **not** claim that humans were tested on the full render-by-wording product behavior. The current clinical review compares render image pairs; the language review compares wording text pairs; neither directly measures a human render-by-wording response interaction.

Accordingly:

- exact-set closure supports “two independently admitted axes”;
- it does not by itself support “human behavior is invariant over their product”;
- a formal model-specific composition interpretation requires an outcome-blind joint human product negative control, or the paper must retain the narrower claim “model nonseparability under independently admitted axes.”

This is a scientific claim guardrail, not a reason to reopen the completed P0 engineering fix.
