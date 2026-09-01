# CECD Scientific Worker Part 3：formal CE shared-cache stage

**Date:** 2026-08-03  
**Verdict:** the seven implemented factorial controls are now connected to a
formal, hash-bound CE-only runner. The stage remains fail-closed for CECD
hidden-state intervention, both Treble semantics and OE. No formal run was
launched because the required write-once confirmation authorization and
preflight artifacts do not exist.

## Scientific scope

This stage evaluates only fixed-claim, single-next-token CE in the quotient
space of centered `supported/refuted/undetermined` logits. Its seven methods
are:

```text
unmitigated
full_orbit
render_only
prompt_only
random_norm
sign_permuted
main_effect_removal
```

They are factorial/logit-space controls. They are not hidden-state
interventions and are not represented as paper-native baselines. In
particular, these remain blocked before model or GPU access:

```text
cecd_interaction_projection
treble_proceedings
treble_released
OE generation/evaluation
```

The runner additionally requires the confirmation authorization fields
`locked_test_behavioral_increment_confirmed=true`,
`full_method_gate_authorized=false`,
`oral_baseline_closure_authorized=false`, and
`official_compatible_dynamic_activation_baseline_present=false`. Therefore an
older pilot authorization cannot silently open this path.

## Shared four-cell raw-logit cache

For each model, the adapter freezes one ordered CE input contract and computes
the two-render x two-prompt cells exactly once:

```text
shared_ce_cache/<model>/records/<record_key>/cells/h00.json
shared_ce_cache/<model>/records/<record_key>/cells/h10.json
shared_ce_cache/<model>/records/<record_key>/cells/h01.json
shared_ce_cache/<model>/records/<record_key>/cells/h11.json
```

Thus `N` records require `4N` model scores per model, not `7 * 4N` scores. All
seven derived methods must bind to the same absolute
`raw_cache_manifest.json` and SHA-256 within a model. A manifest pointing
outside its frozen per-model cache is rejected.

Each raw cell is atomic, finite, exactly tri-state and labeled
`formal_ce_raw_logit_cache_only`. Corrupt or missing cells are recomputed; a
cache-config drift or a changed completed cache manifest aborts. Model loading
is skipped when all cells validate.

## Atomic method shards and incomplete-run semantics

Each model x method result is first written under `partial_ce/`, validated,
and atomically promoted to:

```text
ce_arms/<model>/<method>/ce_rows.jsonl
ce_arms/<model>/<method>/ce_completion.json
```

Completion binds the run, model, method, runtime descriptor, worker hash, raw
cache hash, CE output hash, row count and cluster count. Resume skips only a
fully valid shard. A worker failure records one failed-stop artifact and needs
explicit audited resume.

The CE-only stage produces `ce_stage_manifest.json`; it deliberately never
produces the full-comparison `run_manifest.json`. Its manifest permanently
states:

```text
oe_implemented = false
hidden_intervention_implemented = false
paper_native_treble_claimed = false
results_interpreted = false
paper_claim_authorized = false
```

## End-to-end validation

The fake-scorer test freezes 30 image-disjoint synthetic clusters and invokes
all seven controls. It observes one scorer construction, exactly `30 * 4 =
120` raw scores, one shared raw-cache hash, seven separate CE shards, and zero
extra scorer/orbit calls on replay.

The controlled-runner test executes the two-model closure: 14 atomic CE
shards, two raw-cache manifests, one cache hash per model and no
`run_manifest.json`. A second execution preserves all completion mtimes.

```text
35 passed in 2.82s
```

## Real engineering smoke after GPU release

After confirming the shared nonblocking flock was available, Part 2's same
single VinDr claim was run on Huatuo and Hulu. This was not a formal CE-stage
execution:

```text
claim = aortic_enlargement__d925309691e7929d905eaa42f081833f__badf6249e030
cells/model = 4
Huatuo status = engineering_smoke_complete
Hulu status = engineering_smoke_complete
formal_method_output_authorized = false
paper_claim_authorized = false
```

Artifact summaries:

```text
Huatuo summary fingerprint = bbc87c0cf697c73263edadc047b30ed72e0f9a4a23b44433e1f1ed5f1971ba40
Hulu summary fingerprint   = d01a51fec779eb22f8ab8257469c7634db04403cb509286adba2188dc67438c3
```

All seven controls retained the `refuted` prediction on this one claim for
both models. This confirms real model loading, four-cell scoring, derivation
and atomic persistence only; it is neither evidence of mitigation success nor
a mechanism result.

## Current launch state

At the end of this part, the expected authorization, formal preflight and
three-stage confirmation input gate are absent. Therefore the formal CE runner
correctly remains unlaunched. CECD hidden intervention, both Treble variants
and OE remain fail-closed rather than being approximated by the seven
logit-space controls.
