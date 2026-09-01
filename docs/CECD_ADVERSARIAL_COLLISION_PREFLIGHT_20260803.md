# CECD adversarial collision preflight

**Date:** 2026-08-03

**Audit boundary:** outcome-blind source/code review only. No human return,
sealed model result, method output, or GPU execution was consumed. The
three-stage CECD statistical thresholds are unchanged.

## Verdict

The narrow CECD mechanism question remains conditionally alive:

> Does a clinically admitted render-by-wording interaction add locked,
> reader-grounded error information beyond marginal and generic instability
> controls in two models?

The current **mitigation novelty is not defensible**. Minimal evidence-bounded
editing is occupied by CEBC; source diagnosis and source-targeted/adaptive
steering are occupied by HalluTrace and VLI; prompt-copy heads and system-token
attention provide simpler causal alternatives; HALP occupies generic latent
hallucination prediction; ConRad occupies calibrated clinical confidence.
Until all executable controls below are frozen, the correct endpoint is a
mechanism paper, not a patched mitigation paper. If locked CECD confirmation
fails, CECD terminates rather than changing thresholds or being repackaged as
a method.

## MUST_FIX / ALREADY_COVERED matrix

| Collision | What the closest work already establishes | Already covered | Must fix before a mitigation claim | Verdict |
|---|---|---|---|---|
| HalluCXR length/omission | Response length can itself predict hallucination; fabrication reduction can exchange against omission ([arXiv:2605.20469](https://arxiv.org/abs/2605.20469)). | CE uses exactly one output token. The current envelope records mean claims, length, coverage, omission and refusal, fixes mean claim count, and imposes coverage/length tolerances. | Enforce fixed `K` per record, one-for-one CECD exchanges, claim-identity insertion/deletion logs, length-stratified and length-only risk analyses, negative/hedge rates, and paired omission non-inferiority. Aggregate means can hide casewise deletion. | **MUST_FIX—partial coverage only** |
| System-mediated yes-bias | Causal redistribution away from redundant system attention suppresses yes-bias ([Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1940/)). | Reader-balanced vote bins, signed reader orientation and no-image-independent claim language prevent raw yes-rate from defining truth. | Add a faithful system-attention redistribution arm for both models; report balanced yes-rate, no-image behavior and system/image/text attention mass. Add its score before CECD in the incremental ladder. | **MUST_FIX** |
| Prompt-Induced Hallucination heads | A small, model-specific set of heads mediates prompt copying; ablation reduces PIH by at least 40% in the authors' setting ([ACL 2026](https://aclanthology.org/2026.acl-long.1941/)). | CECD explicitly controls prompt main effects, prompt length and duplicate-prompt noise; it does not claim generic prompt sensitivity. | Identify heads on dev separately per model, freeze them before locked test, run causal ablation, and add a prompt-copy-head score before the CECD residual. A shared universal head set is forbidden. | **MUST_FIX** |
| HALP | Hallucination risk is pre-generatively decodable, but the best layer/modality varies by architecture ([EACL 2026](https://aclanthology.org/2026.eacl-long.287/)). | V3 makes no universal early/late-layer claim and keeps hidden-state authorization false. | Fit visual, decoder-vision-token and query-token probes on dev for each model; freeze the winning coordinate per model; test CECD increment after HALP risk. Never select a layer on confirmation. | **MUST_FIX before latent-mechanism attribution** |
| CEBC | Training-free conformal evidence bounds and minimally edits/suppresses unsupported mentions using an external detector ([ACL 2026](https://aclanthology.org/2026.acl-long.2142/)). | Independent reader evidence and fixed-ontology medical claims differ from generic detector evidence; current claim boundary already says minimal editing alone is not novel. | Compare both a paper-faithful CEBC length/quality Pareto arm and a fixed-`K` common-protocol arm. CECD must improve precision without deletion or recall loss. | **MUST_FIX; standalone RCCP/CBD novelty is dead** |
| HalluTrace | Component ablations distinguish visual grounding failure, language-prior dominance and cross-modal conflict, then select source-targeted decoding ([ALVR 2026](https://aclanthology.org/2026.alvr-main.29/)). | CECD's proposed construct is a second-order admitted nuisance interaction, not a generic perception/prior taxonomy. The current Treble envelope includes static source-oriented controls. | Add vision-encoder/projector/decoder component ablations and a source-targeted HAD-style control. Demonstrate that CECD interaction adds information after source attribution. | **MUST_FIX for source-mechanism or decoding claims** |
| VLI | Instance-specific conflict diagnosis, visual-anchor localization and bi-causal steering already occupy adaptive latent mitigation ([ACL 2026](https://aclanthology.org/2026.acl-long.1784/)). | The current contract explicitly marks the ten-arm static Treble envelope as insufficient for full method closure. | Add an instance-specific conflict-conditioned steering comparator for both models, with dev-only parameter selection and matched compute/coverage reporting. | **MUST_FIX for mitigation novelty** |
| ConRad | Report- and sentence-level verbalized confidence are trained with a proper logarithmic scoring reward ([arXiv:2603.29492](https://arxiv.org/abs/2603.29492)). | Reader-distribution Brier is already required; entropy/self-confidence never define visual evidence; confidence-calibration novelty is disclaimed. | Add reader-distribution NLL and a proper-log-score confidence-only control that cannot edit content. A report-extension claim additionally requires a faithful ConRad comparison or must explicitly exclude calibrated-report novelty. | **MUST_FIX—claim boundary covered, baseline absent** |

## Executable fail-closed preflight

The new validator is
`anchor/corrected_sgta/validate_cecd_adversarial_method_preflight_v1.py`; its
outcome-blind plan is
`configs/cecd_adversarial_method_preflight_v1.json`.

It freezes all of the following before any full-method output exists:

1. the eight primary-source collision identities and occupied-claim
   disclaimers;
2. recordwise fixed-`K`, one-for-one exchange, claim-identity, length,
   omission, refusal, negative and hedge controls;
3. a strongest-baseline ladder ending in
   `system attention + prompt-copy heads + HALP risk`, with CECD added only
   afterward;
4. CEBC native and fixed-`K` comparisons, source-targeted decoding and
   instance-specific adaptive steering controls;
5. proper-score confidence-only comparison;
6. exact Huatuo and Hulu readiness, dev-only selection, untouched locked test,
   source-file hashes, immutable input bindings and an empty output root.

Run it without GPU access:

```bash
cd /home/dbw/ANCHOR
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.validate_cecd_adversarial_method_preflight_v1 \
  --plan configs/cecd_adversarial_method_preflight_v1.json \
  --output corrected_runs/vindr_v2/cecd_adversarial_preflight_v1/audit.json
```

Current truthful result:

```text
status=blocked_mechanism_paper_scope_only
passed=false
full_method_execution_ready=false
mitigation_novelty_authorized=false
paper_claim_authorized=false
three_stage_thresholds_modified=false
```

All eight controls and six immutable bindings are currently unresolved. This
is intentional: the contract cannot be self-satisfied by prose, a surrogate
name, one supported model, an output-dependent layer choice, or an already
nonempty result directory. Even a future passing preflight authorizes only
outcome-blind execution; mitigation novelty and a paper claim remain false
until an independent locked-output validator passes.

## Minimal implementation order

1. **Evaluation first:** unify recordwise claim identity, fixed-`K`, length and
   omission inference. Without this, every OE mitigation number is
   uninterpretable.
2. **Simple causal alternatives:** implement system-attention redistribution,
   model-specific prompt-copy head ablation and HALP probes on dev. If these
   absorb the CECD increment, stop the mitigation branch.
3. **Closest methods:** add CEBC native/fixed-`K`, HalluTrace/HAD and VLI-style
   controls with explicit fidelity labels. Unsupported ports block method
   novelty; they do not justify a convenient local surrogate.
4. **Calibration boundary:** add proper-score confidence-only control and NLL;
   run ConRad only for a report-generation extension.

This order follows the mechanism-research principle that a method is warranted
only if the admitted interaction survives simpler causal explanations and
naturally implies an intervention. It prevents engineering a large mitigation
stack before knowing whether CECD is a distinct mechanism.
