# ICLR paper logic skeleton: reader-calibrated mechanism boundaries

> **SUPERSEDED — DO NOT DRAFT OR CITE AS A CURRENT PAPER PLAN.**  The
> preregistered Huatuo reader-residual development gate subsequently failed
> its AUROC/Brier/random-projection criteria, so confirmation was correctly
> stopped before any held-out feature collection.  The later and authoritative
> scope decision is `Reject and Pivot` in
> `docs/PAPER_SCOPE_GATE_20260802.md`.  This file is retained only as
> pre-outcome design provenance; its references to “pending confirmation” are
> historical, not an unfinished positive claim.

Status: frozen working skeleton, not a claim that the pending VinDr confirmation has succeeded.

## 1. Paper-type positioning

- Type: **New Problem/Setting paper**.
- Rationale: the default contribution is not a decoder that beats a leaderboard; it is the reader-calibrated formulation and causal/statistical boundary of when layerwise medical-VLM evidence is actually usable.
- Conditional conversion rule: convert to a Technique paper only if held-out Early erasure passes in both primary models, survives representation controls and a preregistered causal patch, and a subsequent method improves OE at matched claim coverage without extra omission.

## 2. Thinking template

| Stage | Working content |
|---|---|
| Research background | Medical VLMs make image-grounded clinical claims whose polarity and certainty can affect downstream decisions. VCD (CVPR 2024), SECOND (ICML 2025), OPERA, and recent medical interventions such as VIHD/VLI motivate layerwise or contrastive decoding, while HalluCXR-style analyses expose length and evaluation confounding. Existing gains therefore do not establish that an early layer contains reader-calibrated evidence that the final layer erased. |
| Limitation 1 | Existing mitigation evaluations often conflate fabrication reduction with shorter output, abstention, uniform hedging, or omission, and frequently use no multi-reader reference for weak visual evidence. |
| Limitation 2 | Early-layer decoding methods select or combine layers based on model confidence or benchmark outcome, but do not test the prerequisite claim that adjacent reader-disagreement states are more decodable early than finally after polarity and finding are controlled. |
| Limitation 3 | Binary CE, open-ended VQA, and reports are scored with incompatible parsers and units, so apparent cross-task or cross-model improvements can be evaluator artifacts rather than a shared medical-hallucination mechanism. |
| Key Idea / Our Goal | **Define medical-VLM mitigation eligibility by a held-out reader-calibrated layer-boundary test: locate whether claim clarity is erased early, emerges late, remains equivalent, or is not additionally decodable before permitting any intervention.** |
| Challenge 1 | Reader disagreement is entangled with finding prevalence and positive/negative polarity; a naive probe can decode these nuisances and falsely call them clarity. |
| Challenge 2 | Layer/family/probe-capacity selection can consume the test set, and heterogeneous model frontends can make a cross-model result an implementation mismatch. |
| Challenge 3 | Even a positive observational probe can reflect confidence, entropy, random high-dimensional directions, output shortening, or retrieval priors rather than a causal image-grounded mechanism. |
| Methodology topic sentence | The **Reader-Calibrated Layer Boundary Protocol** separates nuisance-controlled measurement, pre-confirmation specification, and fail-closed causal/method authorization. |
| Module A (addresses Challenge 1) | **Adjacent-vote residual measurement:** within 0/3 versus 1/3 and 2/3 versus 3/3, compare matched-capacity early/final probes only after flexible signed-evidence calibration, polarity strata, and finding fixed effects. |
| Module B (addresses Challenge 2) | **Trace-certified confirmation:** use image-disjoint VinDr splits, dev-only layer/family/PCA selection, immutable direction locks, exact model/frontend identity gates, and confirmation-only clustered inference across Huatuo and Hulu. |
| Module C (addresses Challenge 3) | **Fail-closed mechanism and utility controls:** compare random projections, direct Maybe/commitment, confidence, entropy, null and causal controls; separately audit OE length, claim coverage, omission, RAG relevance, and parser provenance before any mitigation claim. |
| Contribution 1 | A reader-relative formulation and four-state layer-boundary taxonomy for image-grounded clinical claims, with non-significance separated from equivalence and non-decodability (Sections 2–3). |
| Contribution 2 | A trace-certified, multi-reader, two-model VinDr protocol with frozen dev-to-confirmation specifications and image-cluster uncertainty (Sections 3–4). |
| Contribution 3 | A cross-model/finding boundary map plus strong confidence/random/retrieval controls; the exact positive or negative conclusion remains gated on pending confirmation (Section 4). |
| Contribution 4 | Evidence that common apparent mitigation gains can be caused by parser errors, irrelevant retrieval, length/coverage exchange, or backend mismatch, with an executable unified evaluation contract (Section 5). |

## 3. Self-consistency checks

- Check 1, Limitations → Goal: **pass**. The goal directly replaces uncalibrated mitigation with a reader-calibrated eligibility test and a shared claim unit.
- Check 2, Goal → Challenges: **pass**. Nuisance leakage, selection/frontend leakage, and non-causal proxy gains are the three obstacles to making that eligibility test credible.
- Check 3, Challenges → Methodology: **pass**. Modules A, B, and C map one-to-one to Challenges 1, 2, and 3.
- Check 4, Methodology → Contributions: **pass, conditional wording enforced**. Contributions 1–2 are protocol contributions; Contribution 3 cannot be finalized before confirmation; Contribution 4 is supported only by provenance-complete audits.

Severity: **0 critical, 1 major, 0 minor**. The major gap is empirical, not narrative: two-model confirmation and, only if indicated, causal patching remain unfinished.

## 4. Methodology outline

### 3.1 Reader-relative claim clarity

Define atomic image-grounded claims, adjacent reader-vote directions, polarity/clarity coordinates, and the four mutually exclusive boundary states. Explicitly separate `Indeterminate` from `Not decodable` and require equivalence testing for `Layer-stable`.

### 3.2 Nuisance-controlled representation measurement

Fit a flexible scalar-evidence nuisance model with polarity interaction and finding effects; add matched-capacity residual representation probes. Freeze family, non-final layer, and PCA capacity on image-grouped dev folds only.

### 3.3 Held-out boundary confirmation and controls

Apply the locked model to image-disjoint confirmation without refitting. Report pooled direction and finding-wise image-cluster bootstrap intervals, random-direction and direct logit controls, and cross-model consistency with indeterminate cells retained in the denominator.

### 3.4 Conditional causal and intervention path

Observational Early erasure can request an activation-patching experiment but cannot authorize a decoder. A decoder exists in the paper only after causal confirmation and then must pass matched-coverage OE/report utility gates.

## 5. Scope firewall

The paired DICOM-render experiment is an independent exploratory branch. It found heterogeneous per-image sensitivity but failed its held-out signed progression gate for all four findings, so it does **not** establish a training-source-domain center and does not enter the main paper methodology. RAG is a baseline/control family, not a module of the proposed mechanism.

No Introduction prose should be drafted until the pending two-model confirmation determines the exact wording of Contribution 3.
