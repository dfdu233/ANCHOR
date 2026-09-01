# Paper Scope Gate — 2026-08-02

## 1. First impression

- **Paper type:** New Problem/Setting, if it eventually becomes a paper.
- **One-sentence story under audit:** Reader disagreement provides a clinical
  calibration target for determining which apparent layerwise medical-VLM
  uncertainty mechanisms are real, stable, or non-decodable.

This sentence is a research goal, not a result currently supported at ICLR
paper scope.

## 2. Fatal-flaws audit

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | The positive early-erasure/redundant-evidence mechanism is data-refuted: the Huatuo reader-residual dev gate fails its frozen AUROC/Brier/random-control criteria; localized evidence survival has the opposite reader-unanimity sign; the DICOM-render pilot has 0/4 findings; and the presupposition branch is ineligible under its frozen length match. | **CRITICAL** | None for this mechanism. It is permanently excluded from the paper claim and may not be rescued by a new threshold, layer, finding, or post-hoc model. |
| 2 | A mechanism-boundary paper requires the same formal reader-residual boundary in at least two model families, but only Huatuo reached that exact frozen gate; Hulu was correctly not used as a post-hoc rescue. | **MAJOR** | A future boundary study needs a newly preregistered, architecture-neutral protocol and two primary models from the outset. It is not an authorized continuation of the failed Huatuo hypothesis. |

Closest-work pressure also remains high: PSF-Med already studies medical
paraphrase features, MedVIGIL studies evidence-conditional medical-VLM failure,
and prompt-induced hallucination work already localizes language-over-vision
circuits. A paper cannot claim novelty from generic prompt sensitivity,
early-layer decoding, or image/text intervention alone.

## 7. Verdict

**Reject and Pivot.**

Do not draft a paper skeleton for early erasure, reader-evidence redundancy, a
shared source-domain center, or Clinical Presupposition Amplification from the
current artifacts. A polished narrative would not repair their scientific
gates.

The only current paper candidates are independent conditional pivots:

1. Complete real two-physician adjudication for Specificity Ratchet; proceed
   only if supported-parent/unsupported-child edges and spontaneous OE
   trajectories are reproducible.
2. Complete the CECD clinical-equivalence admission; proceed only if image and
   language operations are independently admitted before model scoring.
3. Use the native OE and mitigation review packs to establish the clinical
   failure substrate and detect length, omission, refusal, or hedging exchange.
   If these human gates fail, retain the project as an audited negative result
   rather than manufacturing an ICLR claim.

The technical-paper skeleton remains intentionally unwritten until one pivot
passes its construct-validity gate and has a decisive experiment capable of
supporting its central claim.
