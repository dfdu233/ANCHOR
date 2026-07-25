# MedUniEval SGTA audit and corrected experiment protocol

## Decision

The historical top-level experiment scripts are retained as legacy evidence,
but they are not authoritative.  Corrected experiments must run through this
`corrected_sgta` package and write to `corrected_runs/*`.  Historical result
files are not reused unless their protocol fingerprint matches exactly; none
of the inspected historical caches contains the required logits, features,
split identity, and protocol fingerprint.

## Material findings in the historical implementation

- `sgta_confgen.py` references an undefined `score_fn`, leaves every CE path
  and `oe_eval` unimplemented, calibrates on all rows instead of a disjoint
  calibration split, mixes greedy and sampled candidates, and uses only the
  first generated token's maximum vocabulary probability as a sequence score.
- `AboveLambdaSequenceSelector` combined with maximum admissibility does not
  provide the required monotone nested sequence family.  The corrected path
  uses conf-gen's `RunningMaxSequenceSelector`.
- The old open-ended CP path uses ground-truth ROUGE to retain or choose test
  candidates and truncates its resume cache.  Those numbers are oracle upper
  bounds, not deployable model performance.
- Several old multiple-choice parsers split on every comma.  For example,
  MIMIC-CXR qid 543 (`A, Stent; B, ICD; C, Pacemaker; D,VAD`) is misparsed.
- Old binary handling can derive candidate labels from a row's ground truth.
  Non-Yes/No binary rows are now excluded instead of leaking the answer space.
- Token IDs were not robust to leading whitespace and case.  Each semantic
  class now uses the maximum over an equal policy of valid single-token surface
  forms.
- The old FedDG approximation mixed the complete grayscale spectrum.  It is
  not the low-frequency RGB amplitude interpolation in FedDG-ELCFS.
- Per-image style graphs were labelled LAME/LATA even though both upstream
  methods are cross-example, fixed-class transductive methods.
- SCA-T was not implemented.  Its TIM/TIM(KL) objective requires a fixed class
  prototype space and therefore cannot be applied across unrelated MC option
  letters.
- A directory-wide fatal-name scan finds 176 F821 undefined-name findings
  across legacy exploratory scripts, including `model`, `proc`, `tokenizer`,
  `cov_gap`, and `time`, plus explicit placeholder branches.
  These scripts are not called by the corrected runners.

## Corrected data and model protocol

- Protocol version: `medheval-sgta-v5.2`.
- Dataset and configuration SHA-256 fingerprints are stored beside every
  JSONL cache.  A mismatched partial cache is rejected.
- Splits use SHA-256 of `(seed, qid)` and are independent of input order.
- Calibration and test qids are stored in every analysis report.
- Hulu-Med-14B and LLaVA-Med-v1.5-Mistral-7B use the same prompts, finite-label
  policy, image size, style transformations, and metrics.
- LLaVA open-ended generation passes an explicit all-ones attention mask because
  its tokenizer aliases pad and EOS; the mask policy is included in the cache
  fingerprint.
- CE baseline and all CE adaptations reuse the exact same original-image
  constrained logits.  This avoids comparing free generation to candidate
  logits.
- Malformed schemas and missing images are counted; rows are never silently
  repaired with fuzzy answer matching.

## Relationship to the referenced repositories

| Method | Corrected use | Exactness / limitation |
|---|---|---|
| FedDG-ELCFS | Centered low-frequency RGB Fourier-amplitude interpolation | Operator matches the upstream demo. The source amplitude is an external PubMedVision modality center, so the report calls it `feddg_center`, not literal paired-source FedDG training. |
| LAME | Upstream RBF kNN affinity and CCCP/Laplacian optimization | Only fixed Yes/No rows. Deterministic 256-row windows bound memory and are disclosed as an approximation to one full transductive batch. |
| LATA | Sparse union-kNN RBF graph and paper Eq. 4/Eq. 6 refinement | Only fixed Yes/No rows; the same disclosed deterministic windows are used. |
| SCA-T | Upstream 100-step TIM and TIM(KL) objectives followed by split LAC/APS | The generative-VLM extension uses last-prompt hidden states and averaged semantic LM-head token prototypes. Marginals are evaluated in log space to prevent LM-scale underflow; primary accuracy is test-only and the full transductive pool is diagnostic. It is not claimed to reproduce a CLIP source/target experiment. |
| conf-gen | Installed local package, `RunningMaxSequenceSelector`, maximum admissibility aggregation | Only calibration qids enter calibration. Admissibility is a fixed ROUGE-L proxy because MedHEval's published Knowledge hallucination judge imports a proprietary Bedrock client. |
| SGTA | Full per-image RBF graph across aligned image styles and entropy/mean style baselines | This is the proposed method and is kept distinct from LAME/LATA. |
| SGTA-ConfGen | Equal-budget round-robin original/FedDG/gamma candidate stream analyzed by conf-gen | This is accurately reported as style-augmented ConfGen; variable-length generations do not use the finite-label CE graph update. |

## Authoritative commands

```bash
cd /root/autodl-tmp/Hulu-Med/MedUniEval

# CE baseline, FedDG, TTA, SGTA, LAME, LATA, LAC and APS on both models
corrected_sgta/run_full_ce.sh

# SCA-T TIM/TIM(KL), after CE caches exist
corrected_sgta/run_scat.sh

# Four-row-per-dataset generation validation, then full OE matrix
corrected_sgta/run_oe_matrix.sh smoke
corrected_sgta/run_oe_matrix.sh full
```

## Verification

The authoritative package passes fatal/undefined-name lint checks and 21 unit
tests covering malformed MedHEval choices, deterministic disjoint splits,
FedDG/LAME/LATA/conformal primitives, whole-sequence NLL extraction, lexical
admissibility, conf-gen calibration separation, and synthetic SCA-T TIM.
