# Style-Prior Template Invariance Gate

This mechanism-only pilot asks whether the residual six-disease response to
six fixed PubMedVision-CXR style transforms survives semantically equivalent
question and answer wording. It uses 16 exposed MIMIC development images
(16 patients), HuatuoGPT-Vision, complete-sentence teacher forcing, and no
target labels. The likelihood contrasts are diagnostics, not predictions.

## Frozen comparison

- `original`: “shows” versus “does not show” (existing probe).
- `demonstrates`: “demonstrates” versus “demonstrates no”.
- `evidence`: “radiographic evidence ... is present” versus “... is absent”.

The `evidence` frame has equal positive/negative token counts for every
disease, providing a direct sensitivity check for the original two-token
negation-length difference. Every template uses exactly the same patients,
images, style transforms, diseases, radius (`0.12`), and strength (`0.65`).
Before comparison, the analysis removes the joint span of the patient-specific
clean-centroid direction and the all-disease uniform axis, then case-centers
the six style responses.

## Result

| Pair | matched style cosine | patient-bootstrap 95% CI | held-patient style ID | permutation p |
|---|---:|---:|---:|---:|
| original → demonstrates | 0.570 | [0.014, 0.746] | 17.7% | 0.379 |
| original → evidence | 0.456 | [0.051, 0.619] | 21.9% | 0.058 |

Chance style identification is 16.7%. The pre-registered gate required both
matched cosines to exceed `0.50`, both bootstrap lower bounds to exceed zero,
and both held-patient permutation p-values to be at most `0.05`. The gate
failed.

The positive matched cosines, including the equal-length `evidence` frame,
show that the earlier signal is not entirely a fixed negation/token-length
artifact. However, the negative identity-assignment margins and failed
held-patient style identification show that it is not a stable, uniquely
indexed style-to-clinical-prior map. The result supports, at most, a weak
template-robust susceptibility direction. It does not establish prior
switching and does not justify a decoder.

### Frozen-reference sensitivity

After the gate failed, an external audit requested a stricter post-hoc
sensitivity statistic. Style directions were frozen from all 64 old
`original`-template cases and evaluated as the patient-balanced matched-style
cosine minus the mean mismatched-style cosine:

| Target interface | frozen-reference margin | patient-bootstrap 95% CI | blocked-permutation p |
|---|---:|---:|---:|
| original (paired 16) | 0.142 | [0.045, 0.242] | 0.022 |
| demonstrates | 0.161 | [0.025, 0.292] | 0.003 |
| evidence (equal length) | 0.098 | [-0.006, 0.195] | 0.054 |

The original effect reproduces, and `demonstrates` transfers, but the
equal-length frame remains borderline. Paired non-inferiority intervals for
retaining 50% of the original margin cross zero for both alternatives. This
post-hoc analysis cannot reverse the failed pre-registered gate.

## Reproduction

```bash
PYTHONPATH=. python -m anchor.corrected_sgta.run_style_prior_template_probe \
  --model /autodl-fs/data/data/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL-fast \
  --questions data/mimic_cxr_rule/questions.target.jsonl \
  --image-manifest data/mimic_cxr_rule/image_manifest.jsonl \
  --style-manifest \
    corrected_runs/pubmed_style_lineage_probe_v1/prototypes/manifest.jsonl \
  --output corrected_runs/style_prior_template_probe_v1/huatuo_n16.raw.jsonl \
  --limit 16 --batch-size 16 --radius 0.12 --strength 0.65

PYTHONPATH=. python -m \
  anchor.corrected_sgta.analyze_style_prior_template_probe \
  --reference corrected_runs/visual_evidence_chord_probe_v1_n64/raw.jsonl \
  --probe corrected_runs/style_prior_template_probe_v1/huatuo_n16.raw.jsonl \
  --output corrected_runs/style_prior_template_probe_v1/huatuo_n16.summary.json \
  --figure corrected_runs/style_prior_template_probe_v1/huatuo_n16.png
```

The uncommitted raw diagnostic is 5.5 MiB and is bound in `summary.json` by
SHA256. The repository stores the compact summary, figure, code, and tests.
