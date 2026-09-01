# L1 — Criterion-Shift Mirage

## Question

When the generated answers are held fixed, can changing only the benchmark's
accepted-answer rule reverse which hallucination-mitigation method appears
better?  If yes, the reported method gain is partly an evaluation operating
point, not necessarily a clinical improvement.

The audit reads the same complete LLaVA CXR-VisHal outputs (`5,587` questions,
`475` image clusters) for DoLa, VCD, OPERA, PAI, and VISTA.  It does not
regenerate or reparse answers.  Paired confidence intervals resample image
clusters 5,000 times.

## Result

- Strict ranking: `OPERA > VISTA > PAI > DoLa > VCD`.
- Official-proxy ranking: `OPERA > VCD > PAI > DoLa > VISTA`.
- Five of ten method pairs reverse order between strict and official proxy;
  six of ten reverse between strict and parseable-only scoring.
- VCD minus VISTA is `-4.58 pp` under strict scoring, 95% CI
  `[-5.64, -3.47] pp`, but `+4.98 pp` under official proxy, 95% CI
  `[+3.82, +6.23] pp`.

The images, model generations, and methods are identical; only the scoring
criterion changes.  The sign reversal is therefore a real evaluation effect.

## Boundary

This does not identify which criterion is clinically correct and is not a
mitigation algorithm.  It establishes a mandatory falsification rule: a new
method whose conclusion reverses across predeclared clinical/official parsing
criteria cannot be claimed to reduce hallucination.

Artifact:
`corrected_runs/daylong_idea_search_v1/criterion_shift_llava_cxr_vishal_v1.json`.

