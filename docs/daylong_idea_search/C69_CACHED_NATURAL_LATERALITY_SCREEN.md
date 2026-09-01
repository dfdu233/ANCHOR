# C69b — Natural report laterality cache screen

Date: 2026-08-13.  CPU/cache only; no GPU or baseline job was touched.

## Verdict

**NO-GO for a universal left/right swap on natural reports.**  This does not
invalidate the separately queued controlled screen-frame canary; it shows that
the dramatic controlled Huatuo display-frame error is not a global rule that
can be applied to ordinary report outputs.

We conservatively extracted only positive finding-side pairs for which both a
generated report and its reference contained one unambiguous `left` or `right`.
Across Huatuo, Hulu, LLaVA-Med and Qwen on IU-Xray and MIMIC-CXR, 176 pairs were
matched. Native side accuracy was `.6420`; blindly swapping every side gave
`.3580`, a delta of `-.2841` with bootstrap 95% CI `[-.4205,-.1364]`.

The two largest medical-model cells were especially incompatible with a global
swap:

| cell | matched | native | swapped | swapped - native |
|---|---:|---:|---:|---:|
| Hulu / MIMIC-CXR | 46 | .7826 | .2174 | -.5652 |
| Huatuo / MIMIC-CXR | 20 | .7500 | .2500 | -.5000 |

LLaVA-Med / MIMIC-CXR had a positive point delta (`+.1020`, `n=49`), but its CI
`[-.1837,+.3878]` crossed zero and no second model agreed. The preregistered
cache gate therefore failed.

Boundary: report references are lexical benchmark proxies, not physician truth.
This audit is sufficient to reject an indiscriminate output compiler, not to
estimate clinical laterality accuracy.

Artifacts:

- `anchor/corrected_sgta/analyze_cached_natural_laterality_v1.py`
- `tests/test_cached_natural_laterality_v1.py` (`3 passed`)
- `corrected_runs/daylong_idea_search_v1/cached_natural_laterality_v1/result.json`

