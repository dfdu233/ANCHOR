# C74 — A specialist state is more useful than a specialist vote

## Frozen evidence

The CPU-only audit reuses the official TorchXRayVision DenseNet-121 logits and
the frozen, image-disjoint VinDr development/confirmation manifests.  Each
model has 280 development and 840 confirmation claims over seven findings.
No baseline/GPU process is read or modified.

| model | VLM | + one claim-specific XRV score | + full 18D XRV state with finding-conditioned linear interactions |
|---|---:|---:|---:|
| Huatuo | .7667 | .8264 | **.8599** |
| Hulu | .8606 | .8708 | **.8873** |

The full-state gains over the VLM are `+.0934 [+.0620,+.1251]` for Huatuo and
`+.0265 [+.0033,+.0496]` for Hulu under 2,000 image-cluster bootstrap draws.

The 18D upper bound lowers both error types relative to the VLM operating
point: Huatuo FP/FN `124/123 -> 77/104`; Hulu `75/98 -> 60/92`.  This is a
supervised development-only upper bound, not yet a training-free method.

The scalar expert helps Huatuo mainly where the VLM is weak.  In the lowest
absolute-margin quartile it changes accuracy by `+14.3pp`; the other quartiles
change by `+4.7pp`, `+2.0pp`, and `0pp`.  It repairs 87 Huatuo errors while
creating 45.  On Hulu it repairs 16 but creates 20; its lowest-confidence
quartile changes by `-1.87pp` and the other three quartiles have no binary
decision changes.

The reader continuum supports the same explanation, with the explicit caveat
that intermediate-vote claims are opportunistic overlaps in the XRV cache.
For Huatuo, reader-support Spearman is `.425` for the VLM and `.531` for XRV;
adding XRV lowers exploratory support MSE by `.0221`.  For Hulu the VLM is
already stronger (`.595` versus `.531`) and the MSE gain is only `.0038`.

## What the stronger controls reject

The earlier label-neighbour boundary is not a distinct geometric mechanism.
After the finding-conditioned 18D linear model, adding the neighbour score
**reduces** macro AUROC:

| model | 18D interaction | + neighbour boundary | bootstrap delta (95% CI) |
|---|---:|---:|---:|
| Huatuo | .8599 | .8496 | `-.0102 [-.0176,-.0027]` |
| Hulu | .8873 | .8806 | `-.0065 [-.0123,-.0012]` |

Thus the kNN effect was a lossy way of exposing the full specialist disease
state.  It should not be promoted as a method.

Direct secant stitching is also unsupported.  On 746 joinable images, an
image-disjoint linear map from XRV points predicts held-out VLM positive-minus-
negative feature differences better than a ridge trained directly on those
differences.  For projected post-mean features, variance explained is
`.133` versus `.097` on Huatuo and `.319` versus `.289` on Hulu.  This ordering
holds in every leave-one-finding-out split.  The raw-cache fresh confirmation
images have no XRV cache overlap, so this result is exploratory rather than a
frozen confirmation.

## Mechanism implied by the asymmetry

XRV is complementary to Huatuo but largely redundant with Hulu.  Consistently,
XRV geometry is much more linearly aligned with Hulu's pooled visual state
(`31.9%` held-out secant variance explained) than Huatuo's (`13.3%`).  Expert
accuracy alone therefore cannot determine whether collaboration helps; the
relevant quantity is the expert's **innovation relative to what the VLM already
represents**.

This motivates one new, still-unverified operator.  Treat the specialist as an
observer: learn an unlabeled observation map `H` from pooled VLM state `h` to
specialist state `s`, then apply only the innovation

`h+ = h + H^dagger (s - Hh)`.

This update is zero when the two models agree, is idempotent under repeated use
of the same specialist observation, and is the minimum-norm state correction
that makes the VLM state consistent with that observation.  These are useful
mathematical properties, but collision review and a CPU innovation-residual
gate are required before any GPU intervention.

## Artifacts

* `anchor/corrected_sgta/analyze_xrv_specialist_error_geometry_v2.py`
* `corrected_runs/daylong_idea_search_v1/xrv_specialist_error_geometry_v2/result.json`
* `corrected_runs/daylong_idea_search_v1/xrv_specialist_error_geometry_v2/interaction_vs_vlm_bootstrap.json`
* `anchor/corrected_sgta/analyze_xrv_vlm_secant_stitching_v1.py`
* `corrected_runs/daylong_idea_search_v1/xrv_vlm_secant_stitching_v1/result.json`
