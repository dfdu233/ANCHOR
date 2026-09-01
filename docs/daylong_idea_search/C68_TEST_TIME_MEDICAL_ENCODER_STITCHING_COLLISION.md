# C68 — Test-time Medical Encoder Stitching

Date: 2026-08-13  
Decision: **formula-level NO-GO**

## Candidate

For the same medical image, use patch correspondence between a specialist
encoder `S in R^(n x d_s)` and the VLM visual stream `V in R^(n x d_v)`. Solve
an orthogonal/ridge alignment at inference and inject aligned specialist tokens
instead of expert labels or probabilities.

## Fatal identifiability problem

If the alignment is fit independently on the current image by minimizing
`||SA-V||`, it is trained to reconstruct information already present in `V`.
When dimensions permit interpolation, `SA=V` on the fitted patches and nothing
new is injected. When regularization prevents interpolation, only the component
of `S` predictable from `V` is retained; the specialist-only clinical residual
has no target coordinate in the VLM space. Adding that residual requires a
learned cross-model semantic map or a class head, returning to adapter training
or expert-score fusion.

A global alignment across many images can transport specialist task directions,
but it is a trained model-stitching adapter, not training-free case-level
collaboration. A disease-head-weighted patch map followed by VLM token weighting
is simply expert CAM/attention guidance.

## Collision

- Training-free latent/model stitching via shared activations and orthogonal
  Procrustes is established work.
- PEARL (2026) performs training-free Procrustes alignment inside frozen ViTs.
- Training-free task-vector transport across architectures (2026) explicitly
  solves input/output Procrustes maps.
- PVI and MedBridge cover auxiliary visual feature injection with learned
  adapters; Visual Evidence Prompting and CCD cover expert-output guidance.

## Boundary

The attractive phrase “inject representations, not labels” does not solve the
coordinate-identification problem. Do not run a VLM experiment unless a future
method supplies an identifiable specialist-only coordinate without labels,
adapter fitting, expert score fusion, or reconstruction of the native VLM
features.
