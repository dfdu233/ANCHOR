# RULE Mechanistic Trace Protocol

## Scope and cohort

This is a falsification diagnostic, not an accuracy benchmark. Trace only
clinically adjudicated visual errors and one same-label, question-similar
correct control per error. Exclude RULE parser/interface artifacts and
questions requiring missing comparison evidence. Freeze the cohort before
looking at model states. Qid 917 is an existing adapter rescue suitable for an
interface smoke; it is not evidence of a general mechanism.

Use the original image and the audited external MIMIC-CXR and PubMedVision
source proxies. Do not use the IU-Xray center for a formal IU-Xray test claim.
Keep one frozen transformation strength and record all file hashes.

## Per-token trace

Teacher-force the already cached generated answer under the exact RULE prompt.
For each answer token store vocabulary entropy, top-1/top-2 margin, top-10
tokens, Yes/No surface probabilities, and hidden-state logit-lens values at
decoder layers 0, 8, 16, 24, and 32.

LLaVA-Med inserts 576 CLIP patch embeddings into the Mistral token sequence.
It has no encoder-decoder cross-attention. The relevant quantity is therefore
generated-query to visual-token **self-attention**. Request it only with an
explicit eager attention backend and record an `unavailable_reason` if the
installed Transformers backend does not return weights. Never retain complete
prompt attention tensors.

Capture patch features before and after `mm_projector`. For source-guided view
\(T_d x\), compute patch displacement
\(\delta_{p,d}=1-\cos(z_p(x),z_p(T_d x))\). If attention is available, analyze
\[
R_{t,d}=\sum_p a_{t,p,d}\delta_{p,d}.
\]
A rescue with higher correct-label log odds and moderate displacement in
attended patches supports useful alignment; large displacement with falling
log odds supports information destruction. Attention is observational, so a
causal claim additionally requires patch ablation or activation patching.

The current Fourier banks are global source spectra. A transformed patch is
not a local source center. A formal local center requires streaming source
images through the frozen CLIP tower and storing position-wise 24x24 feature
means, counts, covariance summaries, and provenance.

## Resource budget

The checkpoint has 32 decoder layers, 32 heads, hidden size 4096, and 576 image
tokens. Full fp16 prompt attention at sequence length roughly 650 costs about
0.8--1.0 GiB per example before other activations. Cache-based single-query
attention is about 1.3 MiB per generation step. Process one view at a time and
flush one JSONL/NPZ record per view.

Create the CPU-only audit queue:

```bash
PYTHONPATH=. python -m corrected_sgta.prepare_rule_trace_manifest \
  --questions corrected_runs/rule_protocol_v1/iuxray/dg_pilot128/questions.pilot.jsonl \
  --baseline corrected_runs/rule_protocol_v1/iuxray/dg_pilot128/original_answers.jsonl \
  --output corrected_runs/rule_protocol_v1/iuxray/mechanistic_trace_v1/audit_queue.json \
  --max-errors 12 --force-qid 917
```

The queue must not directly drive a GPU run. Add an external adjudication file
and fail closed unless the paired error is explicitly marked
`genuine_visual_error`.
