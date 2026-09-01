# Diagnostic Completion progression audit

Date: 2026-08-02  
Decision: **NO-GO for the 1,876-image parent-union generation; a new natural-OE pilot of at most 128 images is the only permitted successor.**

## Why the new counts do not authorize expansion

The substrate scan found 12 neutral, 45 existential, and 114
negative-obligation parent-to-diagnosis text transitions. These are useful for
hypothesis generation, but not independent admissible evidence. All three
scans read the same Huatuo generation artifact, whose preregistered
response-geometry qualification failed: only 11/200 existential--neutral and
18/200 negative-obligation--neutral pairs met the frozen length match, versus
50 required for each. That artifact explicitly sets
`human_claim_audit_authorized=false` and
`second_model_generation_authorized_from_this_model=false`.

The largest transition count is also the least natural condition. The
negative-obligation prompt asks for absent abnormalities and uncertainty. Its
outputs have a 99.5% dominant 10-token prefix, only two distinct 10-token
prefixes, and 87% exact cross-image report repetition. This is the already
frozen prompt-conditioned Template Collapse diagnostic. Recounting target
diagnoses inside that template cannot turn it into spontaneous OE behavior.

Finally, a child label with zero of three VinDr readers is an image-reader-panel
state, not sufficient clinical truth that a diagnosis is false. A pneumonia or
tumor impression may depend on history, follow-up, pathology, or other views.
Calling these transitions hallucinations therefore requires physician
construct admission, not only lexical matching to image labels.

The hash-bound executable decision is
`corrected_runs/specificity_ratchet/diagnostic_completion_progression_gate_v1.json`.
It supersedes only the three substrate files' proposed *next action*; their
counts and diagnostic value remain unchanged.

## Permitted next experiment

The surviving question is narrow and testable:

> Under a natural radiology OE request, does a model spontaneously turn a
> reader-supported observation into a more specific diagnostic claim whose
> image support is weaker, or are prior events induced by the prompt/template?

The next substrate must use one concise prompt that does not request absent
findings, differential diagnoses, uncertainty, or any target diagnosis. Freeze
at most 128 images and the parent-to-child ontology before reading model
answers; keep child votes hidden during generation and event extraction. The
pilot must report exact/prefix template concentration, length, cap hits,
refusal, claim count, and parent mention. It stops unless at least two semantic
edges repeat in both child `0/3` and `3/3` strata. Even if that gate passes,
physician review must distinguish image-reader support from clinical diagnostic
truth before any hidden-state replay or larger generation is authorized.

This is a progression decision, not a negative result about diagnostic
completion itself.
