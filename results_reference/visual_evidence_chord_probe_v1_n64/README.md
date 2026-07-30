# Visual Evidence Chord Probe

Compact reference artifacts for the HuatuoGPT-Vision-7B natural MIMIC
mechanism audit.

- Raw diagnostic output remains under
  `corrected_runs/visual_evidence_chord_probe_v1_n64/raw.jsonl`.
- Raw SHA256:
  `463e51311480bbe3bc82b2743acc0b960dd6d53eec91403d3fa67ad91f1bb1b9`.
- `summary_all.json` covers 64 unique images.
- `summary_frontal.json` covers the 40 images classified as frontal CXR by
  the local BiomedCLIP audit in `view_position.jsonl`.
- Teacher-forced sentence likelihoods are mechanism diagnostics, not final
  predictions.

Reanalyze the primary subset with:

```bash
PYTHONPATH=. python -m \
  anchor.corrected_sgta.analyze_visual_evidence_chord_probe \
  --input corrected_runs/visual_evidence_chord_probe_v1_n64/raw.jsonl \
  --view-labels \
  corrected_runs/visual_evidence_chord_probe_v1_n64/view_position.jsonl \
  --output \
  corrected_runs/visual_evidence_chord_probe_v1_n64/summary_frontal.json \
  --figure \
  corrected_runs/visual_evidence_chord_probe_v1_n64/chord_frontal.png \
  --permutation-draws 200
```
