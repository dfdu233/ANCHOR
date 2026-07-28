# Method Registry

Configured methods live in `configs/methods.yaml`. Each method declares supported task types so unsupported dataset-method combinations are skipped with a clear status instead of crashing.

Main groups:

- `baseline`: greedy/beam decoding and complete-sequence Yes/No scoring.
- `anchor`: source-margin VQA, source-word-center report generation, feature SGTA, and SGTA-ConfGen.
- `ce_methods`: SCA-T, LAME, and LATA for constrained Yes/No classification.
- `mitigation`: VCD, DoLa, OPERA, AVISC, M3ID, DAMRO, PAI, and VISTA adapters.
- `judges`: RULE parser, string matching, ROUGE, and opencode judge paths.

The packaged runner creates the standard run layout:

```text
runs/{timestamp}/{dataset}/{model}/{method}/
  config.json
  command.txt
  fingerprint.json
  raw.jsonl
  records.jsonl
  metrics.json
  summary.json
```

Heavy GPU inference implementations are preserved under `anchor/corrected_sgta/` and `third_party/`; new bridges should write into the same run schema. Do not report registry-only smoke outputs as model accuracy.


Note: `third_party/baselines/VISTA/MMHal-Bench/` is intentionally excluded because the local source contained unresolved Git LFS pointer files rather than recoverable image objects. VISTA method code remains packaged; reacquire MMHal assets from the upstream source if needed.

## Extended Method Inventory

The compact YAML registry is intentionally conservative. For the full migration-level inventory of newly added and historically valuable methods, see `docs/METHOD_ZOO.md`. That document records LET, VISTA/SLA, source-margin, source-word-center, ANCHOR-Flow, Riemann/NBP gates, VAF, SGTA/ConfGen, official mitigation baselines, and stopped directions.
