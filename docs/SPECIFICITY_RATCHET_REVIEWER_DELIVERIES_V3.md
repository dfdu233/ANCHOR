# Specificity Ratchet physician deliveries v3

This is the preferred independently blinded delivery. The frozen v2 source
pack remains unchanged. V3 adds a self-contained offline form and a separate,
explicit physician attestation export; v2 archives are historical and should
not be newly distributed.

## Frozen delivery

- Source candidate bundle ID: `87ce8db47dca8c632a13a98d`
- Source candidate SHA-256:
  `87ce8db47dca8c632a13a98dfa5830be85f90cfbce164013ddc334323ea8f5b9`
- External directory:
  `/home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/specificity_ratchet_v3/`
- Delivery index SHA-256:
  `cefea59e8fd479925cf0f2d82898f9a1409453978258091726d0a6be25a33ba3`
- Independent verification SHA-256:
  `e658e4da299b139f231d980dd8213c14f452e527c3544e164cfe2cff53040d61`

| Role | Archive | SHA-256 | Bytes |
|---|---|---|---:|
| 1 | `specificity_ratchet_reviewer_1_87ce8db47dca_v3.tar.gz` | `9f46224e512126b7bfa6903f5c95bd44ad7099b5f54a64e312d250dbc6426018` | 3,523,548 |
| 2 | `specificity_ratchet_reviewer_2_87ce8db47dca_v3.tar.gz` | `065a3f2d40ba13addd008886d67ae1ee568d7df4d06163640f8f9de5b8207920` | 3,523,549 |

Each archive contains exactly one assigned blank CSV, 70 hash-named images,
the frozen schema, checksum inventory, instructions, and an offline structured
form. It contains neither the other reviewer's sheet nor private provenance,
model identity, reference answer, adjudication, or clinical labels.

## Browser and verifier acceptance

Both final archives passed the independent archive verifier and real offline
Chromium 151 acceptance:

- 70 image groups and 127 edges rendered;
- local images loaded with networking disabled;
- edits survived reload through local storage;
- a changed immutable question was rejected;
- completed CSV import/export round-tripped exactly;
- the separate physician/independence/private-provenance-blinding attestation
  exported with the same stable reviewer ID;
- no external request, console error, or page error occurred.

Browser evidence hashes are
`875d52d2b0150366130b5cdce595efa7f6f2a00ab4ad9c045fb28d255293cac4`
for role 1 and
`96fd5dd58edec224dcb6b225ce3de099ab438c3fb0c44ad857cb2758cc21e6dc`
for role 2. Synthetic smoke values were created only inside disposable test
directories and are not clinical annotations.

## Required returns and automatic continuation

Return the two files exported by each role under these exact names:

```text
annotations.reviewer_1.completed.csv
reviewer_1.attestation.json
annotations.reviewer_2.completed.csv
reviewer_2.attestation.json
```

Place them in:

```text
/home/dbw/datasets/public/vqa_rad_hf/physician_review_returns/specificity_ratchet_v3/
```

The persistent job `specificity-ratchet-clinical-pipeline-monitor-v1` requires
unchanged bytes over two polls, validates the separately signed IDs and schema,
and then creates a third-role blinded adjudicator archive. It never fills a
clinical field or signs for a physician. After the returned adjudication passes
the frozen validator, it compiles the replay manifest, performs CPU preflight,
and launches exactly one detached Huatuo native-identity canary. A failed
canary is terminal and is not retried.

The 70-case source pack remains a bounded pilot because its label-blind
lexical-overlap ceiling is below the frozen confirmatory gate. Human admission
does not convert it into confirmatory evidence.
