# Specificity Ratchet review-return and adjudication workflow v1

This is an operator workflow. It does not create clinical truth and must not be
used until both physicians have independently completed and frozen their own
CSV. Keep the source pack and the two delivered reviewer archives immutable.

Use the v3 role-isolated archives and exact return protocol in
`SPECIFICITY_RATCHET_REVIEWER_DELIVERIES_V3.md`. Unlike the historical v2
instructions, v3 requires each physician to export a separate attestation;
the coordinator may validate and combine those records but may not sign for a
reviewer or adjudicator. The persistent clinical monitor implements the steps
below and requires unchanged input bytes over two polls.

## 1. Receive and preserve the independent returns

Store each returned CSV under a distinct reviewer-specific path. Record its
SHA-256 before any spreadsheet export or normalization. Do not reveal the
private provenance, model identity, the other physician's labels, or layer
scores before both returns are frozen.

CSV fields must be imported and exported as text. In particular, do not allow
spreadsheet software to interpret cells beginning with `=`, `+`, `-`, or `@`
as formulas. The merger rejects formula-prefixed rationales and any change to
candidate text, row order, edge IDs, or other immutable fields.

## 2. Merge reviewer fields into a fresh adjudication sheet

Run the fail-closed merger against the frozen source pack:

```bash
python anchor/corrected_sgta/merge_specificity_ratchet_reviews_v1.py \
  --pack corrected_runs/specificity_ratchet/vqa_rad_oe_physician_pack_v2 \
  --reviewer-1 /returned/reviewer_1.completed.csv \
  --reviewer-2 /returned/reviewer_2.completed.csv \
  --output /work/specificity_ratchet/adjudication.with_reviews.csv
```

The output path must not already exist. The tool requires distinct, stable
reviewer IDs, validates every allowed value, copies only reviewer fields, and
leaves every `final_*`, adjudicator, disagreement, and adjudication-rationale
field blank. Save the printed input and output hashes with the experiment
record.

## 3. Adjudicate while still blinded

The physician adjudicator receives the merged sheet and images, but not
`provenance.private.jsonl`, model identities, reference answers, automatic
labeler outputs, or hidden-state results. The adjudicator fills all final
fields, one stable `adjudicator_id`, a rationale for every edge, and a
`disagreement_reason` wherever reviewer categorical fields differ.

The v3 adjudication form exports `adjudicator.attestation.json` separately.
The monitor validates the two reviewer attestations and this adjudicator
attestation against their respective completed CSV IDs, then mechanically
combines the three signed records into `physician_attestations.json`. Reviewer
and adjudicator IDs must exactly match the completed files; no attestation
value may be inferred from a CSV or created by the coordinator.

## 4. Validate in a disposable working pack

Never overwrite the frozen blank source pack. Create a working copy, then put
the two completed reviewer sheets, completed adjudication sheet, and
attestations at the filenames expected by the validator:

```text
annotations.reviewer_1.csv
annotations.reviewer_2.csv
adjudication.csv
physician_attestations.json
```

Run:

```bash
python anchor/corrected_sgta/validate_specificity_ratchet_adjudication_v1.py \
  --pack /work/specificity_ratchet/adjudicated_pack_v1
```

Any refusal is a hard stop. Do not repair clinical states automatically. A
physician must resolve semantic inconsistencies and refreeze the affected
files. Only after validation succeeds may the mechanism-manifest compiler run;
only after that manifest is frozen may the first Huatuo GPU canary start.
