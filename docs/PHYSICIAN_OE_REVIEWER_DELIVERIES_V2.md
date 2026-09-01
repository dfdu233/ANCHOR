# Physician OE role-isolated reviewer deliveries v2

The preferred delivery for the frozen 24-image, nine-arm VQA-RAD T2 review is:

`/home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1_ui_v2/`

Send exactly one archive to each assigned reviewer. Do not send the containing
directory, either browser-smoke result, `review.private_mapping.jsonl`, or any
model/method score. The earlier v1 archives remain historical provenance and
should not be distributed for a new review.

| Role | Archive | SHA-256 | Size |
|---|---|---|---:|
| A | `physician_oe_reviewer_A_86bbf2b71d98.tar.gz` | `22f50b5fd94a356fb221f507ec01202a64bf2ca846e79010163fab3fd5ac113e` | 1,280,721 bytes |
| B | `physician_oe_reviewer_B_86bbf2b71d98.tar.gz` | `86180b7396e94ac4cce9b8f0e3ce54076e8f4534c514cf0a1e173a986913d056` | 1,280,724 bytes |

`delivery_index.json` has SHA-256
`96d2134718629b23bc6bc77b0672e766e0b9a82fdb26f30d4a2b7304e84d51a5`.
The independent archive verifier reports `passed: true` for both roles.

## What changed from v1

Each role-isolated archive now contains `REVIEW_FORM.html`, a self-contained
offline editor for the assigned frozen JSONL. Fixed image, question, reference,
answer, ordering, phase, and reviewer fields are not editable. The form offers:

- group navigation and local-browser autosave;
- structured reference, atomic-claim, omission, harm, confidence, and rationale
  fields;
- immutable-content checks when importing an in-progress JSONL;
- browser-side completeness diagnostics and completed JSONL export.

The form is convenience tooling, not a source of clinical truth. The packaged
Python validator remains authoritative, and no synthetic annotation or model
identity is included in either archive.

## Independent verification

Each archive has 31 safe regular files: its assigned frozen 101-answer JSONL,
24 SHA-256-named images, the offline form, reviewer instructions, frozen
runbook, blank clarification template, checksum inventory, and internal
manifest. Verification fails closed on archive/file hash mismatch, unsafe
members, wrong reviewer slot, other-role content, private/method JSON fields,
form-seed mismatch, or JSONL/image closure mismatch.

A real Chromium 151 offline smoke test was run separately on both final archive
bytes. Each test rendered 24 groups and 101 answer units, loaded the local
image, preserved edits across reload, rejected a changed question on import,
accepted a structurally complete immutable import, passed browser validation,
and exported an exactly equal JSONL round-trip. Both tests recorded zero
external requests, console errors, or page errors. The synthetic in-memory
annotations used by the smoke test are explicitly non-clinician labels and are
discarded with the temporary extraction directory.

Browser-smoke records outside Git:

- `browser_smoke_A.json`, form SHA-256
  `b7caa2be501c64822390c69b457278c292580463e76987842c1df1f48e863a4e`;
- `browser_smoke_B.json`, form SHA-256
  `32deabb7987e1e9dac7cec862e764f2847325742dab12188a750b59f9402e70c`.

## Reproduction

```bash
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.package_physician_oe_deliveries \
  --delivery-dir corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/deliveries_v1 \
  --metadata corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/review.metadata.json \
  --runbook docs/PHYSICIAN_OE_REVIEW_RUNBOOK.md \
  --output-dir /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1_ui_v2

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.verify_physician_oe_delivery_archives \
  --delivery-dir /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1_ui_v2 \
  --output /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1_ui_v2/verification.json

PLAYWRIGHT_BROWSERS_PATH=/home/dbw/.cache/ms-playwright \
  uv run --with playwright python scripts/smoke_physician_oe_review_form.py \
  --archive /path/to/one/physician_oe_reviewer_ARCHIVE.tar.gz \
  --output /path/to/browser_smoke.json
```

Reviewers should extract the archive, open `REVIEW_FORM.html`, complete the ten
calibration groups first, and follow `PHYSICIAN_OE_REVIEW_RUNBOOK.md`. After
export, the coordinator must run `anchor.medeval.validate_physician_oe_review`
before accepting any returned file or beginning adjudication.
