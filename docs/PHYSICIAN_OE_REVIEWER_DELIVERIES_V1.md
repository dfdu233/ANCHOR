# Physician OE role-isolated reviewer deliveries

> Historical package (v1). Do not regenerate or distribute it for a new
> review. The preferred, browser-tested package is documented in
> `docs/PHYSICIAN_OE_REVIEWER_DELIVERIES_V2.md`. The commands below describe
> how this package was produced at the time; the current packager emits v2 and
> must use the separate v2 output directory.

The frozen 24-image, nine-arm VQA-RAD T2 review is packaged into two
self-contained reviewer archives outside Git:

`/home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1/`

Send exactly one archive to each assigned reviewer. Do not send the containing
directory, `review.private_mapping.jsonl`, or any model/method scores.

| Role | Archive | SHA-256 | Size |
|---|---|---|---:|
| A | `physician_oe_reviewer_A_86bbf2b71d98.tar.gz` | `ce466a995bde32c70acc447483c2e68d1497925a70b6991aa56c8e2300036677` | 1,267,433 bytes |
| B | `physician_oe_reviewer_B_86bbf2b71d98.tar.gz` | `959aea3d2b37656509e0dffa3236ad174b52a02c25c4787f8d57300e05a36ff3` | 1,267,436 bytes |

Each archive contains only its assigned frozen JSONL, 24 SHA-256-named images,
reviewer instructions, the frozen runbook, the blank clarification template,
an image checksum inventory, and an internal manifest. It contains 30 regular
files and no other reviewer sheet, private mapping, method identity, symlink,
or unsafe path.

The independent result in `verification.json` reports `passed: true` for both
roles. The reviewer JSONL and image bytes are verified inside the archives,
not merely trusted from filenames. Existing source templates and review-pack
content were not modified.

Historical commands (retained as provenance; do not rerun against this output
directory with the current v2 packager):

```bash
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.package_physician_oe_deliveries \
  --delivery-dir corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/deliveries_v1 \
  --metadata corrected_runs/unified_eval/physician_review/vqa_rad_t2_multiarm_v1/review.metadata.json \
  --runbook docs/PHYSICIAN_OE_REVIEW_RUNBOOK.md \
  --output-dir /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.verify_physician_oe_delivery_archives \
  --delivery-dir /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1 \
  --output /home/dbw/datasets/public/vqa_rad_hf/physician_review_deliveries/vqa_rad_t2_multiarm_v1/verification.json
```
