# CECD v2 blinded reviewer delivery record

> Historical only. Do not newly distribute these archives. The preferred
> structured, separately attested workflow is recorded in
> `docs/CECD_REVIEWER_DELIVERIES_V3.md`.

The frozen admission pack is not itself reviewer-safe because it colocates sealed analysis material and all four reviewer sheets. Four role-isolated, self-contained archives were therefore built outside Git at:

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries/`

Never send the directory as a whole. The coordinator must send exactly the archive assigned to one reviewer. Exact names, SHA-256 values, and member counts are recorded in `delivery_index.json` and `COORDINATOR_README.md` in that directory.

## Verified invariants

- Source admission pack passed its existing full verifier: 252 clinical pairs, 504 PNGs, 8 language pairs, blank sheets, and no visible hidden-field leakage.
- Each clinical archive contains only its own byte-identical frozen CSV, 504 referenced PNGs, self-contained instructions, and a delivery manifest (507 files total).
- Each wording archive contains only its own byte-identical frozen CSV, self-contained instructions, and a delivery manifest (3 files total); it contains no PNGs.
- No archive contains another reviewer sheet, `sealed_mapping`, selected-claim material, model output, private provenance, unsafe paths, duplicate members, links, or non-regular files.
- All decision fields remain blank. No labels were created or inferred.
- A second real build of the 2.54 GB clinical-reviewer-1 archive was byte-identical to the delivered archive.

The independent result is `verification.json` beside the archives and reports `passed: true` for all four roles. The source pack and delivery checks used no model runtime or GPU.

## Reproducible commands

```bash
python anchor/corrected_sgta/build_cecd_reviewer_deliveries_v1.py \
  --pack-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2 \
  --output-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries

python anchor/corrected_sgta/verify_cecd_reviewer_delivery_v1.py \
  --pack-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2 \
  --delivery-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries \
  --output /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries/verification.json
```

Implementation: `anchor/corrected_sgta/build_cecd_reviewer_deliveries_v1.py`; independent verifier: `anchor/corrected_sgta/verify_cecd_reviewer_delivery_v1.py`; regression test: `tests/test_cecd_reviewer_deliveries_v1.py`.
