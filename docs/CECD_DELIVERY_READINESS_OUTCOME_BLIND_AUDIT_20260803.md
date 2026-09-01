# CECD reviewer delivery readiness — outcome-blind audit (2026-08-03)

## Verdict

The CECD four-role admission package is engineering-ready for external human
review, but scientific admission is still blocked on four genuinely independent
human returns: two physicians for the image-pair review, one physician for the
clinical-template review, and one language expert. This audit did not read,
create, repair, infer, or summarize any reviewer answer and did not open or
modify the sealed mapping.

One real delivery blocker was found and fixed. The embedded instructions named
the blank source `*.csv`, while the browser form and monitor require the exported
`*.completed.csv`. A reviewer following the old prose could therefore return a
misnamed file that the monitor would never consume. The current instructions now
name the exact export and require full extraction before opening the form so the
relative image paths remain valid. The frozen scientific sheets, images,
protocol ID, role assignments, and sealed mapping were not changed.

## Current deliverables

Directory:

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries_v3/`

| Role | Archive | SHA-256 | Rows | Images |
|---|---|---|---:|---:|
| physician 1 | `cecd_clinical_reviewer_1_v3.tar.gz` | `dcf31639d6bba8947742f1b9d3c3662de3bb21689998538c9dbdf5bb99024dc5` | 252 | 504 |
| physician 2 | `cecd_clinical_reviewer_2_v3.tar.gz` | `e3b463e4f33fe4137ddf145cd88408bc0c8f14c2ca8a0a72cac1a04acf98dcd6` | 252 | 504 |
| clinical-template physician | `cecd_clinical_template_reviewer_v3.tar.gz` | `cd832ef3729f2e11e420f0757c818c8594c0b5a6ff164d0d78fbcc724b90ca0c` | 8 | 0 |
| language expert | `cecd_language_reviewer_v3.tar.gz` | `3678d7e2a8febaf5a9029085af713e15b02b596f54b3630bc73073d1d94b3ba5` | 8 | 0 |

Each reviewer must receive exactly one assigned archive through an approved
secure channel. The two clinical archives are about 2.4 GB each; the two
wording-review archives are about 6.6 KB each. Archive permissions remain
owner-only and read-only at rest because the images are restricted data;
transfer or access granting is a coordinator action, not part of this
outcome-blind audit.

The superseded instruction-only build is retained outside Git at
`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries_v3_superseded_filename_instruction_bug_20260803/`.
It must not be sent to reviewers.

## Verified human workflow

1. Extract the entire assigned archive into one folder.
2. Open `REVIEW_FORM.html` locally in Chrome or Edge; no web server is needed.
3. Complete all rows independently. The form autosaves by role and stable
   reviewer ID, supports re-import, rejects immutable-field changes, and does
   not make network requests.
4. Export the completed CSV and the separate attestation JSON. The four stable
   reviewer IDs must be distinct, and the coordinator may not sign or repair an
   attestation.
5. Copy the eight returned files under temporary names and atomically rename
   them to the exact names listed in
   `/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_returns_v3/RETURN_FILES.md`.
6. The detached monitor requires byte-identical size and SHA-256 over two polls,
   validates every field and attestation, freezes hash-named copies, and only
   then permits unblinding and admission analysis.

## Verification evidence

- Static verifier: all four archives passed deterministic identity, role
  isolation, regular-file/path safety, exact frozen CSV bytes, image closure and
  hashes, embedded seed equivalence, and leakage checks.
- Real Chromium smoke: all four forms passed image loading, immutable-field
  tamper rejection, autosave/reload, exact CSV round-trip, exact attestation
  schema, and zero remote request, console error, or page error. Synthetic smoke
  values existed only in a discarded temporary directory.
- Return transition: an outcome-blind `--once` run reached
  `waiting_for_four_independent_returns`, with all eight expected paths missing,
  `clinical_or_language_labels_synthesized=false`,
  `attestations_synthesized=false`, and
  `sealed_mapping_exposed_before_returns_locked=false`.
- Persistence: detached supervisor PID 513549 and monitor PID 513552 were alive;
  the shared watchdog reported `cecd-clinical-admission-monitor-v3: alive`.
- Tests: `14 passed` for `test_cecd_deliveries_v3.py`,
  `test_monitor_cecd_admission_pipeline.py`, and
  `test_cecd_admission_gate.py`.

Current supporting hashes:

- `delivery_index.json`: `aa4c729e781ffae7cc03d6f72a4ac83d53eece619575032df477f46bdc8e2190`
- `verification.json`: `bfd6c0f7385cf799cc900010c9aaefe5cacc54fbd20ec9880121bf2a9e3c6a0b`
- `browser_smoke.json`: `45473a12eb13588f83b2fc06a8a5e656bcfbce2f00bd1bf7dde941616af7bc05`

## Remaining external action

No code can complete the admission without replacing the eight currently
missing files with real, independent human exports. Sending the four assigned
archives, obtaining the four reviews, and securely placing the eight exports in
the inbox remain external human/coordinator actions. GPU execution remains
correctly unauthorized until that gate passes.
