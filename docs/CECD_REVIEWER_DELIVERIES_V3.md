# CECD v3 reviewer deliveries and automatic return gate

The frozen CECD v2 scientific pack remains unchanged. Reviewer-safe v3
archives live outside Git at:

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries_v3/`

Send exactly one assigned archive to each reviewer, never the directory or the
source pack. v2 deliveries are historical because they did not provide the
structured offline form and separate explicit professional-role,
independence, and sealed-mapping-blinding attestation required by the current
runtime gate.

The reviewer must extract the entire assigned archive into one folder before
opening `REVIEW_FORM.html` in Chrome or Edge; opening the HTML from an archive
preview can break its relative image paths. The embedded instructions name the
exact `*.completed.csv` and attestation exports expected by the return monitor.

## Frozen archive identities

| Role | Archive SHA-256 | Rows | Images |
|---|---|---:|---:|
| clinical reviewer 1 | `dcf31639d6bba8947742f1b9d3c3662de3bb21689998538c9dbdf5bb99024dc5` | 252 | 504 |
| clinical reviewer 2 | `e3b463e4f33fe4137ddf145cd88408bc0c8f14c2ca8a0a72cac1a04acf98dcd6` | 252 | 504 |
| clinical template reviewer | `cd832ef3729f2e11e420f0757c818c8594c0b5a6ff164d0d78fbcc724b90ca0c` | 8 | 0 |
| language reviewer | `3678d7e2a8febaf5a9029085af713e15b02b596f54b3630bc73073d1d94b3ba5` | 8 | 0 |

Supporting artifact hashes:

- `delivery_index.json`: `aa4c729e781ffae7cc03d6f72a4ac83d53eece619575032df477f46bdc8e2190`
- `verification.json`: `bfd6c0f7385cf799cc900010c9aaefe5cacc54fbd20ec9880121bf2a9e3c6a0b`
- `browser_smoke.json`: `45473a12eb13588f83b2fc06a8a5e656bcfbce2f00bd1bf7dde941616af7bc05`

The earlier same-path v3 build is superseded because its prose named the blank
source CSV rather than the form's actual `*.completed.csv` export. It created
no human labels and is retained only as an audit copy outside Git. The current
archives change delivery instructions only; the frozen scientific sheets,
images, protocol ID, and sealed mapping are unchanged.

All four archives passed role isolation, regular-file/path safety, exact frozen
CSV bytes, image closure and content hashes, embedded seed equivalence, and
absence of sealed mapping/model material. A real offline Chromium 151 run then
verified local image rendering, immutable-field tamper rejection, autosave and
reload, exact CSV export, exact attestation schema, and zero network, console,
or page errors. Synthetic smoke decisions existed only in a discarded
temporary directory and are not human labels.

## Human return contract

Each reviewer opens `REVIEW_FORM.html`, completes every row independently, and
returns the form's two exports. Put the eight files in:

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_returns_v3/`

Exact names are:

1. `clinical_reviewer_1.completed.csv`
2. `clinical_reviewer_1.attestation.json`
3. `clinical_reviewer_2.completed.csv`
4. `clinical_reviewer_2.attestation.json`
5. `clinical_template_reviewer.completed.csv`
6. `clinical_template_reviewer.attestation.json`
7. `language_annotator.completed.csv`
8. `language_reviewer.attestation.json`

The four reviewer IDs must be distinct. The coordinator must not fill, repair,
or sign any returned field. Copy files under temporary names and atomically
rename them only when complete.

## Persistent continuation

`cecd-clinical-admission-monitor-v3` polls every 30 seconds under a detached
supervisor and the shared watchdog. It requires unchanged bytes over two
polls, validates all decisions and attestations, freezes hash-named copies,
and only then opens the sealed mapping for the preregistered admission
analysis. Failed admission is terminal. A passing analysis is hash-bound into
new Huatuo and Hulu canary/full configurations and launches exactly one
`cecd-two-model-stage1-v2` job; failed scientific compute is not retried.

A `done` job state alone is insufficient. The monitor additionally rechecks
the detached job identity, admission SHA-256, the input gate, both 160-claim /
3,040-row raw-run hashes, the exact two-model closure, analyzer source hash,
five-fold/5,000-bootstrap/seed-42 statistics, the behavioral gate, and the
explicit hidden-state prohibition. A passing behavioral screen can authorize
only the separately frozen official-Treble method-level collision step; it
cannot authorize hidden-state work or a scalar-surrogate substitute.

No prior CECD model shard is formal evidence. The older 32/160 Huatuo partial
run remains an engineering artifact because it preceded human admission and
its config records admission as pending.

Reproduction entry points are:

- `scripts/build_verify_cecd_deliveries_v3.sh`
- `scripts/smoke_cecd_review_forms_v3.py`
- `scripts/monitor_cecd_admission_pipeline.py`
- `scripts/run_cecd_two_model_stage1_v2.sh`
