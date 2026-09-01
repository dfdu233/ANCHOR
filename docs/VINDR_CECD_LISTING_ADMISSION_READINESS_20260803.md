# VinDr CECD listing admission：冻结的盲审交付与执行边界

**Freeze:** 2026-08-03
**Verdict:** **STRUCTURALLY READY, NOT CLINICALLY ADMITTED.**
**Execution:** outcome-blind CPU construction and validation only. No model
output or score was read, and this work did not launch or authorize GPU use.

## 1. 当前结论

14-finding closed-ontology listing 的独立 admission pack 与四个角色隔离的
delivery archives 已经构建并通过结构、hash、像素、空白返回表与泄漏检查。
这只证明材料可交给四位独立 reviewer，不证明 render 或 prompt 已等价。

在四份真人回传通过固定 schema 验证和预冻结 adjudication 之前，以下字段均
保持 `false`：

- `listing_render_equivalence_admitted`；
- `listing_prompt_equivalence_admitted`；
- `model_scoring_authorized`；
- `gpu_authorized`；
- `efficacy_claim_authorized`。

因此现在不得生成 Huatuo/Hulu listing outputs，也不得称作 clinical admission。

## 2. Outcome-blind 样本冻结

没有根据任何模型行为二次挑选影像。pack 原样使用 420-image breadth manifest
中预冻结的完整 pilot：60 images，三个 strata 各20例：

1. `unanimous_no_finding`；
2. `one_unanimous_target_finding`；
3. `multiple_unanimous_target_findings`。

selection SHA-256 为
`d8b8170b8c07eabdc5cb9867adadb2ff32932ec84f936bff0fa8125b72f4d747`。
源 breadth manifest SHA-256 为
`79e7469be61ef17ad9ac7764652e077434cad39fc9a806d7118ce659ec97be06`。

同一原始 DICOM/像素可与早先 clinical image review 重合，这是对同一患者影像
的资源复用；但本 pack 的任务语义是“14类多-claim listing 中所有 claim 的
support 是否在 render pair 间保持”，不是二元 polar claim。旧任务的决定、
标签或准入结论均未导入，reviewer 必须对14类联合重新判断。

## 3. 四角色、两种构念

Clinical render review 使用两位独立 clinical reviewers：

- 60 images × 4 non-baseline science renders = 240 primary pairs；
- 另加12个 exact-identity controls；
- 共252 pairs、504 blinded PNGs；
- 每个 pair 联合判断全部14个 findings，而不是沿用某个 polar prompt 的准入。

Prompt review 不接触影像，由一位 clinical-template reviewer 与一位独立
language reviewer 完成：

- `inspect_and_list` 是冻结 baseline；
- 两个 science prompts 分别与 baseline 比较；
- 一个 exact duplicate control；
- 共3个 blinded prompt pairs。

四个 delivery archives 每个只包含本角色的 instruction、唯一空白 CSV、唯一
attestation template 及必要材料。clinical roles 各含504张图；两个 prompt
roles 不含图像。sealed mapping、reader truth、stratum、其他 reviewer sheet、
模型输出与模型分数均未交付。

## 4. 计算 guard 失败的预冻结处理

252 pairs 中251个通过冻结的像素工程 guard；1个
`center_plus_0p05w` pair 的 ROI saturation increase 超过预设上限。这个 pair：

- 不删除、不替换、不放宽阈值，继续留在盲审材料中；
- 其 identity 在 sealed mapping 中 hash-bound；
- 无论未来医生意见如何，永久不具备完整 model-orbit scoring 资格；
- 只表示该 render cell 的工程有效性失败，不表示临床等价性失败。

这样同时避免 outcome-aware sample removal，并阻止临床 review 被用来覆盖已经
失败的计算安全条件。guard-failure pair-ID set SHA-256 为
`9c4dfea0970392d21be5cd554c33747ea8ebf96e1ff59d7acffafdf9e7f02a48`。

## 5. Exact return contract

`RETURN_SCHEMA.json` 是唯一回传合同。validator 要求：

- CSV header、row identity、顺序和所有 immutable fields 与交付表完全一致；
- clinical support-state 改变时必须填写 canonical semicolon-delimited
  `changed_finding_ids`，未改变时必须为空；
- `unable_to_assess` 与决定字段严格一致；
- prompt reviewer 分别判断 ontology、inclusion obligation、certainty demand、
  answer space 与 exact-output grammar；
- comments 拒绝 spreadsheet formula prefixes；
- attestation 的 schema、角色、reviewer identity 与完成时间完整；
- 四个 reviewer IDs 必须互不相同。

return validator 只验证回传是否合法，不计算 admission，不自动 adjudicate，
也不授权模型/GPU。人类回传目前尚不存在。

## 6. 冻结 artifacts 与哈希

Restricted pack 位于：

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_pack_v1`

- `manifest.json`: SHA-256
  `54cb1d96dc5bd66d5ad59ea1cd8bcdbb7dc4acd2102dfc90dd092c51e661f109`；
- `sealed_mapping.json`: SHA-256
  `97040e92a4b9b6d890abbc36d9a8e81ca7f3ef6acc592f5e35a289aa385916bd`；
- `integrity_verification.json`: SHA-256
  `ed72569b02f7dfa21de87de57cfaff0a83729077e18aa376ae3b633a89a84332`；
- structural status:
  `structurally_valid_awaiting_four_independent_human_returns`。

Role deliveries 位于：

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_deliveries_v1`

- `delivery_index.json`: SHA-256
  `7e4343ad61d8788310f409780e7e8d7bc0c2200f5f41e73a538a32450ce313ae`；
- `delivery_verification.json`: SHA-256
  `fe89195d0dec817d8c2a2c68ebd156c924e4147c7760140a57b59aa89c909bbc`；
- clinical reviewer 1 archive:
  `d95340020abb0b6aa6e597b891d9d9f96286bc6301ff4cc4096abecc3e2cb379`；
- clinical reviewer 2 archive:
  `4cd3dc415d87940c37066b15a658e338433c6a492c8529a2f9abdb9be0555911`；
- clinical-template archive:
  `5484aa4c957de3d4cf730889c4f8fd57f97818c2430cd0c234a18d67799d4029`；
- language archive:
  `fbf0a98d5b39fe2d5583a4272a74ca6285f9722bea202ba3f5a981b2a58060d3`；
- delivery status:
  `four_role_delivery_skeleton_verified_awaiting_humans`。

Restricted derived PNGs、sealed truth 和 delivery archives 均位于 repository
之外，不进入 Git 或 Git LFS。

## 7. 可复核实现与命令

实现：

- `anchor/corrected_sgta/build_vindr_cecd_listing_admission_pack_v1.py`；
- `anchor/corrected_sgta/verify_vindr_cecd_listing_admission_pack_v1.py`；
- `anchor/corrected_sgta/package_vindr_cecd_listing_admission_deliveries_v1.py`；
- `anchor/corrected_sgta/validate_vindr_cecd_listing_admission_returns_v1.py`；
- `tests/test_vindr_cecd_listing_admission_v1.py`。

验证命令：

```bash
cd /home/dbw/ANCHOR

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  corrected_sgta.verify_vindr_cecd_listing_admission_pack_v1 \
  --pack-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_pack_v1 \
  --output /new/write_once/integrity_verification.json

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  corrected_sgta.package_vindr_cecd_listing_admission_deliveries_v1 verify \
  --pack-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_pack_v1 \
  --delivery-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_deliveries_v1 \
  --output /new/write_once/delivery_verification.json

PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_vindr_cecd_listing_admission_v1.py \
  tests/test_vindr_cecd_ontology_listing_v1.py \
  tests/test_vindr_manifest_v2.py \
  tests/test_cecd_oe_report_transfer_v1.py
```

该阶段冻结回归结果为 `19 passed`。下一步只能是分发四个角色 archive、
收集独立回传、运行 exact return validator 与预冻结 adjudication；在此之前
不打开 listing model inference。

## 8. 断线持久 return monitor

`scripts/monitor_vindr_cecd_listing_returns_v1.py` 只观察专用 inbox 中四角色的
八个 exact filenames。输入必须连续两次 poll 保持 size 与 SHA-256 不变，之后
仅调用上述 pack verifier 与 exact return validator。别名或意外文件会形成
fail-closed heartbeat；`.tmp`/`.partial` 只用于原子复制，永不作为输入。

该 monitor 的 terminal state 仅为
`four_independent_returns_structurally_valid`。它不复制/合并回传、不打开 sealed
mapping、不做 adjudication 或 admission decision，也没有任何 model/GPU launch
代码路径。其 canonical detached job 名称为
`vindr-cecd-listing-returns-v1`；state、log 与 heartbeat 分别位于：

- `/home/dbw/ANCHOR/corrected_runs/detached_jobs/vindr-cecd-listing-returns-v1.json`；
- `/home/dbw/ANCHOR/corrected_runs/detached_jobs/vindr-cecd-listing-returns-v1.log`；
- `/home/dbw/ANCHOR/corrected_runs/vindr_v2/cecd_listing_admission_returns_v1/monitor.heartbeat.json`。

专用 inbox 为：

`/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_returns_v1`

要求的 exact filenames 是每个 role 的 `<role>.completed.csv` 与
`<role>.attestation.json`，其中 role 必须严格属于：
`clinical_reviewer_1`、`clinical_reviewer_2`、
`clinical_template_reviewer`、`language_reviewer`。

monitor 加入后的聚焦回归结果为 `22 passed`。首次 detached launch 的 supervisor
PID 为 `612205`，monitor child PID 为 `612206`；后续以 canonical state JSON 中
记录的 PID 和 `/proc` 存活状态为准，而不是依赖本段的瞬时 PID。
