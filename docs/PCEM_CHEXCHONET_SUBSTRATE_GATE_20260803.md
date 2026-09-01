# PCEM substrate gate：CheXchoNet exclusion 与 MIMIC-IV-ECHO access block

Date: 2026-08-03  
Decision: **ACCESS-BLOCKED / UNIDENTIFIED; GPU NO-GO**  
Scope: official documentation plus metadata-only CPU audit; no image downloaded.

## 1. Outcome

新线索修正了 PCEM 的数据前景，但今天仍不能进入模型实验：

1. **CheXchoNet 不能构造 PCEM。** 官方 cohort 明确只保留 PA，portable AP 在建库时全部排除。它没有 AP/PA projection contrast，并且来自 Columbia，不能与 BIDMC 的 MIMIC-CXR 通过 patient ID 连接。
2. **MIMIC-IV-ECHO + MIMIC-CXR 是正确的候选 substrate。** 两者属于 MIMIC identity/time system；MIMIC-IV-ECHO 提供 clinician-verified structured echo measurements 与 `measurement_datetime`，MIMIC-CXR metadata 提供 `subject_id`、`ViewPosition`、`StudyDate/StudyTime`。
3. MIMIC-CXR metadata/split 已合法下载并完成全量 CPU count；MIMIC-IV-ECHO `structured_measurement` 在 Basic authentication 后返回 HTTP 403，未下载任何 byte。故所有 echo-qualified join、LVIDd 分层和 borderline counts 必须为 `null`，不能用网页的 206,488 studies 代替实际 join count。
4. 即使获得 echo 文件，**LVIDd/SLVH/DLV 也不等同于 radiographic cardiomegaly**。LVIDd 是 LV cavity diameter，IVSd/LVPWd 是壁厚；CXR cardiomegaly 是投影下总 cardiac silhouette。Echo 是独立结构真值，但必须改变 claim contract，不能把一个 LVH proxy 包装成“心影增大真值”。

因此当前状态是：**projection metadata substrate identified，independent heart-size truth unjoined，construct unresolved，GPU unauthorized。**

## 2. Official-source audit

### 2.1 CheXchoNet v1.0.0

[PhysioNet 官方页](https://physionet.org/content/chexchonet/1.0.0/)给出：

- 71,589 CXRs，24,689 patients，CUIMC，2013--2018；
- 只包含 PA films；portable AP 为防止 label leakage 被设计性排除；
- echo 与 CXR 允许相距最多 12 个月，多次 echo 时对每项 measurement 取 12 月窗内最大值；
- continuous labels 为 IVSd、LVIDd、LVPWd；binary labels 为 SLVH、DLV 和二者 composite；
- patient identifier 是 CheXchoNet 内部 32-character hash。

这造成三个 fatal failures：

| Requirement | CheXchoNet |
|---|---|
| AP 与 PA 自然 projection contrast | **0；AP 全排除** |
| 与 MIMIC-CXR `subject_id` 可连接 | **否；不同医院、不同 ID system** |
| 同一短时 cardiac state | **未保证；echo window 可达 ±12 months，且取窗口最大值** |

CheXchoNet 对“PA CXR 能否筛查 echo-confirmed LV structural disease”很有价值，但不是 PCEM 数据。将 CheXchoNet echo labels 横向贴到 MIMIC AP/PA 图像是不可连接的 cohort mixing，不是 join。

### 2.2 MIMIC-IV-ECHO v1.0

[官方页](https://physionet.org/content/mimic-iv-echo/1.0/)说明：

- 206,488 structured echo studies / 91,372 patients；其中 TTE 179,928、stress 16,389、TEE 10,171；
- `subject_id` 使用跨 MIMIC-IV modules 的相同 deterministic mapping；
- `measurement_datetime` 保留同 patient 内相对时间关系；
- structured measurements 经过 clinician review/correction；
- 约 180--230 个 measurement variables，包括 chamber sizes/volumes、function、valves 和 hemodynamics；
- 系统迁移导致 measurement naming/availability 异质，LVEF 就存在多个字段族；同样必须审计 LVIDd/diameter 字段和单位，不能关键词拼接。

这满足 PCEM 的**潜在** identity/time/independent-measurement 条件。但网页规模只是 release 描述，不是本地 join count。

Access probe：

```text
requested: mimic-iv-echo/1.0/structured_measurement.csv(.gz)
authentication: interactive Basic auth
result: HTTP 403 Forbidden
downloaded bytes: 0
```

最可能原因是当前账号尚未完成该 2026 项目的 required training/DUA。密码未写入脚本、文件、环境变量或报告。

## 3. MIMIC-CXR metadata CPU counts

仅下载以下两个 metadata files；没有请求 image path：

- `mimic-cxr-2.0.0-metadata.csv.gz`, 16 MB, SHA-256 `6a3748ce...e1649d6b`;
- `mimic-cxr-2.0.0-split.csv.gz`, 12 MB, SHA-256 `515997bd...bedb2f`.

可复现代码：

- `anchor/corrected_sgta/audit_pcem_mimic_metadata_v1.py`;
- `corrected_runs/pcem_chexchonet_substrate_gate_v1/counts.json`;
- counts SHA-256 `9b8f2b9b...0b5fd1c`.

全量结果：

| Quantity | Count |
|---|---:|
| images | 377,110 |
| patients | 65,379 |
| AP images | 147,173 |
| PA images | 96,161 |
| patients with at least one AP and PA | 15,185 |
| exact datetime parse failures among AP/PA | 0 |

对每个 AP 选择同 patient 时间最近的 PA 后，metadata-only candidate availability 为：

| Maximum AP--PA interval | nearest links | unique patients |
|---|---:|---:|
| 6 h | 1,491 | 1,158 |
| 24 h | 5,246 | 3,280 |
| 72 h | 13,085 | 5,982 |
| 7 d | 21,825 | 7,593 |
| 30 d | 38,619 | 9,773 |

另有 345 个 nearest AP/PA links 位于同一 `study_id`。这些是**候选链接而非独立样本数**：同一 PA 可成为多个 AP 的 nearest neighbor，一个 patient 也可贡献多次检查。正式 manifest 必须以 patient/episode 聚类，并在 echo join 后只保留 outcome-independent frozen selection。

这些数量说明 projection side 远超预设 `>=300 episodes` 的可行性下限；它们不能说明 echo truth side 可行。

## 4. Echo construct 不是 cardiomegaly ground truth

需要严格区分三个 claim：

```text
radiographic apparent cardiac silhouette enlargement
intrinsic cardiac/chamber structural enlargement
left-ventricular hypertrophy or dilation
```

- AP geometry 可把正常 cardiac silhouette 放大；因此 AP 图像上的 apparent enlargement 不自动是 hallucination。
- LVIDd 测量 LV end-diastolic internal diameter，不覆盖 left/right atria、RV、pericardial effusion 或所有影响 total silhouette 的因素。
- SLVH 主要由 wall thickness 定义；LV hypertrophy 可能不产生相同比例的 CXR silhouette enlargement。
- DLV 比 SLVH 更接近“intrinsic enlargement”，但仍不是 total heart size 的同义词。

所以不能把 `normal LVIDd -> cardiomegaly claim false` 或 `SLVH positive -> cardiomegaly claim true` 写成自动规则。

PCEM 若重开，primary endpoint 应改成：

> 在近时点 echo 显示没有/存在预注册 structural enlargement 的条件下，VLM 是否把 projection-induced apparent enlargement 不恰当地升级为无条件 intrinsic cardiomegaly commitment？

回答必须区分：

1. `apparently enlarged cardiac silhouette`；
2. `assessment limited by AP magnification`；
3. `definite intrinsic cardiomegaly`。

Echo 用于约束第 3 项，不能否定第 1 项。最终 truth 需要 echo composite 加 blinded clinician review，而不是单一阈值 label。

## 5. Frozen one-day admission gate after access is restored

### Gate A — access and schema

- 实际获取 `structured_measurement.csv.gz`；hash 与 row count 完整；
- 只保留 TTE primary，stress/TEE 单独报告；
- 列出所有 LV/LA/RV/chamber-size variables、description、unit、system era 和 missingness；
- 对 LVIDd/IVSd/LVPWd 同义字段做 source-system-aware mapping，不能模糊 substring union；
- 若无法从 `subject_id + measurement_datetime` 与 CXR date/time 连接，立即 NO-GO。

### Gate B — temporal join

预注册 windows 为 echo 与 paired AP/PA episode 的中心时间相距 `<=24 h` primary，`<=72 h` sensitivity。要求：

- `>=300` unique patients with AP+PA and qualified TTE；
- independent structural enlargement positive/negative 各 `>=100`；
- primary borderline structural state `>=100`，borderline 必须由临床 guideline 和 normalization variables 定义；
- AP--PA 间无 major procedure/diuresis/ventilation-state change 的 60-case blinded audit；如果无法从可用 MIMIC events检查，则只允许同-study/极短窗并人工审计。

### Gate C — construct admission

由 cardiologist/radiologist 共同冻结 multi-measure echo composite。至少考虑：

- LVIDd 或 indexed LV end-diastolic volume；
- LA/RV size when available；
- pericardial effusion；
- sex/BSA/indexing；
- measurement system era 与 missingness。

抽取至少 100 个 candidate episodes 双人盲审，确认 `apparent silhouette`、`projection-limited`、`intrinsic enlargement` 三态能可靠区分。自动 CheXpert cardiomegaly label 只能作 report-behavior covariate，不能定义 truth。

任一 gate 失败都不下载 images、不跑 GPU。

## 6. Current machine decision

Machine-readable result explicitly records:

```json
{
  "metadata_projection_substrate_identified": true,
  "echo_join_identified": false,
  "independent_heart_size_truth_identified": false,
  "borderline_truth_identified": false,
  "gpu_authorized": false,
  "decision": "ACCESS_BLOCKED_UNIDENTIFIED"
}
```

No value is imputed for:

- echo rows actually accessible to this account;
- shared CXR--echo patients;
- AP/PA pairs within echo windows;
- LVIDd or composite positive/negative/borderline bins.

## 7. Research decision

MIMIC-IV-ECHO 是目前找到的第一个同时可能提供自然 acquisition state、同 patient 时间轴和独立 cardiac structure measurement 的高价值 substrate；它使 PCEM 从“需要机构数据”提升为“公共数据可能可做”。但今天只能给出 **promising-access-blocked**，不能给 scientific GO。

访问恢复后，优先只运行上述 CSV join。若 24 h 内 qualified episodes 与 construct bins 通过，再下载冻结 manifest 对应的少量 JPG。若 echo construct 不能把 projectional appearance 与 intrinsic enlargement 分开，PCEM 仍应终止，不能退回 report-derived cardiomegaly 或 transform stability。

## 8. Persistent access-to-audit continuation (implemented 2026-08-03)

The access boundary is now operational rather than a manual reminder:

- `scripts/monitor_pcem_echo_access_v1.py` watches only four explicit
  `structured_measurement.csv(.gz)` paths below
  `/home/dbw/datasets/physionet/`. It never authenticates or downloads data.
- The monitor is supervised as detached job `pcem-echo-access-monitor-v1` and
  is listed in `configs/research_active_jobs.json`. Its parent is adopted by
  PID 1 and the research watchdog can recover it after shell or VSCode loss.
- A newly mounted file must keep the same size and `mtime_ns` for 120 seconds.
  Multiple candidates are an ambiguity error; a truncated gzip, missing schema,
  identity conflict, or changed raw-file hash fails closed.
- Once stable, `anchor/corrected_sgta/audit_pcem_echo_join_v1.py` performs only
  a CPU schema and temporal-join audit. It verifies the official fields
  (`subject_id`, `measurement_id`, `measurement_datetime`, `test_type`, long
  measurement name/description/result/unit), selects one AP/PA episode per
  patient without looking at outcomes, and counts nearest-TTE joins for frozen
  pair/echo windows.
- The output never writes patient identifiers. Even a count-qualified result
  is exactly `DATA_GATE_COUNTS_AVAILABLE_CONSTRUCT_REVIEW_REQUIRED`, with
  `independent_heart_size_truth_identified=false`,
  `image_download_authorized=false`, and `gpu_authorized=false`.

The no-access live path is currently
`waiting_for_authorized_echo_mount`. Synthetic integration tests cover absent
input, ambiguous/escaping paths, raw-file drift, truncated gzip, inconsistent
study identity, insufficient temporal counts, and the 300-patient count-passing
case. The latter still cannot cross the clinical construct gate.
