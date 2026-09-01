# PCEM alternative public substrate audit（2026-08-03）

## 冻结标准

本审计只寻找能在**同一身份空间**内闭合以下链路的数据；跨机构 patient ID 不允许拼接：

1. CXR 有可靠 `AP/PA` projection metadata；
2. 有 patient identity 与真实/相对时间，能构造短窗配对；
3. 有独立于 CXR report 的 echo、cardiac CT/MRI 或 chamber measurement truth；
4. report-derived `cardiomegaly`、图像派生 CTR、模型预测均不算独立 truth。

## 结论

**NONE FOUND：截至 2026-08-03，没有检索到可直接下载/申请后即可运行、同时满足 AP/PA、同患者短窗和现成独立 cardiac truth 的公开数据集。**

最接近的两个底座互为缺项：

- **MIMIC-CXR + MIMIC-IV-ECHO**：科学设计最合适；MIMIC-CXR 已有大量 AP/PA 同患者短窗对，但当前账号对 ECHO structured measurement 返回 403，join 尚未闭合。
- **BIMCV-COVID19+**：同一发布内有 projection、subject/session/date、CXR 与 CT，但**没有 chamber/heart-size truth**；只有在取得 metadata 后证实存在足量短窗 `AP/PA–CT` 交集，并额外建立 CT-derived cardiac truth，才可升级为替代底座。

因此现在不应下载任何候选影像，也不应把 PadChest/CheXpert/NIH 的 report cardiomegaly 包装成 gold standard。

## 逐项审计

| 数据集 | AP/PA metadata | patient/time identity | 独立 cardiac truth | 可获得字段、规模与许可 | 判定 |
|---|---|---|---|---|---|
| **CheXpert** | CSV 含 `Frontal/Lateral` 与 `AP/PA`；frontal 同时含 AP/PA | `Path` 编码 patient/study，可 patient-wise split；公开 CSV 不提供可用于短窗 join 的 study datetime | 无 echo/CT/chamber truth；14 类标签由报告抽取，验证集专家标签仍是 CXR interpretation | 224,316 CXRs、65,240 patients；`Path, Sex, Age, Frontal/Lateral, AP/PA` 与 14 observations；Stanford AIMI 下载并接受 DUA/Research Use terms。[official page](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)；[paper](https://arxiv.org/abs/1901.07031) | **NO-GO**：缺时间与独立 truth |
| **PadChest** | `Projection` 明确区分 `PA, AP, AP-horizontal, L, COSTAL` | `PatientID, StudyID, StudyDate_DICOM, ImageID` 可作患者纵向分析 | 无 echo/CT/chamber truth；cardiomegaly/findings 来自 radiology report，约 27% physician-annotated、其余自动抽取 | 160,868 labeled CXRs、69,882 patients、2009–2017；33-field CSV；research-only、免费申请、禁止再分发/分享下载链接。[paper fields](https://arxiv.org/abs/1901.07441)；[official page](https://bimcv.cipf.es/bimcv-projects/padchest/)；[RUA](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/) | **NO-GO**：投影机制现象很好，但无独立 truth |
| **NIH ChestX-ray14** | metadata 含 `View Position`（AP/PA） | `Patient ID` 与 image/follow-up ordinal；没有可靠 acquisition datetime 可定义小时/日短窗 | 无 echo/CT/chamber truth；14 labels 为 report-NLP labels | 112,120 frontal CXRs、30,805 patients；PNG + `Finding Labels, Follow-up #, Patient ID, age, sex, View Position`。官方为公开 Box 下载；官方页未给出可独立确认的标准许可证文本，故只按其发布条款使用，不把第三方 CC0 声明外推到原始数据。[official repository](https://nihcc.app.box.com/v/ChestXray-NIHCC)；[original paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html) | **NO-GO**：缺真实时间与独立 truth |
| **BIMCV-COVID19+** | 官方说明提供 projection type 与 acquisition parameters | MIDS 以 subject → session → series 组织；JSON/TSV 含 anonymized DICOM metadata 与 `StudyDate`，同发布内包含 CXR/CT | 有原始 CT，但无 cardiac chamber/heart-size measurements、segmentation 或 adjudicated labels；报告 cardiomegaly 不算 truth | 当前官方页面列 21,342 CR、34,829 DX、7,918 CT studies；metadata/derivative/session TSV 可单独申请；research-only 免费申请、禁止再分发/再识别。[official page](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/)；[official repo](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19)；[paper](https://arxiv.org/abs/2006.01174)；[RUA](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/bimcv-covid19-dataset-research-use-agreement-2/) | **CONDITIONAL ONLY**：先做 metadata triple-overlap；需新建 CT truth，不能直接开实验 |
| **UK Biobank** | UKB imaging study 没有 CXR image modality；住院表中的 `U07.3 Plain x-ray of chest` 是 procedure code，不是可下载 CXR pixels/AP-PA metadata | participant ID 与 longitudinal health records 强 | 有 cardiac MRI images 与 LV/RV size/function derived phenotypes | imaging modalities 为 brain/heart/abdominal MRI、carotid ultrasound、DXA、ECG；需要正式申请、付费、签 MTA 并在 UKB-RAP 使用；新申请截至 2026-08-03 暂停，官方称计划 2026 年末恢复。[imaging catalogue](https://biobank.ndph.ox.ac.uk/ukb/cats.cgi)；[Heart MRI](https://biobank.ndph.ox.ac.uk/ukb/label.cgi?id=102)；[access](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/) | **NO-GO**：有强 cardiac truth，但无 linked CXR images |
| **CheXchoNet** | **PA-only**；明确排除全部 portable AP | hashed patient ID 与 patient-relative shifted CXR time；echo 与 CXR 在 ±12 months，多个 echo 取最大值 | `IVSd, LVIDd, LVPWd` 与 `SLVH, DLV, composite`，来自 echo | 71,589 PA CXRs、24,689 patients；metadata 另含 pixel spacing、age/sex、transplant、pacemaker/ICD；PhysioNet Restricted Health Data License 1.5.0 + project DUA。[official page](https://physionet.org/content/chexchonet/1.0.0/) | **NO-GO for PCEM**：缺 AP contrast，且 echo window 过宽；可作 PA-only external truth validation |
| **Dávila-García et al. 2026** | **AP-only** | 每例 AP CXR 在 TTE 24 h 内；6,467 unique patients | same-day TTE chamber enlargement reference；这是最强的 AP truth cohort | 5,158 train / 655 val / 654 test；4,060/6,467 有任一 chamber enlargement；论文有 supplement，但检索不到公开 patient-level data、images、manifest 或 code release，属于机构回顾性队列。[PubMed/DOI](https://pubmed.ncbi.nlm.nih.gov/42390349/) | **NO-GO for PCEM**：缺 PA contrast且数据未公开；可联系作者获取 AP truth cohort/派生 score |

## 为什么 BIMCV 仍不能写成“已找到”

BIMCV 的确比 PadChest 多了一条同身份空间内的 CT 路径：其论文说明纳入观察期内同一 subject 的全部 CR/DX/CT，MIDS 保留 subject、session 与 StudyDate。这使它有机会回答 projection-conditioned apparent geometry，而不是源域分类。

但目前仍缺三个决定性量：

1. 同一 patient 是否同时有 AP、PA 与 CT；
2. 三者最近时间差在 24 h / 72 h / 7 d 内各有多少；
3. CT 是否足以形成与研究 claim 一致的 truth。COVID chest CT 多为非心电门控，适合总心脏/心包 silhouette morphometry 的可能性高于精确 chamber volume；必须先做 blinded clinician protocol，不能直接把 CT 报告中的 cardiomegaly 当 truth。

所以 BIMCV 是一个**需要新建 reference standard 的队列发现底座**，而不是现成 PCEM benchmark。

## 最小 access action 与执行门槛

### 主路径：补齐 MIMIC-IV-ECHO 权限

1. 在 PhysioNet 项目页确认当前 credentialed user 已被明确授予 MIMIC-IV-ECHO，而不只是 MIMIC-CXR/MIMIC-IV；完成该项目要求的 training、DUA 与身份审批，403 未解除则向 PhysioNet support 提交 project-access ticket。
2. 权限恢复后只下载 structured measurement 与必要 index，不先下载 echo waveform/video。
3. 与本地已审计的 MIMIC-CXR metadata 做 `subject_id + study time` join；本地已有 15,185 名同时有 AP/PA 的患者，最近 AP/PA 距离 `<=6 h` 有 1,491 links（1,158 unique patients），`<=24 h` 有 5,246 links（3,280 patients）。

这是唯一无需重新创造 cardiac truth、同时具有规模和机制辨识力的路径。

### 低成本备选：BIMCV metadata-only gate

只申请/download `derivative`, `metadata`, `sessions_tsv`，不下载影像。统计：

- 每位 subject 的 AP、PA、CT 存在性；
- AP–CT、PA–CT、AP–PA 最近间隔在 24 h / 72 h / 7 d 的数量；
- triple-overlap unique patients 与医院/设备/住院状态混杂。

只有在 **至少 300 位患者具有可比较的 AP/PA–CT 短窗结构**，且抽样 CT 能稳定获得与 claim 对应的 independent morphometry 时，才值得建立 CT truth。若只是分别拥有 CXR 与 CT、或跨模态时间过远，立即淘汰，不下载 pixels。

## 冻结后的科学边界

- 当前 PCEM 状态：**ACCESS-BLOCKED / GPU NO-GO**，不是机制被证伪。
- PadChest 可用于验证“projection 会改变 apparent geometry 和语言 commitment”的现象，但不能证明内在心脏大小不变。
- CheXchoNet 可验证 PA 下模型对 echo structural truth 的方向性，但不能识别 AP→PA 的 projection-induced erasure。
- 真正可发表的因果单元仍应是：同一或可严格匹配的 intrinsic cardiac state 下，projection 改变 apparent evidence，而 decoder 是否把该 conditional evidence source 抹除；干预只修正 certainty，不替换 claim identity。
