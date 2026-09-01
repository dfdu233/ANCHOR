# CFED 致命构造审计：Reject and Pivot

**审计日期：** 2026-08-02  
**范围：** 仅元数据、标注结构与一手文献；未运行 GPU。  
**裁决：** 原 **Clinical Frame Equivariance Defect (CFED)** 立即 **REJECT**。
不得在 VinDr 上执行既定 flip admission，不得靠重写 flip 语义保留该 idea。

## 第一印象

CFED 原本试图把 `finding identity` 的 reflection-invariance 与
`patient laterality` 的 reflection-equivariance 分开，并用 side-only patch
修复 wrong-location hallucination。叙事简洁，但决定性实验依赖两个未经证明
的前提：VinDr 能给出 patient-side truth，以及水平像素翻转能定义唯一的
临床 counterfactual。新的全量审计同时否定了两者。

## 致命问题

### F6 — 不可验证命题：CRITICAL

全量 `/workspace/vinbigdata/train` 的 15,000 个 DICOM 中，
`PatientOrientation`、`ImageLaterality`、`Laterality`、`ViewPosition`、
`ImageOrientationPatient` 均为 `0/15,000` 非空；`train.csv` 的 bbox schema
也没有 patient laterality。此前 3,537 例子集已有同样的 `0/3,537` 结果，
全量扫描排除了抽样偶然。

因此 bbox 的左/右 pixel hemifield 不能自动当作患者左/右。更根本的是，
水平像素翻转 `g(x)` 有两个互斥解释：

1. **display flip：** 只是改变显示约定，患者侧别不变；预期 side token
   不应反号。
2. **patient reflection：** 假设患者解剖被镜像，side token 应反号；但心脏、
   胃泡、主动脉、device 与 burned-in marker 一起变成不自然组合，已不是
   finding-preserving 的临床等价图像。

同一个输出变化可被两套相反真值解释。因此

\[
f_c(gx)=f_c(x),\qquad s_c(gx)=-s_c(x)
\]

不是一个可由 VinDr flip 实验证伪的临床命题。double flip、marker mask、
identity-invariance 或 norm-matched patch 都不能补回缺失的 patient-side
ground truth；它们只能证明图像变换实现正确，不能证明医学构造正确。

### F7 — 当前数据访问阻塞：CRITICAL

当前本地可用 VinDr 没有 patient-side 标签。VQA-RAD 中几个明确 left/right
问答只能做 canary：样本太少、没有系统性 paired design，也不能支撑
layerwise/causal 结论。本地 MedHEval 虽有 1,677 张 MIMIC 子集图像，关联
报告至多提供 silver laterality；自动抽取不能单独定义正式真值。

找到的正式底座均不是“当前即得”：

| 数据 | laterality substrate | 规模/质量 | 当前阻塞 |
|---|---|---|---|
| [PadChest-GR](https://arxiv.org/abs/2411.05085) | finding sentence、patient-side/location label、finding bbox；每个 positive finding 最多两位 reader boxes | 4,555 studies；7,037 positive findings；location 表含 right 845 images/1,207 boxes、left 645/896、bilateral 419/929 | 官方数据需申请；本地未发现；先验文献仍不提供真正同一患者的 opposite-side counterfactual |
| [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) | 29 个带 `left/right` 的 anatomy objects，与 report-derived attributes 构成 object–attribute relations；500-patient manual gold | gold 中 left/right lung relations 分别 1,453/1,436，但具体 finding×side 的有效数量尚未审计 | PhysioNet credentialing、CITI、DUA；底层 MIMIC images 也需授权；silver 不得单独作真值 |
| [MS-CXR](https://physionet.org/content/ms-cxr/1.1.0/) | radiologist-verified finding phrase 与 bbox；phrase 可含 location | 1,162 pairs/8 findings，但单类仅 46–333；未发布专门 side-stratified counts | 同样要求 PhysioNet credentialing/CITI/DUA 与 MIMIC images；不保证两侧数量充足 |

PadChest-GR 是最小且最干净的正式候选，因为它直接给 finding-level location
与专家 boxes，而不是从图像 x 坐标猜 patient-side。Chest ImaGenome gold
可作为 anatomy-object 复现集；MS-CXR 只适合作补充，不能在未审计 side counts
前当主底座。

## 裁决

**REJECT CFED。** 两个 CRITICAL flaw 已触发 short-circuit：不再给该 idea
打分、调阈值或做防御性改名。原定“VinDr、无人工、2–4 GPU-hours”的组合
不能维持；继续运行只会得到无法解释的 flip sensitivity，不会得到临床
reference-frame mechanism。

## 允许的 Pivot：Anatomy–Finding Conjunctive Binding

若后续取得 PadChest-GR 或 Chest ImaGenome/MIMIC 的合规访问，研究问题改为：

> 在**真实临床图像和真实 patient-side 标注**上，模型是否分别保留 finding
> identity 与 anatomy side，却在生成时错误组合二者？

这不是 CFED 的“修补版”，因为它删除 synthetic flip 等变性，改为真实数据
上的 conjunction/binding 问题。最小设计如下：

1. 只用专家确认的 `(finding, patient-side, bbox/anatomy)`，按 patient/study
   划分 dev/test；自动 report parser 只做检索，不能定义 gold。
2. 对同一 finding 构造真实 left 与真实 right 的跨患者 matched sets；匹配
   view、finding size/severity、共病数、device、报告长度与可见 marker。
3. 每层独立解码 finding、side 与 conjunction。关键 prediction 不是“flip 后
   应反号”，而是 `finding AUROC` 和 `side AUROC` 均高、`joint accuracy`
   额外下降，且输出 wrong-side error 被该 conjunction residual 预测。
4. 因果测试只能在真实 opposite-side matched images 之间做 activation
   interchange，并要求 side 改变而 finding/polarity/claim count 不变；若无法
   构造匹配对，结论降级为相关性，不包装为 causal binding。
5. OE 端固定 finding identity、polarity、certainty、claim 数和长度，只允许
   side attribute 更正；报告 hallucination 与 omission 必须分开报告。

### 数据 gate（通过前不运行 GPU）

- 已获得合规图像与 annotation 文件，而非只有论文或 image IDs；
- 至少 3 个 unilateral findings，每个 finding 每侧 `>=100` 个 patient-disjoint
  cases；双侧、side-unspecified、dextrocardia/situs 与不可判定病例单列；
- 至少一个专家 gold test，不用 silver parser/LLM judge 独立定义真值；
- patient-side 与 display coordinate 的映射由数据说明或专家标注给出；
- 预注册 matched-set balance、prompt paraphrase、text-only、marker-mask、
  random/norm-matched patch 与 image-cluster bootstrap。

### 成本判断

- **当前：** 不满足“无新增人工、2–4 GPU-hours”，因为正式数据尚未在本地，
  access、下载、schema/count audit 与 gold linkage 都是前置工作。
- **取得 PadChest-GR 后的 admission：** 可用已有专家标注，不需新增人工；
  约 400–600 个平衡病例、两个模型的 teacher-forced behavior/readout screen
  有希望控制在 `2–4 GPU-hours`。
- **足以写机制论文的验证：** layerwise probes、真实配对 causal interchange、
  第三模型和 OE fixed-content 验证不能诚实承诺在 2–4 GPU-hours 内；预估应按
  `8–16 GPU-hours` 加数据准备，并保留医生复核需求。

因此当前执行决定是：**CFED 从队列删除；Pivot 仅进入 data-gate，不进入
GPU-gate。** 若拿不到 PadChest-GR/Chest ImaGenome gold，或 side×finding
计数未达门槛，就彻底放弃该分支，继续 AR-SoS/其他 problem-first 方向。
