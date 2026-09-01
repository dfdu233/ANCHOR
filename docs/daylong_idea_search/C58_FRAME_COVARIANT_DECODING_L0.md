# C58：Frame-Covariant Decoding——空间属性幻觉的坐标编译

## 当前裁决

**条件 GO：允许进入跨模型与自然开放生成 canary；尚不是 ICLR 方法。**

它是当前搜索中第一个同时满足以下三点的正向 L0：

1. 直接纠正一个明确 hallucination 子问题，而不是预测错误或改变评测；
2. finding 集合和非空间文本逐字保持，不靠少说、拒答、缩短或统一阴性；
3. 方法由本地已确认现象自然推出：模型能正确读取屏幕位置，却把屏幕参考系误说成患者参考系。

最大风险也必须提前写清：当前只在 Huatuo、13 个可解析样本、且 prompt 已给出两个 findings 的 canary 上成立；坐标搬运本身是标准数学，论文新颖性只能来自“reference-frame hallucination 是可分离且可编译的 VLM 生成错误”这一跨模型、跨模态规律。

## 1. 自然现象

在同一批 VinDr 双侧不同 finding 病例中：

- 直接要求 Huatuo 说患者左右，13 个可解析回答仅 1 个正确；
- 要求它只说显示图上的左右，13 个可解析回答有 12 个正确；
- 直接回答中的错误几乎全是完整镜像，而非随机绑定。

最简单的真实例子：

```text
真值：Nodule/Mass 在患者左侧；Lung Opacity 在患者右侧。
原回答：The nodule or mass is on the patient's right and the lung opacity is on the patient's left.
```

模型没有丢掉两个 finding，也没有随机乱说；它给出了正确的**屏幕左右**，却用“patient's left/right”这个语言标签输出。标准胸片采用 radiological display：屏幕左侧对应患者右侧，屏幕右侧对应患者左侧。

这个错误更像编译器把一个坐标类型错当成另一个坐标类型，而不是视觉感知失败。

## 2. 最小数学对象

把临床 claim 分成两部分：

\[
c=(a,s),
\]

其中 \(a\) 是 frame-invariant atom，例如“pleural effusion”；\(s\) 是 frame-covariant attribute，例如 left/right。

胸片左右变换构成二元群 \(G=\mathbb Z_2=\{e,g\}\)。它对 claim 的作用为

\[
\rho(g)(a,\mathrm{left})=(a,\mathrm{right}),\qquad
\rho(g)(a,\mathrm{right})=(a,\mathrm{left}),
\]

而 finding atom 保持不变：

\[
\pi_a\rho(g)c=\pi_a c.
\]

Frame-Covariant Decoding 让模型先在其实际读取的显示坐标中产生 claim，再用 DICOM / renderer 给出的几何变换 \(T\) 编译为患者坐标：

\[
D_T(x)=\rho(T^{-1})D_{\mathrm{display}}(x).
\]

### 两个直接性质

**内容守恒。** 编译器只作用于 spatial attribute，因此

\[
\pi_aD_T(x)=\pi_aD_{\mathrm{display}}(x).
\]

它不会增删 finding，也不会改变报告长度来伪造收益。

**重新渲染不变性。** 若同一病人图像又经过一个显示变换 \(h\)，显示几何变成 \(hT\)，且显示坐标预测随图像一起变换，即 \(D_{\mathrm{display}}(hx)=\rho(h)D_{\mathrm{display}}(x)\)，则

\[
D_{hT}(hx)
=\rho((hT)^{-1})\rho(h)D_{\mathrm{display}}(x)
=D_T(x).
\]

直观上：无论 viewer 怎样翻转或旋转，编译后的患者坐标报告都相同。这个等式是标准群作用/坐标变换，不应包装成新数学定理；其价值是明确了 VLM 输出空间中哪些字段应变、哪些字段必须不变。

## 3. 与已有工作的碰撞

### 已占据的部分

- DICOM 标准已经定义 Patient Orientation 以及像素轴到患者解剖方向的映射；坐标变换不是新知识：[DICOM Common Image IE Modules](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.html)。
- EquiTune / frame averaging 已能把任意 pretrained model 包装成群等变模型，例如 \(|G|^{-1}\sum_g\rho(g^{-1})f(gx)\)：[Equi-Tuning, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25832)。
- SpatialVLM、SpatialCoT 等工作已经强调 coordinate canonicalization 或坐标—语言对齐；多数依靠训练或显式 reasoning。
- 医学 VLM 的左右混淆已被 Radiology 明确指出，且被解释为模型按 image frame 而不是 anatomical frame 理解 laterality：[Laterality: A Potential Pitfall, Radiology 2024](https://pubs.rsna.org/doi/10.1148/radiol.241421)。

### 尚未找到的窄差异

当前检索尚未找到以下组合的同构工作：

```text
冻结医学 VLM
+ 将自然语言 claim 类型分成 frame-invariant atom 与 frame-covariant attribute
+ 仅对 attribute 应用 DICOM acquisition/display group action
+ 一次原始生成后的确定性编译
+ finding/claim content 精确守恒
+ 直接评估开放生成中的 laterality hallucination
```

它与 EquiTune 的关键差别不是符号，而是操作边界：EquiTune 对多个变换输入做多次模型前向并平均整个输出；C58 只做一次 VLM 生成，再对有空间类型的输出字段作确定性 push-forward。它也不是普通“明确告诉模型左右定义”的 prompt，因为最终正确性不依赖模型执行坐标换算。

但这一差异目前仍可能被审稿人概括为“rule-based left/right post-processing”。只有扩展到一般 DICOM orientation group、自然 OE 多 claim、多个模型和不同医学模态后，才可能上升为新的 typed generation principle。

## 4. Cache-only L0

入口：

```bash
PYTHONPATH=/home/dbw/ANCHOR \
  /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python \
  anchor/corrected_sgta/analyze_frame_covariant_decoding_l0_v1.py \
  --input corrected_runs/daylong_idea_search_v1/binding_conservation_huatuo_n16/raw.jsonl \
  --output corrected_runs/daylong_idea_search_v1/frame_covariant_decoding_l0_v1/result.json \
  --bootstrap-draws 10000 \
  --seed 20260813
```

干预只做 whole-word `left <-> right`，其他字符不改。用把 left/right 都替换为同一 `<side>` 后的字符串完全一致，验证非 frame 内容守恒。

| 指标 | 结果 |
|---|---:|
| 原始样本 | 16 |
| 联合可解析 | 13 |
| 原回答 laterality accuracy | 7.69%（1/13） |
| 编译后 accuracy | 92.31%（12/13） |
| 配对提升 | **+84.62pp** |
| image-bootstrap 95% CI | **[+53.85, +100]pp** |
| 非 frame 内容保持率 | **100%** |

预注册 L0 门为：至少 8 个可解析样本；提升至少 20pp；bootstrap CI 下界大于 0；非 frame 内容保持率 100%。本实验全部通过。

## 5. 为什么还不能叫论文方法

1. prompt 已告诉模型两个 findings，因此只验证 attribute binding，没有验证自然 OE finding 发现；
2. 只有 Huatuo，可能是单模型错误习惯；
3. 只有标准 frontal CXR 的 \(\mathbb Z_2\) 左右变换；
4. 当前词级替换不能可靠处理“left-sided”“bilateral”“right greater than left”等复合表达；
5. 若自然报告中模型有时使用患者 frame、有时使用显示 frame，无条件搬运会把原本正确答案变错；
6. 数学是标准坐标 transport，新颖性不能来自公式本身。

## 6. 下一道致死门

只有下面两个 canary 都通过，才允许给方法命名或放量：

### L1-A：跨模型 frame separation

- 同一 image-disjoint 64 例在 Hulu 上复现；
- patient-frame 直接准确率与 display-frame 定位准确率差至少 20pp；
- transport 后 laterality accuracy 提升至少 20pp，image-bootstrap CI 下界大于 0；
- finding identity、claim 数和非 frame attributes 100% 保持。

### L1-B：自然开放生成

- prompt 只要求描述异常及位置，不提供 finding 名称；
- 对原报告与 compiled report 做同一 matched-claim 解析；
- 在原报告已经生成的 lateralized claims 上，laterality error 相对下降至少 20%；
- finding hallucination、omission、claim 数、报告长度完全不因编译变化；
- 显式 radiological-convention prompt 作为控制，必须证明收益不是一句 prompt 就能等价替代；
- 同时报告本来正确却被 transport 伤害的比例，要求不超过 1pp。

若 Hulu 不复现或自然 OE clear-case harm 超过 1pp，关闭该路线，不用阈值决定“何时交换”。因为一旦引入学习式 gate，它会退化为校准/错误检测，违背当前 goal。

## 7. 当前最诚实定位

> 医学 VLM 的一部分空间属性幻觉可能不是视觉证据缺失，而是模型把 display-frame 中正确的变量绑定，错误地标注成 patient-frame 语言。将 finding atom 与坐标 attribute 分型，再由 acquisition geometry 编译坐标，可以在不改变任何临床内容的前提下精确消除这类错误。

这是一个可信、简单且有真实正信号的子问题方法种子；它离 ICLR Oral 仍差跨模型、自然 OE、一般群作用与“不是 rule-based swap”的规模证据。

## 8. 2026-08-13 执行修正：自然 OE 不再使用无条件 swap

跨模型 runner 已升为内部 `frame-covariant-cross-model-v2`。旧版自然臂把
patient-frame 原回答直接左右互换，只能诊断模型是否系统性使用显示坐标，不能定义方法。
修正版使用两条清晰分开的路径：

1. native control：直接要求 patient-anatomical locations；
2. method arm：明确要求模型在 `screen-left/screen-right` 中完成自然异常描述，再由已知
   DICOM/radiological-display 的 `Z2` 作用只编译左右词。

自然臂现在还同时报告 native/method 对两个 reader-supported target findings 的提及率和
exact mention-set rate；正式门要求 method target recall 不得比 native 低超过 1pp。这样
screen prompt 若改变 finding 内容，不能被 laterality 指标掩盖。method arm 内部的
screen→patient compilation 仍要求非 frame 内容 100% 保持。

入口：

`anchor/corrected_sgta/run_frame_covariant_cross_model_v1.py`

回归测试：

`tests/test_frame_covariant_cross_model_v1.py`，当前 `4 passed`。该修正只完成代码与 CPU
分析门，尚未运行新的 GPU 样本。

## 9. 2026-08-13 DICOM 几何可信度审计

对 L0 的 16/16 个 DICOM 逐项检查发现，下列方向标签全部缺失：

```text
PatientOrientation
ImageLaterality / Laterality
ImageOrientationPatient
ViewPosition / PatientPosition
```

所以当前 VinDr 文件不能支持“每张图的变换直接由 DICOM tag 读取”这一表述。转换函数
`dicom_to_pil` 只做强度归一化与 `MONOCHROME1` 灰度反转，没有水平翻转。

同时生成 4x4 图像拼图做人可读审计：可清楚辨认方位字母的样本中，`R` 标记均在屏幕
左侧，`L` 标记均在屏幕右侧；其他无显式标记的图像也呈标准 radiological display，
但这不能替代逐图元数据证书。

因此可信度边界更新为：

1. L0 的 12/13 仍是支持“该样本遵循 display-frame convention”的强 canary，但不是
   metadata-certified 结果；
2. 下一轮 primary cohort 必须只纳入有显式 `L/R` marker 或独立 renderer orientation
   record 的图像；无标记样本只能作 secondary sensitivity analysis；
3. 部署算法必须从 viewer/DICOM pipeline 接收已知变换 `T`，不得从模型回答或疾病位置
   反推 `T`；否则会把待验证结论重新放进约束本身。

随后对冻结 seed `20260813` 的 64 例候选生成原始像素拼图，只做烧录字母与屏幕侧别
审计，不解释病灶。39/64 例具有清晰且方向一致的 `L/R` 标记，已冻结在：

`configs/frame_covariant_orientation_cert_v1.json`

runner 新增 `--orientation-certificate` 与 `--candidate-pool-size`；证书中的 `R` 必须在
screen-left、`L` 必须在 screen-right，否则 fail closed。建议首轮使用证书前 32 例，
剩余 7 例不在看过输出后补入。证书是单人 marker audit，因此仍需作为限制报告，但它已
消除“用无方向标签的全部 VinDr 图硬编码患者侧别”这一主要构造风险。测试现为
`5 passed`。
