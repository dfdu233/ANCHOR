# Next Goal：从传感器接口恢复被丢弃的病例证据

## 冻结目标

在不影响现有 baseline 队列的前提下，发现并验证一种 **training-free、单次前向、开放式可用** 的医学 VLM 幻觉缓解方法，先解决一个明确子问题：

> 标准 8-bit 灰度渲染丢失了原始医学传感器中的病例证据，导致模型生成图像不支持的阳性 finding；能否在保持标准显示图不变的同时，把被丢弃的信息编码进预训练 RGB 接口的冗余方向，从输入源头减少这种虚构？

这是一项算法工作，不以评测、风险预测、拒答、删 claim、缩短回答、阈值校准或医生复核作为贡献。

## 最小方法对象

设原始高位医学图像为 `x`，标准临床显示图为

```text
q = R(x),
```

其中 `R` 包含固定窗、归一化和 8-bit 量化。令 `r=x_bar-q` 为标准接口丢掉的残差信息。普通 VLM 输入把同一张 `q` 复制到 RGB 三通道；候选方法改为

```text
E(x) = q·1 + alpha·r·v,
```

其中 `v` 位于标准亮度算子的零空间，因此

```text
w^T E(x) = q.
```

也就是说，人和旧系统仍看到原标准图，模型却可通过 RGB 冗余通道接收额外传感器信息。

`v` 不用标签学习，而从冻结视觉塔第一层的通道几何中闭式选择：

```text
v* = argmax_{||v||=1, w^T v=0} ||A v||^2,
```

其中 `A` 是第一层 patch 投影的有效通道算子。这是受零失真约束的最大可读载波方向。`alpha` 只由 clipping 容量解析确定，不在测试标签上调参。

## 可声称与不可声称

若成立，核心 insight 是：

> 一部分医学 VLM 幻觉不是 decoder 忘了证据，而是 scientific sensor 到 web-RGB VLM 的接口先丢了证据；解法应是零失真的接口编码，而非继续修改 logits。

线性零空间、Rayleigh quotient 和可逆编码本身都是标准数学，不作为理论首创。论文新颖性必须来自跨模型实证规律和一个以前未被利用的 sensor-interface mismatch。

## 三阶段致死门

### L0：额外信息是否真是临床信息

只用 CPU/冻结视觉塔，比较：

- 标准灰度 `q`；
- 真实 residual 编码；
- 同图空间打乱 residual；
- 跨病例 residual；
- 等能量随机 chroma；
- 灰度方向 residual；
- 已证伪的多窗输入。

必须在两个冻结视觉塔、image-disjoint holdout 上同时满足：

- 相对标准图 macro AUROC `>= +0.02`，95% image-bootstrap CI 排除 0；
- 真实 residual 显著优于所有 placebo；
- 增益不能只来自 scanner/site identity；
- 分别报告 0/3 reader-vote specificity 与 3/3 sensitivity。

失败即关闭“低位 residual”载荷，不调标签阈值救结果。

### L1：是否直接纠正 VLM 错误

仅 L0 通过后，最多使用 2 小时 GPU，在 Huatuo 和 Hulu 上做 CE canary：标准图与编码图使用完全相同 prompt、解码和 token budget。

必须同时满足：

- false-positive hallucination 相对下降 `>=20%`；
- false negative 增加不超过 `1pp`；
- clear-case accuracy 下降不超过 `1pp`；
- 方法关闭态 32/32 token-exact；
- 真实 residual 优于 spatial-shuffle、cross-image 和 equal-energy chroma，而非只产生颜色 OOD 工作点漂移。

若只提高阳性 recall，则只能定位为遗漏恢复，不能声称 hallucination mitigation。

### L2：是否可用于开放生成

仅 L1 通过后，运行固定内容预算的 OE abnormality listing 和报告生成：

- 固定阳性 claim 数 `K` 或 matched-coverage；
- 不允许删句、拒答、统一 hedge 或缩短报告获益；
- claim-level FP 相对下降 `>=20%`；
- omission 不增加；
- 至少两模型、两任务复现；
- 报告长度、claim 数、阴性率、拒答率全部配对报告。

## 公式级碰撞基线

必须直接比较并明确区别于：

- 多窗 / pseudo-RGB 医学输入；
- bit-plane 与 high/low-byte 模型；
- chroma steganography / data hiding；
- learned input adapter；
- VCD、SECOND、视觉证据 prompting 和外部 expert verification。

方法的独特组合必须保持为：原始高位传感器数据、标准显示精确保留、丢失 residual 的模型可读零空间编码、冻结 VLM、一次前向、开放生成纠错。

## 失败后的唯一合法转移

若 residual L0 失败，不回到层融合、风格/DG、NCD/DID、attention mask、RAG、普通 crop、多视图一致性或阈值校准。只允许寻找另一个满足以下条件的 sensor-level 载荷：

1. 它在标准 VLM 输入前被确定性丢弃；
2. 它是病例特异而非人群先验；
3. 可在保持 legacy observation 不变的约束下传输；
4. 有 matched placebo 可证明模型读取的是临床结构而非接口异常。

没有通过 L0 的载荷不得进入 GPU，没有通过 L1 的方法不得命名，没有通过 L2 的方法不得作为论文主线。

## 最终交付

1. 公式级碰撞表；
2. 可复现编码实现及关闭态一致性测试；
3. L0/L1/L2 完整结果与 bootstrap CI；
4. 至少一个真实输入—输出病例，展示标准图产生虚构、编码图在不减少内容预算下纠正；
5. 若三门全部通过，形成“接口诱发幻觉 + 零失真传感器编码”的论文主线；否则继续探索机制不同的 sensor-level 载荷，不以负结果评测论文代替算法目标。
