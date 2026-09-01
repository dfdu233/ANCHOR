# Selection–Reuse Inflation：端到端致死实验预注册

日期：2026-08-12。确认标签只用于下述冻结分析；不因结果换窗口、claim数或阈值。

## 问题与数学背景

许多局部视觉方法分两步工作：先从多个病种和区域中挑一个响应最大的候选，再把该
候选放大、裁剪或增强后交给同一模型确认。令 `A_i` 表示第 `i` 个候选用于“选择”的
分数，`B_i` 表示选中后最终解码器给出的“确认”分数，候选总数为 `M`，选择结果为

`J = argmax_i A_i`。

若无病灶时每个固定候选的 `A_i,B_i` 均值都是零，但二者相关系数为 `rho`，则在一个
理想化的高斯模型下：

`E[B_J] = rho * E[max_i A_i]`。

直观解释：从很多噪声中挑赢家后，同一模型的第二次判断会部分保留这次偶然的高值；
候选越多，假证据可能越大。该等式是经典选择后偏差的背景，不是论文原创定理。可能
的新发现只能是：它是否真实解释 VLM 局部增强的 false-positive hallucination。

## 冻结数据与干预

- VinDr-CXR 固定七个 findings；每张测试图对七项均为三位读者一致 `0/3`。
- development global-null `n=166`，只估计每个finding和patch位置的均值/方差；
  confirmation global-null `n=62`，只在所有设置冻结后打开。
- Huatuo visual grid为`24×24`。固定`6×6`窗口，存在361个位置；先在同一claim下
  嵌套增加区域数`R=16,64,361`，再在完整361区域上增加claim数`K=1,7`。
- `selected`：按development冻结的视觉方向选取最高标准化窗口；`random`：同claim、
  同面积的确定性随机窗口；`full`：未经裁剪的原图。
- 三个变体使用完全相同问题、模型和解码；全部真实标签为No。四个搜索配置共
  `62×4×3=744`次单步评分。

## 唯一主门

同时满足才为 GO：

1. `K=7`时 selected-minus-random 最终 margin 的image-bootstrap 95% CI下界大于0；
2. 固定同一claim时，该差值从`R=16`到`R=361`的增长，95% CI下界大于0；
3. `K=7`时 selected-minus-random 的最终 false-positive rate差值，95% CI下界大于0。

任一失败即说明内部max响应没有稳定传入最终幻觉，关闭 Selection–Reuse 主线。通过也只
确认机制；后续必须在第二模型、多个局部方法、自然图像小目标和matched-Yes-rate上复现。

## 方法晋级条件

只有机制门通过，才测试完整流水线 null calibration：对每个局部方法在development阴性图
上完整重放“生成候选—选择区域—重新解码”，用经验rank校准最终响应。校准必须使阴性
FP不再随搜索空间增长，同时清晰阳性recall下降不超过1pp；OE保持claim数固定，一删一补。

边界：裁剪会改变图像分布，因此随机同面积crop是必需placebo；若selected和random同时
相对full大幅漂移而二者无差异，只能解释为crop OOD，不是选择—复用偏差。
