# C50 High-Bit Chroma Residual Coding：公式级碰撞审计

审计日期：2026-08-13  
范围：公式级文献与本地 substrate 审计；未运行 GPU，未修改或中断 baseline。  
候选：把高位 DICOM 经固定窗得到的连续灰度 `x` 分解为标准 8-bit 显示值 `q` 与量化残差 `r=x-q`，再把 `r` 编入 RGB 的亮度零空间：

```text
X = q·1 + α(q,r) r·v,      w^T v=0,  w^T 1=1.
```

其中 `w` 是冻结的 RGB luminance 权重，`v` 是一条 chroma 方向。目标是在一次前向、无训练条件下保持显示亮度，同时让冻结 RGB patch embedding 读取常规灰度复制丢弃的低位信息。

## 1. 先把方法还原成标准数学对象

### 1.1 这是两通道信号到 RGB 的固定线性复用

忽略防 clipping 的自适应 `α` 时，

```text
X = [1, αv] [q, r]^T.
```

因此 C50 在编码论上就是：把两张单通道图 `q,r` 经固定 `3×2` mixing matrix 填入三个 RGB channels。若第一层 patch embedding 是线性算子 `A`，则

```text
A(X)=A(1)q + αA(v)r.
```

它精确等价于一个接收 `[q,r]` 的两通道线性层，其两组有效权重分别为 `A(1)` 和 `αA(v)`。所以：

- `A(v)≠0` 只保证 residual 在第一层**可见**；
- 它不保证 `A(v)` 把 residual 解读为病灶，而不是异常颜色纹理；
- `A(v)=0` 或后层抑制该 OOD 方向时，额外信息虽被像素编码，却对模型不可达。

“利用闲置 RGB 通道携带另一个灰度信号”不是新编码对象。C50 的可能新意只能来自一个更窄的接口假设：**自然图像预训练形成的 chroma filter 恰好能在零训练下解释医学量化残差。**

### 1.2 两个初等性质成立，但都不是新定理

在浮点、线性 RGB、无 clipping 条件下，

```text
w^T X = q,
```

即选择的线性 luminance 完全不变。又因为 `v⊥w`，只要 `α≠0`，便可从编码图恢复

```text
q = w^T X,
r = v^T(X-q·1)/(α||v||²).
```

所以编码在 `(q,r)` 上可逆。这只是直和分解/零空间 side-channel 的直接性质；在颜色编码、隐写与辅助通道传输中是标准数学，不能作为论文理论贡献。

如果输出还要量化成 8-bit RGB，令每通道舍入误差不超过 `Δ/2=1/510`，则 residual 重建误差至多

```text
|r_hat-r| ≤ sqrt(3)Δ/(2|α| ||v||).
```

因此只有当 `α` 足够大且不触发 clipping 时，低位信息才真的穿过 PNG/PIL 输入接口。

### 1.3 “亮度不变”只对指定线性函数成立

实际输入通常经历：sRGB gamma、RGB 裁剪/舍入、各通道不同的 mean/std normalization、resize 与 patch projection。若标准化为

```text
N(X)=diag(1/σ)(X-μ),
```

残差方向变为 `diag(1/σ)v`，不再是视觉塔坐标中的 luminance-null。真实人眼感知亮度也不是 gamma-compressed RGB 上的简单 `w^T X`。所以 C50 的严谨表述只能是：

> 编码保持预先指定的线性 luma functional；它不保证感知等色，也不保证模型不受 common-mode 色彩偏移。

为避免 clipping，必须冻结一个 label-blind bound，例如

```text
|αr| ≤ min_j {q/|v_j|:v_j<0, (1-q)/v_j:v_j>0}.
```

边界像素可用 chroma 容量趋于零；标准窗外已经被截断的高位原始信息也无法由 `r=x-q` 恢复。

## 2. 与 C07 Diagnostic Channel Coding 的关系

C07 的典型输入是

```text
X_C07 = [g_{θ1}(raw), g_{θ2}(raw), g_{θ3}(raw)],
```

三通道分别承载不同 window/contrast view。C50 则是

```text
X_C50 = q·1 + αr·v,
r = x-Q_8(x),
```

其中一个分量保持标准显示图，另一个分量只携带**同一窗内的 sub-8-bit quantization residual**。

二者的本质差异：

| 维度 | C07 | C50 |
|---|---|---|
| 额外信息 | 不同窗的低频/对比重映射 | 同一窗内被 8-bit rounding 丢失的低位残差 |
| 对标准图的约束 | 无；各通道均可改变 | 指定线性 luminance 精确保持为 `q` |
| 输入自由度 | 通常三张 windowed views | 两个标量场 `[q,r]` 的固定 RGB embedding |
| 既有直接近邻 | WindowNet、多窗 CT pseudo-RGB、MedGemma 多窗/多图接口 | bit-plane / high-low byte、chroma data hiding；未找到同样的 frozen-VLM residual interface |
| 失败是否互相证伪 | 五 render 无增益不能直接证伪低位 residual | 若 residual 无 label 信息，则不能靠改窗复活 |

因此 C50 **不是 C07 完全相同的实验**；C07 的本地 five-render 负结果不能直接关闭它。但在更高层数学上，两者都属于“把灰度源的互补表示复用到 RGB 输入带宽”，C50 不能重新声称 Diagnostic Channel Coding 或 rate–distortion 本身是新贡献。

## 3. 公式级与系统级碰撞

### 3.1 Pseudo-RGB / multi-window medical imaging：系统目标高度重叠

- [WindowNet](https://www.mdpi.com/2313-433X/9/12/270)直接研究 CXR 从高 bit depth 降到 8-bit 会隐藏细微诊断特征，并学习多个 window；论文报告 12-bit 优于 8-bit，且多窗分类优于普通基线。
- CT 中将 brain/subdural/bone 等三个窗口堆成 RGB-like image，再使用 ImageNet-pretrained CNN，是成熟做法；例如 [MIDL 2020 的 ICH 模型](https://openreview.net/pdf?id=1IoPbyuPFT)明确采用三窗三通道。
- 医学 foundation model 已支持原始高维影像或多图输入；因此“高位医学图像不应先压成单个 8-bit 灰度”本身不是新问题。

这些工作没有保持一个固定 luma，也没有专门编码 quantization residual；所以不是 exact formula collision。但它们使“复用 RGB 三通道保留医学灰度细节”成为拥挤的工程邻域。

### 3.2 Bit-plane / high-low byte features：信息分解已被覆盖

- [Hybrid deep features from bit-plane maps for CXR classification（2024）](https://doi.org/10.1016/j.jrras.2024.101024)明确从 X-ray bit planes 提取与原图互补的深特征。
- [Coronary MRI bit-plane slicing](https://pubmed.ncbi.nlm.nih.gov/37936441/)把不同 information planes 分别送入弱分类器再融合。
- 高位灰度拆成 high/low byte 或逐 bit-plane progressive coding，是标准无损/渐进图像表示；[Progressive Transmission of Medical Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC8099520/)也直接以医学图像 bit planes 和 residual bit-plane 为对象。

C50 的 `q+r` 与 high/low-plane 分解属于同一家族。剩余差异仅是：不用训练专门分支，而把 `r` 旋转进现有 RGB chroma 子空间。

### 3.3 Chroma side-channel / steganography：零空间承载信息不是新数学

- [Hiding Data in Colors](https://arxiv.org/abs/2201.07444)把隐藏数据映射到灰度 host 的 color information，本质就是以颜色自由度承载不改变主要灰度内容的 payload。
- reversible color-to-gray、watermarking 与 HDR residual coding 广泛使用 luminance/chrominance 分离，在 luma/legacy layer 外放置 residual side information。
- chroma-only adversarial examples 进一步说明“人眼亮度近似不变”绝不意味着 DNN 表示不变；冻结网络可能对这种色度方向非常敏感，却以非临床方式响应。

这些方法通常训练专用 decoder，或目标是安全传输/攻击而非诊断。因而它们直接覆盖编码性质，不直接覆盖“冻结医学 VLM 即为 decoder”的应用假设。

### 3.4 Pretrained grayscale adaptation：固定通道变换是常规接口处理

常用做法包括：灰度复制到 RGB、修改第一卷积为单通道、把多个医学窗/切片放进 RGB、或训练一个浅层 channel adapter。由于

```text
A(q·1+αrv)=A(1)q+αA(v)r,
```

C50 只是无需改权重的固定 input adapter。其优势是 zero-training 和一次前向；其代价是没有任何 learned alignment。若后续为了让模型读懂 `r` 而训练 adapter、选择 `v` 或按标签优化 `α`，方法会退化为普通 pretrained-input adaptation，失去 training-free 核心。

## 4. 本地自然性与可执行性

本地 VinDr DICOM 确实提供所需源信息。对 `/workspace/vinbigdata/train` 前 1000 张按文件顺序做只读 header 审计：

```text
BitsAllocated: 16 for 1000/1000
BitsStored:    10-bit 6, 12-bit 680, 14-bit 242, 16-bit 72
Photometric:   MONOCHROME2 834, MONOCHROME1 166
```

因此 substrate 不是伪问题：样本绝大多数在存储上超过 8 bit，且本地标准 renderer 当前明确输出 `uint8 RGB`。WindowNet 的既有结果也支持“bit-depth reduction 可能损伤 CXR 分类”这一广义现象。

但尚未确认三个关键前提：

1. 在**最终 VLM 输入分辨率**做 float resize 后，`r` 是否仍有足够非零能量；
2. `r` 是否含 reader-confirmed finding 信息，而非量化噪声、设备噪声或 scanner fingerprint；
3. 冻结 Huatuo/Hulu/LLaVA 的 `A(v)` 是否以可迁移的临床方向读取它。

高 bit depth 的存在只证明 payload 可构造，不证明该 payload 是诊断证据。

## 5. 是否存在不可约新性质？

### 已有/不可声称为贡献

- luma-null 编码；
- `(q,r)` 可逆；
- 用 RGB 空闲自由度传 side information；
- bit-plane / residual progressive representation；
- 多医学窗或 pseudo-RGB；
- frozen RGB backbone 接收非自然三通道输入。

### 尚未检索到机制等价工作的窄 delta

在记录的检索式下，未检索到以下四项同时成立的工作：

```text
raw high-bit medical image
+ exact base-luma preserving residual-to-chroma map
+ completely frozen RGB VLM (no trained decoder/adapter)
+ direct open-generation hallucination mitigation.
```

这不是新的编码定理，而是一个新的**接口假设**：

> 当医学灰度输入浪费了预训练 RGB 模型的 chroma response directions 时，可否把传感器低位 residual 放进这些方向，从而在不增加视觉 token 和前向次数的情况下增加病例证据？

如果它成立，最有价值的实证规律不是“颜色好用”，而是：冻结视觉模型存在一种 **channel-capacity mismatch**——输入源有高精度单通道，模型接口有低精度三通道；一个确定性 basis change 可以跨越该 mismatch。

不过，这一表述仍与 C07 的 rate–distortion 主旨相邻。仅有 accuracy gain 最多是优秀 preprocessing trick，不足以成为 ICLR Oral 核心。

## 6. 最大新颖性与科学风险

1. **数学是标准线性复用。** 审稿人可以准确概括为“把高位灰度的 residual 伪彩化后送给 frozen VLM”。
2. **颜色 OOD 可能只移动工作点。** 人眼 luma 不变不代表 VLM 不受 global chroma bias；必须排除 Yes-rate/length shift。
3. **低位不等于病灶。** detector noise、后处理和设备域都可能主要存在于 low bits；可能强化域 shortcut，而非临床信号。
4. **通用性受 raw source 限制。** PNG/JPEG benchmark 已永久丢掉原始低位，C50 只适用于保留高位传感器数据的医学/科学成像。
5. **跨模型方向不统一。** 不同视觉塔的 channel normalization 和 first-layer color filters 不同；固定 `v` 不保证跨架构有效。逐模型选 `v` 又像调参/adapter。
6. **若需要监督选择 `α(x)`，会破坏 training-free 公平性。** `α` 必须由动态范围、量化误差或 clipping 约束确定，不能在 test labels 上调。

## 7. 严格裁决

| 门 | 结论 | 依据 |
|---|---|---|
| 本地真实信息源 | **PASS** | VinDr 抽样 994/1000 为 12/14/16-bit，标准入口输出 uint8 RGB |
| 与 C07 是否完全重复 | **否** | C07 是多窗；C50 是同一窗的 sub-8-bit residual 且保持 base luma |
| exact method collision | 暂未检索到 | 未找到 frozen RGB VLM + luma-null high-bit residual + OE mitigation 的同构工作 |
| 数学新颖性 | **FAIL** | 两通道到 RGB 的固定 mixing、零空间 side-channel、可逆性均是标准编码 |
| 新接口/机制空间 | **条件保留** | 无训练地把 scientific-sensor precision 映射到 pretrained color directions 尚有窄 delta |
| 当前是否可称为方法 | **否** | 尚不知 residual 是否含病例标签信息，也不知 frozen VLM 是否临床读取 |
| 是否运行 GPU | **否，先 L0** | 先过信息存在性和非 OOD-response 门 |

最终判定：

```text
RETAIN FOR A STRICT L0 FATAL TEST ONLY.
Not an ICLR method yet; not a new coding theorem.
Do not run GPU before the L0 gate passes.
```

### 唯一合理的 L0 门

不使用 VLM generation，按最终视觉输入分辨率冻结以下处理：

1. raw DICOM modality transform、MONOCHROME1 修正、唯一固定窗；
2. float resize 后再分解 `x=q+r`，避免把 uint8 resize 差异混入 residual；
3. 冻结一个解析式 `v` 和由 clipping 上限确定的 `α`；
4. 比较 `base q·1`、`true chroma residual`、同图空间打乱 residual、跨图 residual、等幅随机 chroma、`q+βr` 灰度方向以及 C07 多窗；
5. 先测真实 residual 相对 shuffled/placebo 是否在 image-disjoint holdout 对 reader finding 提供条件增量，而不是只测 feature norm 是否变化。

建议的致死标准：在至少两个冻结视觉塔、多个合格 findings 上，真实 residual 相对 base 的 macro AUROC 增量至少 `+0.02`，image-bootstrap 95% CI 排除 0；并且真实 residual 显著超过空间打乱、跨图 residual 与等幅随机 chroma。任何一项失败都说明模型只是响应颜色 payload，而没有读取临床 low-bit structure，应立即关闭。

只有 L0 通过后，才值得做不超过 2 小时的 CE/OE GPU canary；该 canary 仍须固定 claim 数，证明 FP 下降不来自统一阴性、回答缩短或 omission 上升。

## 8. 检索记录

核心检索式包括：

- `high bit depth DICOM residual RGB channels pretrained model`
- `pseudo-RGB multi-window medical imaging ImageNet pretrained`
- `WindowNet chest X-ray bit depth learnable windows`
- `medical image bit-plane deep features classification`
- `pack 16-bit grayscale into RGB channels neural network`
- `luminance preserving chrominance side information encoding`
- `chroma side channel pretrained CNN steganography`
- `grayscale adaptation pretrained RGB medical imaging`

承载判断的工作核对了论文主页、正式出版页面或论文正文。当前裁决是“未检索到四项同时成立的机制等价工作”，不是证明绝对首创。

## 9. 追加致命审计：direct-float dominance 与可逆 RGB 先例

后续审计发现两个进一步降低 C50 方法价值的事实。

### 9.1 对开源 VLM，8-bit PIL 并不是模型约束

Huatuo/Hulu 的 patch embedding 最终接收浮点 tensor。若 wrapper 当前把 DICOM 先转为 `uint8 PIL`，可以直接把 float-resized 高位灰度 `x` 复制到三通道、应用冻结 processor 的同一 mean/std，再送入 patch embedding：

```text
pixel_values = normalize(repeat_rgb(x)).
```

这保留了 `x` 的全部浮点精度，不引入 chroma OOD，也不需要恢复 `r=x-q`。因此在允许访问模型输入 tensor 的常规开源设置中，direct-float 严格支配 chroma side-channel：

- 若 direct-float 相对 8-bit 没有增益，低位信息对当前模型无用；
- 若 direct-float 有增益，合理修复只是纠正 renderer，不需要新的 RGB 编码方法。

C50 只在输入被外部黑盒 API 强制限制为 8-bit RGB 文件时还有工程意义；这与“通用开源医学 VLM mitigation”目标不一致。

### 9.2 高位医学数值可逆映射到 RGB 已有直接先例

2025 年 *Color as a High-Value Quantitative Tool for PET/CT Imaging* 已将高位 CT/PET 的 HU/SUV 通过加权 color distribution 映射到 RGB PNG，并明确声称输出保留全部 raw values、可作为 raw DICOM input 使用。它虽然不是 luma-preserving frozen-VLM prompt，但已经直接覆盖“用 RGB bytes 携带高位医学灰度/定量值”的表示目标：

https://doi.org/10.3390/info16050352

所以即使严格 L0 通过，C50 也不能声称高位医学信息的 RGB 复用是新表示；剩余 delta 仅是特定 luma 约束和 frozen VLM 应用。

### 9.3 更新后的结论

```text
C50 may finish as a substrate experiment, but is no longer eligible as the
main general/ICLR-Oral method even if L0 is positive.
```

保留正在运行的 CPU L0 是为了回答“低位 residual 是否包含可读临床结构”这一事实问题；不因正结果自动启动 GPU。
