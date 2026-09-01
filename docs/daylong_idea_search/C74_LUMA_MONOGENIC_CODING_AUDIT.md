# C74 Luma-Preserving Monogenic/Riesz Coding：碰撞与接口审计

审计日期：2026-08-13  
范围：公式级文献检索 + 32 张真实 VinDr DICOM 的 CPU 接口审计；未运行、暂停或修改任何 GPU baseline。

## 结论

**不把 C74 升为主方法，也不排 GPU。** 它在接口上可实现且数学形式很漂亮，但核心临床表示是经典 monogenic signal 的线性换基；本地审计又证明它几乎不恢复 resize 丢失的信息。剩余新意只有“保持一个指定 luma functional、把经典 Riesz 两分量塞进冻结 RGB VLM 的 chroma 子空间”，不足以支撑 ICLR oral 主贡献。

## 1. 候选公式与必要背景

二维灰度图 `f` 的一阶 Riesz transform 是两个方向分量：

```text
F(R_j f)(ω) = -i ω_j / ||ω|| · F(f)(ω),   j=1,2.
```

它可理解为不会随频率线性放大的、归一化方向导数。经典二维 monogenic signal 恰好是三分量对象：

```text
M(f) = (f, R_1 f, R_2 f).
```

C74 选择 RGB luma 权重 `w`，以及 `ker(w^T)` 的正交基 `v1,v2`，构造

```text
X = f·1 + α[(R_1 f)v1 + (R_2 f)v2],
w^T v1 = w^T v2 = 0,  w^T 1 = 1.
```

因此连续浮点下 `w^T X=f`：普通灰度显示被保留，两个 Riesz 方向放在 chroma 平面。

## 2. 数学性质中哪些是真的，哪些不能夸大

### 2.1 真性质

- `M(f)` 与 `X` 之间只是固定可逆 `3×3` 线性换基（无 clipping/量化且 `α≠0`）。
- `R=(R1,R2)` 对二维旋转是向量协变的；整体 Riesz 向量满足标准 `L2` 等距性质。
- 连续浮点、指定线性 luma 下，`w^T X=f` 精确成立。

### 2.2 不能声称的性质

- **没有新增观测信息。** `R1f,R2f` 都是 `f` 的确定函数；它们只是重新表达已经在输入中的结构。
- **不是自动的强度不变性。** 对 `af+b`，零频以外有 `R_j(af+b)=aR_jf`；只有进一步形成 local phase/幅值比值时，才可能消去全局对比尺度。C74 原始分量本身随对比度线性变化。
- **不是高分辨率细节恢复。** 若先 resize 成 `f` 再算 Riesz，它不可能重建 resize 的 nullspace；理想带限各向同性 resize 与 Riesz Fourier multiplier 基本可交换。
- **不是 encoder 表示不变。** 保持人选的 BT.709 线性 luma，不等于保持 sRGB 感知亮度，也不等于经过视觉塔 channel normalization 后仍是 null direction。

## 3. 直接文献碰撞

### 3.1 公式对象已是经典对象

- [Monogenic Wavelet Scattering Network](https://arxiv.org/abs/2202.12491) 明确把一阶层写成 isotropic component 加 vertical/horizontal Riesz components，即同一个 `(f,R1f,R2f)` 三分量表示。
- [Riesz Networks: Scale-Invariant Neural Networks in a Single Forward Pass](https://link.springer.com/article/10.1007/s10851-024-01171-4) 已把 Riesz 低层特征作为固定基函数送入神经网络，并系统使用其 scale equivariance。
- 2026 ICLR 的 Riesz neural-operator 工作也使用 Riesz components 作为网络 channel-mixing 的基础对象；因此“Riesz + neural network”不是新组合。

### 3.2 医学影像中已经直接使用

- [Chest X-ray image phase features for improved diagnosis of COVID-19](https://pmc.ncbi.nlm.nih.gov/articles/PMC7794081/) 从 CXR 的 monogenic signal 提取多个局部相位/能量图，再与原 CXR 一起交给 CNN。
- [Fusing learned representations from Riesz Filters and Deep CNN for lung tissue classification](https://doi.org/10.1016/j.media.2019.06.006) 已在肺部 CT 中联合 Riesz texture signatures 与 deep CNN features。
- [Wavelet Guided 3D Deep Model to improve Dental Microfracture Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11017217/) 明确定义 `(f,R1f,...,Rdf)`，用 phase image 强化细小裂纹后输入 U-Net。
- [Image Biomarker Standardization Initiative filter report](https://pubs.rsna.org/doi/10.1148/radiol.231319) 将 Riesz transformations 与 wavelet/Gabor/LoG 并列为医学 radiomics 的标准滤波对象，同时提醒其跨实现复现仍困难。

这些文献不一定使用“BT.709 luma-null RGB + 冻结 VLM”，所以不是系统入口的逐式完全相同；但它们已经覆盖了 C74 想声称的**临床表示与神经网络结合**。剩余 delta 是接口包装，而非新的医学证据基本单元。

### 3.3 与仓库既有 C50/C54 的关系

C50 已完整审计 `X=f1+payload·v, w^Tv=0` 的 sensor-nullspace coding：第一层可见不代表后层理解 payload。C74 只是把 payload 从量化 residual 换成经典 `(R1f,R2f)`：

```text
C50: one residual scalar field -> one chroma direction
C74: two deterministic Riesz fields -> the full two-dimensional chroma plane
```

它比 C50 的 channel usage 更自然，但不能绕过 shared-codebook 问题；冻结自然图像 VLM 从未被要求把这种方向色解释为临床结构。

## 4. 真实 CPU 接口结果

脚本：`anchor/corrected_sgta/audit_monogenic_luma_interface_v1.py`  
结果：`corrected_runs/daylong_idea_search_v1/monogenic_luma_interface_v1/result.json`

设置：排序后 32 张真实 VinDr train DICOM；0.5/99.5 percentile render；512 工作分辨率、336 模型分辨率；BT.709 luma；`α∈{.02,.05,.10}`。

### 4.1 编码本身可实现

- 连续 luma 最大误差约 `2.2e-16`。
- 直接 8-bit round 后，32 图所有 α 的 luma mismatch 均为 `0`。
- `α=.05` 时 anti-clipping 后平均保留 `95.23%` 请求 payload energy；量化 payload 与连续 payload 平均相关 `.9886`。

因此失败原因不是“RGB 装不进去”。

### 4.2 但它没有带回 resize 丢失的信息

比较：

```text
A = Riesz(resize_336(f_512))
B = resize_336(Riesz(f_512)).
```

32 图结果：

| 量 | 均值 | 最差 |
|---|---:|---:|
| `corr(A_R1,B_R1)` | `.99961` | `.99842` |
| `corr(A_R2,B_R2)` | `.99879` | `.99770` |
| joint relative L2 | `.03792` | `.05907` |

这是 Riesz 与各向同性 resize 基本可交换的实证表现。C74 的 payload 几乎完全由已经 resize 到 336 的灰度图确定；它不是高分辨率 lesion packet。

### 4.3 OOD 幅度不小

即使 `α=.02`，平均 `73.55%` 像素至少一个 RGB channel 改变；`α=.05` 为 `92.48%`。这不是局部小提示，而是全图色度重编码。若模型输出变化，必须优先怀疑颜色 OOD/criterion shift，而不能直接归因于临床结构。

## 5. 严格判断

| 门 | 结果 |
|---|---|
| 数学自洽 | PASS |
| RGB 有限接口可实现 | PASS |
| 新临床信息/高分辨率恢复 | FAIL |
| 与 monogenic medical-CNN 区分 | FAIL |
| 与既有 luma-null sensor coding 区分 | 仅 payload 选择不同 |
| 值得抢占 baseline GPU | NO |
| ICLR oral 主贡献 | NO |

因此不做 VLM generation canary。若以后仅把它作为低成本 preprocessing baseline，可与 original-gray、Sobel/LoG、wavelet pseudo-RGB、spatial-shuffled Riesz、equal-energy random chroma 一起报告；但不能包装成新主算法。

