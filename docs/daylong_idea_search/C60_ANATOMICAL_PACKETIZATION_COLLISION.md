# C60 — Anatomical Packetization 公式与系统碰撞审计

审计日期：2026-08-13  
资源边界：公式、文献和本地接口审计；未运行 GPU，未修改 baseline 队列。  
裁决：**作为新 hallucination mitigation 原语严格 NO-GO。**

## 1. 候选的最小形式

冻结 VLM 产生投影后的视觉 token

\[
V=(v_1,\ldots,v_n),\qquad v_i\in\mathbb R^d.
\]

一个小型 anatomy specialist 只给 patch 地址

\[
a_i\in\{\text{left lung},\text{right lung},\text{heart},\ldots\},
\]

不预测病种。设 \(R_r=\{i:a_i=r\}\)，\(e(r)\) 是目标 LLM 自己对区域名称
\(r\) 的词嵌入。候选把原来的 raster 序列改为

\[
\mathcal P_a(V)
=
\bigoplus_{r\in\mathcal R}
\left[e(r),\, (v_i)_{i\in R_r}\right].
\]

所有 \(v_i\) 的数值和数量均保持不变，只改变序列顺序并插入语言 header；然后
进行一次原生 prefill 和解码。

## 2. 保留 token 值不等于建立地址绑定

### 2.1 输出变化有三个不可分离来源

设冻结 decoder 为 \(F\)，原序列输出为 \(F(V)\)。候选响应是

\[
\Delta_a
=
F\!\left(H_a\oplus\Pi_a V\right)-F(V),
\]

其中 \(H_a\) 是 header embeddings，\(\Pi_a\) 是 anatomy-dependent permutation。
这个差值同时混合：

1. 新文字 prompt 的语义作用；
2. 视觉 token 被重排后的 1D position / RoPE 变化；
3. header 与视觉 token 的非线性交互。

因此即使准确率改变，也不能把变化唯一归因于地址绑定。

### 2.2 causal attention 没有 packet scope

对放在 header 后的 patch，第一层可以沿路径

\[
e(r)\rightarrow v_i\rightarrow y
\]

影响答案。但 causal mask 只表示能看见所有过去 token，并不表示只属于最近的
header。例如序列为

\[
[e(L),V_L,e(R),V_R,e(H),V_H],
\]

则 \(V_R\) 同时能读取 \(e(L)\) 与 \(e(R)\)，\(V_H\) 能读取三个 header。没有
block mask、segment embedding、显式 region token 或训练目标时，header 不具有词法
作用域。加入 begin/end 文本也不会从 KV cache 中删除之前的 header。

### 2.3 真正的地址对象至少应满足 packet-order invariance

如果每个 packet 的 header 与内容保持配对，只改变 packet 的排列 \(\pi\)，一个
区域集合接口应近似满足

\[
F\!\left(
\bigoplus_r[e(r),V_r]
\right)
\approx
F\!\left(
\bigoplus_r[e(\pi(r)),V_{\pi(r)}]
\right).
\]

标准 causal Transformer 不具备该性质：causal receptive field、1D RoPE 和位置偏置
均随 packet 顺序改变。反过来，若加入 block attention 或 segment IDs 来强制该性质，
方法便进入 attention-mask / region-token architecture 家族，也不再是当前的
training-free token packetization。

### 2.4 任意冻结模型上不存在 training-free binding 保证

存在三个同样兼容当前架构的冻结模型：

- 忽略所有 header 的模型；
- 只读取 header 的 bag、忽略 header 与 patch 配对的模型；
- 只对重排后 position pattern 响应的模型。

三者都能处理 \(\mathcal P_a(V)\)，但均没有地址绑定。因此使用目标模型自身词嵌入
只保证 header 位于已知语言空间，不保证对应 patch 被解释成该解剖区域。绑定能力必须
来自既有训练中的偶然泛化，或新增 region-language alignment；它不是这个操作的数学
性质。

## 3. 机制级碰撞

### 3.1 最直接的旧对象：region feature + language anchor

- **Oscar, ECCV 2020** 已使用检测到的 object tags 作为 anchor，把图像 region
  features 与语言语义对齐。候选把 object tag 换成 anatomy name、把 region feature
  换成一组原 patch，基本对象不变。  
  https://arxiv.org/abs/2004.06165
- **GPT4RoI, 2023** 把 RoI visual feature 与 language embeddings 组成
  interleaved sequence，并同时保留纯文字 region reference；其论文明确使用
  region-text alignment 和 spatial instruction tuning 建立该能力。  
  https://arxiv.org/abs/2307.03601
- **Groma, ECCV 2024** 把图像分解为可引用 region tokens，将位置与 region token
  绑定并插入语言 instruction。  
  https://arxiv.org/abs/2404.13013
- **Omni-RGPT, CVPR 2025** 同时把 Token Mark 注入视觉区域和文字 prompt，直接建立
  visual-text region connection。  
  https://arxiv.org/abs/2501.08326
- **VLX-Seek 1.5, 2026** 明确把候选区域变成 language-addressable region tokens，
  先做 region-language alignment，再做 perception instruction tuning；不存在目标时
  的低 hallucination 还依赖 hard-negative rejection training。  
  https://github.com/om-ai-lab/VLX-Seek

C60 唯一表面差异是不用训练、也不池化 region feature；但这正好删掉了上述工作用于
保证 region-language binding 的部分，而不是产生新机制。

### 3.2 医学中的 anatomy token 已是成熟对象

- **Finding-Aware Anatomical Tokens for Chest X-Ray Automated Reporting
  (2023)** 用解剖定位得到的局部 token 代替全局图像 token，并显式训练 token 携带
  finding，目标就是减少报告中的错误 finding。  
  https://arxiv.org/abs/2308.15961
- **Anatomy-VLM, WACV 2026** 先定位解剖结构，再做 region-specific matching 和
  global disease prediction；输入包括 patch tokens 与 anatomical tokens。  
  https://openaccess.thecvf.com/content/WACV2026/papers/Gu_Anatomy-VLM_A_Fine-grained_Vision-Language_Model_for_Medical_Interpretation_WACV_2026_paper.pdf
- **AnatomiX, 2026** 采用先识别 anatomy、再提取区域特征、最后交给 LLM 的两阶段
  CXR 理解框架。  
  https://arxiv.org/abs/2601.03191

所以按临床解剖重写视觉基本单位在医学端也已有直接覆盖。

### 3.3 与相邻 inference-time 方法的边界

- 它不是 **Set-of-Mark** 的像素 overlay，但二者都用可说出的 region label 提示
  冻结或既有 VLM；只把 marker 从像素移到 embedding sequence 是载体变化。  
  https://arxiv.org/abs/2310.11441
- 它不是 **ARCD** 的 token、attention、logit 三层 mask 和 contrastive branch，但若
  为了建立 packet scope 而引入 block mask，就会直接进入 ARCD 或 region-guided
  attention 家族。  
  https://ojs.aaai.org/index.php/AAAI/article/download/37620/41582
- 它也不等于 **Visual Evidence Prompting** 的病种专家证据；然而外部视觉模块把
  spatial metadata 变成目标 VLM 可读提示的系统角色相同。  
  https://aclanthology.org/2025.acl-long.205/

## 4. 本地 canary 可行性

### 技术上可实现

Huatuo 和 LLaVA-Med 的现有代码可在
\(prepare\_inputs\_labels\_for\_multimodal\) 展开后取得完整
\(inputs\_embeds\)。本地已有以下入口：

- anchor/corrected_sgta/cecd_system_pih_canary_factories_v1.py
- anchor/corrected_sgta/models.py
- anchor/corrected_sgta/run_huatuo_token_dependence.py

它们足以定位 image span、调用 get_input_embeddings 取得 header embeddings、
重建 attention mask 与 position IDs，并在不改权重的情况下做至多 16 例 canary。

### 但当前 substrate 不足以做可信 canary

1. 本地没有已审计的 chest anatomy segmentation specialist/checkpoint；
2. VinDr finding boxes 不是 left-lung、right-lung、heart 的完整 anatomy masks；
3. 固定几何粗框可用于 plumbing，却不能证明 specialist addressing；
4. Hulu 的 native visual-token compression 假设连续原生 image block，插入文字与重排
   不能直接声称 faithful/general port；
5. Qwen2.5-VL 一类模型使用 2D/3D multimodal RoPE，重排 token 时还必须同步维护坐标，
   进一步说明该操作并非架构通用。

### 若仅作已知方法消融，最低限度 factorial

即使未来作为 baseline，也必须至少包含：

1. 原始 raster；
2. 纯重排、无 header；
3. 正确 header + 正确分组；
4. neutral header + 正确分组；
5. 错置 header + 相同分组；
6. 正确 header + 随机分组；
7. header-content 配对保持但 packet 顺序置换。

只有第 3 项同时优于所有控制，并且只改变 location binding、不统一移动 finding
positive rate，才可说存在经验地址效应。这个 factorial 仍只能验证一个已有 region
prompt 家族在某模型上的 zero-shot 泛化，不能修复新颖性碰撞。

## 5. 为什么不运行 16 例 GPU

本轮的死亡条件是公式或系统碰撞，而不是预期效果差：

1. region visual content + language address 已被 Oscar、GPT4RoI、Groma、
   Omni-RGPT 和 VLX-Seek 系统覆盖；
2. anatomy-localized tokens 已在 CXR reporting 和 Anatomy-VLM 中直接使用；
3. 当前 training-free 版本没有 scope、equivariance 或 binding 保证；
4. 若增加建立绑定所需的训练、region token 或 block mask，就回到上述已有方法；
5. 即使 16 例有正结果，也只能说明冻结模型偶然理解 OOD interleaving，不能形成新的
   通用 hallucination mitigation 原语。

因此不下载 segmentation specialist、不写 runner、不占用 GPU。

## 6. Baseline 状态

审计时 GPU 仍由 baseline PID 916425 独占约 35278 MiB；持久
baseline_cross_methods_v3 等 tmux 会话均在。本轮没有暂停、终止或修改任何 baseline
任务。

## 7. 最终裁决

| 问题 | 判断 |
|---|---|
| 是否保留原视觉 token 值 | 是 |
| 是否因此建立 anatomy address binding | **否** |
| 是否有 packet scope 或 order invariance | **否** |
| 是否只是 prompt engineering | training-free 版本本质上是 embedding-level region prompt |
| 是否与 region token 文献碰撞 | **是，机制直接碰撞** |
| 是否与医学 anatomy-token 文献碰撞 | **是，系统直接碰撞** |
| 是否值得本地 16 例 GPU | **否；正结果也无法修复贡献** |

最终结论：C60 只是从既有 region-token 范式中删去了 region-language alignment
training，却没有用新的 binding principle 替代它，因此不能作为新方法晋级。
