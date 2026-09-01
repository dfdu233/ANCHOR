# Blackwell–Computation Paradox v1：局部图为什么看似“创造”了临床信息？

> 2026-08-12；只做现有 artifact 的 CPU 审计和机制/碰撞裁决，不占 GPU，不修改 baseline。

## 裁决

**NO-GO：不把它作为 ICLR Oral 主方法，也没有推出一个区别于普通 global–local fusion 的新算法。**

它给出了一个干净但经典的解释：crop 没有为理想观察者创造信息，却能让一个固定、受 token/分辨率/计算限制的 VLM 更容易读出局部模式。这个现象属于 **Blackwell 信息序在受限决策器下不再保证性能单调**，而不是新的信息论悖论。顺着该解释自然得到的“保留全局、放大局部、冲突时不确定”已经被 global–local decoding、foveated token allocation 和 budgeted acquisition 大面积覆盖。

## 1. 先纠正证据口径

不能把“阳性 expert bbox crop 的 recall 91.9%”与“阴性 random ROI crop 的 FP 67.7%”直接组成一条 ROC：两类样本使用了不同的选区策略，selector 本身知道的标签信息不同。

最干净的病例内证据是 `full` 与 `native_context_removed`：ROI 的位置、大小和像素不变，只模糊 ROI 外部；在 62 张三 reader 一致阴性图上：

| 同病例输入 | FP rate | mean Yes−No margin |
|---|---:|---:|
| Full，neutral prompt | 8.1% | -0.679 |
| Native context removed，neutral prompt | 71.0% | +0.323 |
| 变化 | **+62.9pp**，95% CI `[+50.0,+74.2]` | **+1.002**，95% CI `[+0.851,+1.155]` |

这确认：**删除全局上下文会把 Huatuo 的工作点猛烈推向阳性。** 但它并未确认新临床信息。对当前 62+62 的已检视 panel 做 post-hoc CPU 审计：

| Score | AUROC | 相对 full 的 AUROC 变化（image bootstrap 95% CI） |
|---|---:|---:|
| Full | 0.7946 | — |
| Native context removed | 0.7980 | +0.0035 `[-0.0623,+0.0730]` |
| Native+sham panel | 0.8485 | +0.0539 `[-0.0173,+0.1255]` |
| Zoom+sham panel | 0.8278 | +0.0330 `[-0.0445,+0.1130]` |
| Zoom+true-context panel | 0.8405 | +0.0458 `[-0.0212,+0.1160]` |

所有 CI 都跨 0。`full+crop` cross-fit calibration 的 AUROC 为 0.8548，而 full cross-fit 为 0.8221；但该 panel 已被反复检视，而且 selector policy 不统一，所以它只能生成假设，不能当 confirmation。

更重要的是，补回真实全图 thumbnail 并没有“救回”阴性：`zoom+true context` 相对 `zoom+sham` 的 FP 反而高 9.7pp，CI `[+3.2,+17.7]`。这否定了“只要同时保留 global negative certificate，就能直接修好 crop”这一朴素版本。

## 2. 所谓悖论为什么并不矛盾

### 2.1 理想观察者：crop 不可能更有信息

令原图为 $X$，crop 为 $C=T(X)$；$T$ 是确定性后处理。对任意临床真值 $Y$、损失函数 $\ell$，允许决策器任意优化时，完整图的最优风险满足

\[
R^*(X)=\inf_d \mathbb E[\ell(d(X),Y)]
\le
\inf_a \mathbb E[\ell(a(T(X)),Y)] = R^*(C).
\]

背景解释：看到完整图的理想观察者总可以先自己执行同一个 crop，再照 crop 观察者做决定。因此 crop 不会在 Blackwell 意义上优于 full image。这是标准 data processing / Blackwell 结论，不是本文的新定理。

### 2.2 固定 VLM：Blackwell 保证的前提不成立

真实系统不是任意决策器，而是固定视觉编码器 $\phi$ 和固定解码规则 $h$：

\[
f_{full}(X)=h(\phi(P_{full}(X))),\qquad
f_{crop}(X)=h(\phi(P_{crop}(T(X)))).
\]

即使 $C=T(X)$，通常也不存在一个变换 $G$ 使

\[
\phi(P_{crop}(T(X)))=G(\phi(P_{full}(X))).
\]

原因是 full 与 crop 会在进入视觉塔前经过不同的 resize、patch 化和 token 压缩。局部小病灶可能在 full 路径下被一个 patch/低分辨率平均掉，在 crop 路径下却占据许多 token。因此，**原始输入的 Blackwell 顺序不等于模型内部 representation 的 Blackwell 顺序。**

也可定义受限规则族 $\mathcal H$ 的风险：

\[
R_{\mathcal H}(Z)=\inf_{h\in\mathcal H}\mathbb E[\ell(h(Z),Y)].
\]

只有当 $\mathcal H$ 对“先 crop 再决策”的复合操作封闭时，full 的单调优势才保留；singleton 的冻结 VLM 显然不封闭。受限/刚性决策规则下的信息排序不同于 Blackwell 顺序已有经典研究，所以这是一种解释框架，不是足够的新理论贡献。

## 3. 能推出的最简非平凡边界

把图像分成 $M$ 个可能含病灶的区域，疾病存在表示为

\[
Y=\bigvee_{i=1}^{M} L_i,
\]

即只要任一区域 $L_i=1$ 就为阳性。若一次局部观察只精细检查 $k<M$ 个区域，且单一病灶位置在 $M$ 区域中均匀，则即使局部识别器完美，命中病灶的概率也至多为

\[
P(\text{hit}\mid Y=1)\le \frac{k}{M}.
\]

没有命中时，“全图阴性”与“病灶恰在未检查区域”对该局部观察完全相同。因此，一个局部 crop 可以提供 **存在的 witness**，却不能单独提供 **不存在的 certificate**；若强制回答而不允许 `undetermined`，条件 FN 下界为 $1-k/M$。

这个边界解释了为什么局部化天然是极性不对称的，也说明一个安全系统应：

- 用局部高分辨率寻找阳性 witness；
- 用全局覆盖维持阴性证据；
- 两者冲突时进入 `undetermined`，而不是平均后强行二选一。

但该结果是 OR/MIL 搜索的基础边界；它没有给出新的可识别 clinical likelihood，也没有解决 crop 本身 +50pp 以上的假阳性偏置。

## 4. 为什么没有得到一个非普通 fusion 算法

最自然的 “duplex” 规则是从 full 得到负证据 $e_-$、从 local 得到正证据 $e_+$，再用三态输出：

\[
z_{support}=e_+,
\quad z_{refute}=e_-,
\quad z_{undetermined}=\tau+\min(e_+,e_-).
\]

背景解释：两边都强时，`min` 大，表示“局部说有、全局说无”的冲突；不是把两者平均掉。

这个公式看起来优雅，但当前不能作为方法：

1. crop margin 不是可信的 positive likelihood；随机阴性 crop 的 FP 已达约 63–71%；
2. full margin 也不是严格 negative certificate，小病灶正是 full 路径容易遗漏的对象；
3. 把冲突变成不确定，与 BCEA 的 answer/abstain/acquire 和普通 selective prediction 高度重合；
4. 若把 $e_+,e_-$ 再校准/学习融合，方法退化为普通 global–local calibration；
5. 当前 `zoom+true-context` 没有救回阴性，直接反驳最简单的 single-canvas duplex 实现。

因此没有诚实的“只加一行公式便同时提高 recall、降低 FP”的算法。强行提出会重复本项目此前的问题：response change 被误写成 evidence gain。

## 5. 机制级碰撞

| 工作 | 已占据内容 | 对本候选的影响 |
|---|---|---|
| Fine-Grained Visual Prompting, NeurIPS 2023 | crop/blur/mask，并用 blur reverse mask 保留空间上下文 | “保留 global、突出 local”不是新方法 |
| HALC, ICML 2024 | adaptive focal grounding + local/global focal-contrast decoding | local witness + global context 的解码已直接覆盖 |
| AGLA, CVPR 2025 | global generative 与 local discriminative feature assembly | full/local score 或 feature fusion 直接碰撞 |
| Perception Magnifier, ACL 2026 | 逐 token 定位、放大，同时保留 structural/context information | foveated refinement 直接碰撞 |
| BCEA, 2026 | crop/zoom acquisition、post-acquisition calibration、Blackwell/ROC 价值条件 | “选后再校准/不确定/继续看”高度碰撞 |
| CropVLM, CVPRW 2026 | 学习 zoom policy 做 fine-grained perception | crop 提高有限模型可读性已占 |
| Focus-Scan-Refine / PromPrune, 2026 | 固定 visual-token budget 下平衡 local saliency 与 global coverage | rate–distortion/token allocation 版本也拥挤 |

剩余空间最多是一个机制/测量命题：**固定 VLM 的有效信息序会随计算预算反转**。但这需要发现跨模型、跨任务稳定的 `budget × view` scaling law，而不是再做一次 global/local assembly。

## 6. 若要做最后一次 fresh 致死实验

这个实验只判断是否存在“受计算限制的 Blackwell reversal”，不再测试一个 fusion 配方。

### 公平策略

- 为每个 finding 冻结 label-blind anatomy ROI 或使用同一个外部 selector 处理阳性和阴性；禁止 positive expert bbox、negative random ROI 混用。
- patient-disjoint dev/test；至少 320 test images，finding 与 label 分层。
- 所有阈值和融合权重只在 dev 冻结。

### 预算轴

在相同视觉 token/FLOP 预算 $B$ 下比较：

1. `Global(B)`：完整图；
2. `Local(B)`：同 policy ROI；
3. `Duplex(B_g,B_l)`：$B_g+B_l=B$，全局低分辨率 + 局部高分辨率；
4. 普通 score-average、HALC/AGLA/Perception Magnifier/BCEA 作为直接碰撞基线。

定义计算反转量

\[
D_B=\operatorname{AUROC}(Local(B))-\operatorname{AUROC}(Global(B)).
\]

核心独有预测不是“某个 split 最好”，而是：若现象真由 full 路径的 rate/compute bottleneck 引起，随着 $B$ 增大，$D_B$ 应系统性收缩并趋近 0；若只是 crop 诱发的 criterion/shortcut，AUROC 不会出现这一预算规律。

### GO / NO-GO

机制 GO 必须同时满足：

- 两模型、多数 finding 上低预算 $D_B>0$，bootstrap CI 排除 0；
- $D_B$ 随预算单调下降，预注册 trend test 通过；
- 在 matched FPR 下观察到相同趋势，不允许只靠 Yes-rate 平移；
- 公平 selector 下成立。

方法 GO 还需 `Duplex` 相对最佳已有 global–local baseline：AUROC 至少 +0.02、matched-FPR recall 至少 +5pp、FP 不增加，CI 均通过。鉴于强碰撞，即使机制 GO、方法 gate 通过，也只够继续研究，不自动达到 Oral。

## 7. 最终价值判断

| 维度 | 裁决 |
|---|---|
| 现象自然性 | 高：同病例删上下文使阴性 FP +62.9pp |
| 理论正确性 | 高，但经典：受限规则不继承理想 Blackwell 单调性 |
| 当前实证 | 主要是 criterion shift；所有 view AUROC 增量 CI 跨 0 |
| 算法新颖性 | 低：自然算法就是 global/local foveation、融合或 abstention |
| ICLR Oral readiness | **不满足** |

因此本方向可以作为论文中的解释性框架或一个 scaling-law 致死实验，不能作为当前主方法。若 fresh 公平策略下的 budget law 不成立，应彻底关闭；若成立，优先写“计算预算改变有效信息序”的机制论文，再判断是否真的存在超越现有 global–local 方法的算法。

## 已核实参考

1. Yang et al., *Fine-Grained Visual Prompting*, NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html>
2. Chen et al., *HALC*, ICML 2024. <https://proceedings.mlr.press/v235/chen24bi.html>
3. An et al., *AGLA*, CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html>
4. Mao et al., *Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding*, ACL 2026. <https://aclanthology.org/2026.acl-long.2059/>
5. Xu et al., *Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models*, 2026. <https://arxiv.org/abs/2606.16667>
6. Carvalho et al., *CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception*, CVPRW 2026. <https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Carvalho_CropVLM_Learning_to_Zoom_for_Fine-Grained_Vision-Language_Perception_CVPRW_2026_paper.html>
7. Rauh et al., *Coarse-Graining and the Blackwell Order*, Entropy 2017. <https://www.mdpi.com/1099-4300/19/10/527>
8. Salant, *Procedural Analysis of Choice Rules with Applications to Bounded Rationality*, AER 2011. <https://www.aeaweb.org/articles?id=10.1257/aer.101.2.724>

## Artifact provenance

- `corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1/analysis.json`
- `corrected_runs/daylong_idea_search_v1/context_completion_signal_huatuo_v1.json`
- `corrected_runs/daylong_idea_search_v1/search_reuse_huatuo_v1/analysis.json`

