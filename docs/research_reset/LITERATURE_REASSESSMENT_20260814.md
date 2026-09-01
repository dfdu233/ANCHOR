# 文献重审：医学 VLM 幻觉缓解与专家协作

## 1. 文献给出的现实边界

MedHEval 已在 11 个医学/通用 LVLM、7 种缓解方法上显示，现有方法对知识和上下文型错误尤其有限：https://arxiv.org/abs/2503.02157 。这与本项目“通用 mitigation 经常只移动工作点”的观察一致。

训练免费解码已经非常拥挤：VCD、DoLa、OPERA、AvisC、VISTA、SECOND、VDGD、VTI、SPIN，以及医学场景的 Med-VCD、ARCD、CCD、LEAD 都覆盖了图像扰动、层间对比、注意力/head 抑制、视觉描述和专家引导等接口。继续提出普通 logit 加减、层选择或 mask 细化，很难形成新颖性。

## 2. 专家—通才协作并非空白

- VILA-M3 把分类/分割专家知识加入医学 VLM 的专门训练阶段：https://openaccess.thecvf.com/content/CVPR2025/html/Nath_VILA-M3_Enhancing_Vision-Language_Models_with_Medical_Expert_Knowledge_CVPR_2025_paper.html
- Chimera 用渐进训练与 collaboration masking 融合领域专家特征：https://arxiv.org/abs/2412.05983
- GSCo 把专家预测与相似病例作为上下文交给 generalist，并在 32 个数据集上评估：https://www.nature.com/articles/s41551-026-01653-3
- CoMed-TR 用问题类型路由多个医学专家并进行关系蒸馏：https://doi.org/10.1016/j.imavis.2025.105820
- LEAD 把多个病理专家特征逐层注入报告生成模型：https://arxiv.org/abs/2602.04617
- CCD 用任务专家做医学 contrastive decoding，但其 ICLR 2026 记录为 withdrawn，不应误写为录用：https://openreview.net/forum?id=eEnW7lUXxY

因此，“使用小模型”“加入专家分数”“融合专家特征”都不能作为贡献。未解决的缺口只能是：**专家应传递什么数学对象，才能跨接收模型保留病例证据，同时不把 verdict 当作新事实。**

## 3. 邻近数学领域

### 分布式检测与充分统计

经典分布式检测表明，已知生成模型与条件独立性时，likelihood ratio 是二元检测的充分消息。这为“专家发送证据而非标签”提供原则，但本身是经典结果，不能作论文理论贡献。

### Blackwell 信息序

Blackwell 框架比较一个信息通道是否能在所有决策问题上优于另一个。2026 年已有 multi-agent LLM 工作用该框架分析 voting/debate：https://arxiv.org/abs/2605.06028 。因此简单把 Blackwell sufficiency 搬到医学 VLM 不足以创新；它更适合定义门槛。

### 语义通信

task-oriented semantic communication 强调只传对下游任务有用的统计量；2026 年 UniSC 已研究面向任意文本查询的可匹配语义子空间：https://pubmed.ncbi.nlm.nih.gov/42113660/ 。这提示“固定诊断票不适合开放问题”，但直接做特征压缩/传输会与已有工作碰撞。

### 受控生成与风险控制

FUDGE 已用未来判别器按 Bayes 分解修改 token 分布：https://arxiv.org/abs/2104.05218 。NeurIPS 2024 Selective Generation 和 AAAI 2026 COIN 已对生成/QA 做 FDR 风险控制：https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a6815122f533193a022cbc41786c1cc-Abstract-Conference.html ，https://ojs.aaai.org/index.php/AAAI/article/download/40667/44628 。e-BH 在任意依赖的 e-values 下控制 FDR：https://arxiv.org/abs/2009.02824 。因此“给 claim 做 conformal/FDR 筛选”也不是天然空白，且很容易退化为拒答或删 claim。

## 4. 文献与本地证据的交叉结论

1. 小模型协作有真实价值，但主流接口已覆盖 verdict、prompt、retrieval 和 feature fusion。
2. 本地完整状态增益说明“单个专家票压缩过度”；跨 VLM 迁移失败又说明“模型特定融合”不是通用证据接口。
3. 报告中同一疾病词大多可能处于否定语境，说明 token-level expert bias 在语义上欠定义。
4. 统计风险控制虽优雅，但若只删掉低置信 claim，会与现有 selective generation 重叠，也违反本项目“不以遗漏换幻觉”的要求。

## 5. 研究空白的严格表述

当前只能保留下面这个未决问题：

> 是否存在一种由专用医学模型独立产生、与接收 VLM 无关、同时保留 finding 极性与不确定性的病例级证据消息；它能在至少两个不同 VLM 上同时减少 FP 与 FN，并自然扩展到开放报告？

它是下一步要验证的问题，不是已经成立的论文贡献。
