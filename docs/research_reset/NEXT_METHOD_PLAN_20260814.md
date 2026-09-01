# 下一阶段计划：停止重复融合，先找到一个尚未被解释的自然现象

## 1. 复盘后的边界

此前计划的“VLM-agnostic 专家消息 Gate A”与仓库 C52 已完成的 conditional likelihood-ratio / innovation 实验重复。该对象等价于 correlated classifier stacking，且跨模型门失败，故取消，不重复实验。

对冻结 VLM，小模型若改变输出，只能通过输入、hidden/attention、token 分布、搜索或后处理介入；这些接口分别已有 visual prompting/foveation、feature fusion/steering、guided decoding、reranking/veto 和 editing。没有“第六条通道”。

因此下一步不能再从一个优化公式出发。必须先发现一个满足下列条件的新现象：跨至少两个模型、病例级、不是工作点移动、且现有方法的干预接口没有覆盖它。

## 2. 下一轮只做“现象发现”，不先命名算法

### Stage A：从已有输出中寻找跨模型共同错误（CPU）

对 694 个 MIMIC 报告与已完成 OE/CE 输出，按可验证来源拆成：

1. 局部病灶 finding；
2. 设备存在与位置；
3. measurement；
4. temporal/prior；
5. finding polarity/uncertainty。

只保留在 Huatuo/Hulu 至少两个模型中同方向、各不少于 50 个错误、并有结构化真值或公开专家标注的数据子问题。词法 regex 只能筛选，不能定真值。

### Stage B：对幸存子问题做“信息存在性”门（CPU/已有缓存）

候选小模型/元数据必须在 VLM 原分数之上提供病例级增量；要求至少两个模型 `>= +0.02 AUROC` 且 CI 下界大于 0。只在一个弱模型有效，解释为互补 baseline，不升主线。

这一门防止再次把“模型有响应”错当成“增加了证据”。

### Stage C：只有现象与信息门都通过，才做方法设计

方法必须改变一个尚未被占据的计算对象；普通 prompt、score fusion、层融合、crop、mask、rerank、veto、conformal deletion 和 report editing 自动淘汰。若必须放宽约束，优先允许 `<1%` 参数训练，而不是伪装成 training-free。

小规模 GPU canary 最多 4 小时；方法关闭态 32/32 token-exact；matched claim count 下 FP 相对下降 15%，FN 不增加。通过后再做 OE/report 放量。

## 3. 当前最有希望的现象种子

唯一已经跨模型确认的是“小病灶获得更弱支持”，但已知 crop、sparse scan、foveation、coarse-to-fine token refinement 均因实证失败或强文献碰撞关闭。它只能作为现象种子，不能直接推出方法。

大小模型协作保留为组件而非主创新：只有当 Stage A 找到一个专用模型具有稳定增量、且现有 VLM 接口不能表达的错误子问题时，才重新启用。

第一轮 MIMIC 自动筛选发现一个待确认候选：医疗器械。central-line 错误为 Huatuo `20 FP/66 FN`、Hulu `24 FP/51 FN`；pacemaker 为 `26/30` 与 `18/25`。骨折和结节则在两个模型中呈相反错误方向，不能形成统一机制。器械候选的下一门不是直接做 decoding，而是确认：

1. 这些错误在公开专家标注或结构化器械标签上仍成立，而非报告省略/regex 误判；
2. 小型器械模型能否同时给出存在、曲线路径和末端解剖关系，而非另一个 scalar vote；
3. 图结构是否提供超出普通 device classifier 的病例增量。

若三项成立，才研究“结构证据编译”这一窄子问题；若只剩分类概率，则退回已有 expert fusion，立即关闭。

## 4. 暂不启动的数学包装

- Blackwell order、likelihood ratio、information bottleneck 可解释“消息为何不该是硬标签”，但都是既有数学，不能单独作贡献。
- e-values/e-BH 可控制被接受 claim 的 FDR，但邻近 selective generation 已很强，且容易用删 claim 换低幻觉；只保留为安全约束候选，不作为当前主线。
- KL/I-projection、FUDGE 式 potential、控制 barrier、DID 双中心化均已有明显碰撞。

## 5. 资源调度

1. Stage A/B 只用 CPU 与已有缓存，不占 baseline GPU。
2. 立即恢复 Hulu–DoLa IU-Xray `274/590` 及其余 baseline 队列。
3. 只有 Stage B 通过时，才在完整 chunk 边界给新方法最多 4 GPU 小时。

## 6. 当前论文状态

- 已有完整负结果资产和一个可靠的小病灶机制现象。
- 已有“小模型完整状态可补充弱 VLM”的正信号，但它是 stacking 上限。
- 尚缺：一个未被现有接口解释的跨模型自然现象、由其推出的新计算对象、OE 无遗漏增益。

因此当前不是 ICLR-ready，更不是 Oral-ready。下一阶段的目标是在 1–2 天内完成一次不重复旧路线的现象筛选；没有通过者就诚实保留负结果，而不是继续换名。
