# 十天 Baseline 收口与 ICLR 新方法探索 Goal（待确认）

**拟启动时间：** 用户确认后记为 D0  
**最低运行窗口：** D0–D10（连续约十天）  
**仓库：** `/home/dbw/ANCHOR`  
**硬件现状：** 1 × RTX 4090 48GB；GPU 任务必须通过既有 canonical flock 串行运行  
**状态：** 仅规划，尚未创建 Goal、尚未启动新的十天任务

**启动确认：** 用户已确认，剩余 Codex 用量约 38%；按低额度策略执行，Baseline 不因 Codex turn 用尽而停止。

## 1. Goal objective（确认后原样创建）

在不训练或微调任何模型、不使用测试图像参考池、不依赖病种分类的前提下，连续推进两条共享资源但互不阻塞的工作线：

1. 收口论文 Baseline 矩阵。恢复现有队列，完成所有在给定模型、方法和数据集上语义与架构适用的实验；对确实不适用的格子，只在形成可复核的入口一致性、方法激活、依赖、权重或架构不兼容证据后标记 N/A；完成统一评分、覆盖审计和论文表导出，最终不得遗留无解释的 pending、partial 或 generated-unscored。
2. 探索一个满足 ICLR 论文强度的通用、简洁、推理时医学 VLM 幻觉缓解方法。冻结核心问题为“当单张医学图像不足以约束答案时，语言先验及输出历史如何接管生成”；以单图内部的视觉因果支持为主变量，验证并形成无需标签、无需图像池的候选级或 claim 级推理解码规则；先做致死实验，只有通过机制门才扩展到多模型、多数据集和正式 Baseline 对照。

十天是最低工作窗而不是伪完成条件。只有达到第 3 节的终止标准才可将 Goal 标记 complete；接近十天但仍有安全且有信息量的工作时继续运行。

## 2. D0 冻结快照

### 2.1 Baseline 当前状态

2026-08-15 重新执行覆盖审计后的主矩阵：

| 状态 | 格子数 |
|---|---:|
| 全部矩阵 | 336 |
| completed | 69 |
| generated-unscored | 1 |
| running/partial | 2 |
| pending | 87 |
| N/A | 177 |

辅助控制共 40 格：16 completed、1 running/partial、8 pending、15 N/A。

87 个 pending 中，31 个属于 training-free 路径，56 个属于 LLaVA-1.5 训练型方法路径。后者覆盖 base、HA-DPO、OPA-DPO、DA-DPO、SENTINEL、Less-is-More、FactMM-generator、VHR × 7 个数据集；必须先通过官方入口的 token identity / method activation / checkpoint 与依赖门，不能直接把“未运行”当作 N/A。

已知立即需修复的问题：

- native 主队列已经退出 tmux，但 Huatuo VQA-RAD greedy 停在 164/200；应从现有合法前缀恢复。
- Huatuo shared-medical-RAG 的 Visual-MIMIC 停在 18/490；应从现有合法前缀恢复。
- Huatuo Knowledge-MIMIC greedy 已生成 2000/2000，但评分没有绑定当前 evaluator source；只重评分，不重生成。
- LLaVA 的若干 RAG/report 任务因缺失或错误解析 `openai/clip-vit-large-patch14-336` 权重而失败；先审计本地 checkpoint，修复依赖后只重试受影响格子。
- 单卡当前空闲，但多个旧 tmux 只剩监控/等待进程；D0 必须区分“活任务、死队列、僵尸进程和合法监控”，不能凭 session 名判断任务仍在执行。

### 2.2 新方向已有证据

在 HuatuoGPT-Vision-7B、VQA-RAD 34 个自然相反答案配对（68 图）上：

- 只阻断最终 prompt query 对 image tokens 的边不能测出约束：pair-error AUROC 最高约 0.61。
- 阻断所有 image → prompt-suffix 的注意力因果边后，视觉方向对比预测 pair error 的 AUROC 为 **0.875**，bootstrap 95% CI **[0.733, 0.975]**；其与自然配对 between-image JS identifiability oracle 的相关强度约为 **0.896**。
- 熵基线 pair-error AUROC 仅 **0.583**。
- 但当前强指标使用了配对方向或真值，只能证明“视觉约束这个潜变量存在且可干预”，尚不能部署；现有无标签 selected-candidate support 仅约 0.54 pair AUROC / 0.60 image AUROC。

因此十天的科学重点不是重复证明存在容易幻觉的图，而是解决唯一真正的缺口：把强的、标签依赖的机制信号变成单样本、无标签、可直接参与推理的因果证据预算。

## 3. 完成定义

### 3.1 Baseline 完成标准（必须全部满足）

1. 最新覆盖审计中，主矩阵和辅助控制均满足：`pending=0`、`running_or_partial=0`、`generated_unscored=0`。
2. 每个 applicable cell 均有完整生成、qualification、统一 evaluator 绑定的评分及运行 fingerprint。
3. 每个 N/A cell 均有机器可读 reason 和证据路径；允许的原因仅包括：任务语义不适用、官方方法依赖不存在、权重不可取得、架构接口不支持、T1 关闭态不一致或 T2 方法未激活。普通 OOM、网络抖动、锁竞争和代码错误不能作为 N/A。
4. 输出按 model × method × dataset 的覆盖表、主结果表、失败/N/A 表；数值表不得混入不合格输出。
5. 对仍失败的普通工程问题至少三次有区分度的恢复尝试；修复不得改变数据、方法或评测语义。

### 3.2 新方法阶段性完成标准（必须形成明确结论）

至少交付以下两种结论之一：

- **GO：** 得到一个标签自由、单图内部、推理时算法，在至少 2 个模型族和 2 类医学任务上通过机制验证；在正式缓解实验中，相对 greedy 和强 training-free baselines 于 matched claim coverage / matched answer rate 下显著降低 hallucination，且 omission、准确率、临床矛盾和长度没有不可接受退化；完成关键消融、成本报告和最近工作碰撞表。
- **NO-GO/PIVOT：** 预注册致死门明确否定核心机制或无标签转化，给出可复现实验、置信区间、失败原因以及最多一个由证据直接推出的下一问题。不能用换名字、增加模块或只换数据集挽救。

“找到新方法”不等于保证论文录用。目标是形成一条达到 ICLR 投稿所需的机制、方法、对照和跨设置证据链，或在十天内诚实关闭不成立的路线。

## 4. 冻结的科学合同（确认本文档即确认三项 freeze）

### 4.1 Problem freeze

**问题：** 医学 VLM 的幻觉并不只来自“模型没看图”，而可能来自图像对语言答案的约束不足；此时 prompt 与自回归历史提供的语言支持超过视觉可提供的证据。能否在单次测试样本内部测量这一失配，并在推理时阻止语言先验越过视觉证据边界？

研究对象保持通用：不按疾病、模态或幻觉类型设计分类器；分类只用于分层报告，不进入算法。

### 4.2 Mechanism freeze

对候选 token 或原子 claim `y`，保持相同 image、prompt、已生成 response、token 位置和长度，比较三条白盒前向路径：

- `Full`：正常模型的候选相对 margin `mF(y)`；
- `−V`：阻断 image → prompt-suffix / response 的视觉因果边，得到 `m−V(y)`；
- `−H`：保留 image 与 question，但阻断当前生成位置对先前输出 token 的直接历史边，得到 `m−H(y)`。

定义候选级视觉支持 `V(y)=mF(y)-m−V(y)`，历史自支持 `H(y)=mF(y)-m−H(y)`。冻结的机制预测是：

> 当候选的历史自支持为正，而视觉支持弱、为零或反向时，生成主要由语言自强化维持；这种 `H` 超过 `V` 的证据赤字应先于或伴随 hallucination，并且比熵、raw attention mass、no-image 差值或 prompt 扰动更稳定。

优先探索一个无标签、零点有意义的规则，而不是在验证标签上学习融合器：

`penalty(y) = max(0, H(y) - max(V(y), 0))`

候选方法暂称 **Causal Evidence Budgeting (CEB)**，名称不构成贡献。最简在线形式只对当前 top-k 候选减去该证据赤字；最简离线形式对 draft 中高风险 claim 做删除、拒答或约束重生成。两者共享同一机制变量，不同时堆叠成多个模块。

### 4.3 Substrate freeze

- 主仓库：`/home/dbw/ANCHOR`。
- 首个模型与最小机制设置：HuatuoGPT-Vision-7B + 已冻结 VQA-RAD 68 图自然配对，复用现有视觉边干预代码和 raw logits。
- 第一独立确认：Hulu-Med；只有 Huatuo 致死门通过才占用 GPU。
- 第二架构确认：LLaVA-Med 或当前 Baseline 中第一个完成且能暴露 token spans / attention edges 的不同架构模型。
- 任务扩展顺序：短答案 VQA → 开放式 VQA → 报告生成。报告生成只有在 claim-level teacher-forcing 门通过后进入在线解码。
- 不允许增加训练、图像检索池、疾病特定规则或额外医学模型来挽救主算法。

## 5. 两条并行工作线

### Track A — Baseline matrix closure

#### A0. D0：重建执行真相

- 再跑覆盖审计；保存 D0 snapshot。
- 检查 GPU、flock、tmux、进程、输出行数、qualification 和 evaluator fingerprint。
- 终止判定只针对已经退出且不持锁的死 shell；不得删除或覆盖已有结果。
- 按“仅评分 → 合法前缀恢复 → 缺依赖修复 → 未开始格子”的成本顺序恢复。

#### A1. D0–D2：收掉近完成与纯评分格子

- 完成 164/200、18/490 等 partial。
- 对 generated-unscored 和已生成但 evaluator fingerprint 过期的输出统一重评分。
- 验证现有 scoring monitors 是否仍实际推进；失效则直接运行既有评分入口。
- 修复 LLaVA CLIP 本地依赖解析，做 1 条真实样本 canary，再恢复相关队列。

#### A2. D1–D7：完成 training-free applicable cells

- native greedy/beam、shared medical RAG 及已通过 T1/T2 的 VCD、DoLa、OPERA、PAI、AvisC、VISTA 等方法按现有 manifest 运行。
- 单卡一次只加载一个模型；同模型相邻任务连续执行以减少重复载入。
- report/OE generation 在 GPU 上运行时，CPU 同时做 completed cells 的 qualification、claim extraction、bootstrap 和表格导出。
- 失败重试必须识别原因；OOM 可降 batch/关闭 cache，但不能改变 decoding method、max token、seed 或数据顺序。

#### A3. D2–D8：训练型方法逐方法门控与执行

- 先完成 official-entry T1（关闭态 token identity）和 T2（方法实际改变且激活）审计。
- 已有 checkpoint 且通过门的方法进入 7 数据集矩阵。
- checkpoint/依赖确实不可取得或架构不支持者，生成正式 N/A 证据，不反复下载或用非官方替代物偷换方法。
- VHR 单独沿官方实现与当前 VHR 队列验收；不得把其它 attention masking 结果记为 VHR。

#### A4. D8–D10：补洞与最终审计

- 对剩余非零状态逐格追踪根因并收口。
- 重新执行 coverage audit、paper table export、compile、相关 smoke、Git/LFS 状态检查。
- 产出最终 `baseline_completion_report.md`，明确 completed 与 N/A 的比例和原因。

### Track B — mechanism-first ICLR method discovery

#### B0. D0–D1：无 GPU 的标签自由转化致死门

直接复用现有 68 图的 Full/−V raw logits，不重跑模型。至少比较以下无需标签或配对池的 per-image 量：

- selected-candidate visual support；
- top-k 候选的视觉支持 gap / rank change；
- 视觉反对当前选择的 signed deficit；
- Full 与 −V 的分布距离、但以候选 margin 变化而非 raw attention 为主；
- prompt paraphrases 只作稳定性压力测试，不参与拟合。

冻结门：在 image-level error 上 AUROC ≥0.70，且对 entropy 的 paired bootstrap 差值下界不为明显负；prompt 改写下 rank correlation ≥0.60。若所有无标签量均失败，停止直接 mitigation，优先解释“为什么 truth-aligned 信号强而 selected signal 弱”，不得调标签权重硬过门。

#### B1. D1–D3：历史自强化机制门

- 在冻结的 native open-ended outputs 上 teacher-force 同一个 draft，执行 Full/−V/−H。
- factual content token 聚合为 atomic claim；claim extractor 仅用于评价边界，不参与学习机制分数。
- 主比较：`V`、`H`、`H−max(V,0)` 对 grounded/hallucinated claim 的 AUROC 与随 response position 的变化。
- 强控制：等数量随机 text-edge block、prompt-history block、直接删视觉 token、raw attention mass、entropy、no-image logit difference；保持 token、位置和长度完全一致。
- 机制通过门：证据赤字 claim AUROC ≥0.70，优于 entropy 至少 0.03，且 later-response hallucination 的 `H−V` 高于 matched grounded claims；至少一个独立数据集同向。

如果 −H 不提供独立信息，则简化为纯视觉证据预算方法；复杂度只能减少，不能为了原计划保留历史模块。

#### B2. D3–D5：最小缓解算法

- 先做 teacher-forced top-k replay，验证零阈值 penalty 是否把错误候选降出 top-k，同时保留正确候选。
- 再做一个真实在线解码 canary。默认不学习阈值、不拟合融合器；仅允许固定的 top-k 和计算预算。
- 短回答以 accuracy / BAcc / FP / FN / abstention 报告；开放式与报告以 hallucinated claims、supported claims、clinical contradiction、coverage、长度和 latency 报告。
- 主判据必须 matched coverage 或 matched answer rate；单纯少说、变短或多拒答不算缓解成功。

缓解通过门：相对 greedy 的 hallucination/FP 下降且 paired CI 下界 >0，同时 accuracy/BAcc 下降不超过 1pp 或在 risk-coverage 曲线上支配；相对熵拒答和直接 prior subtraction 仍有增益。

#### B3. D5–D8：跨模型、跨任务确认

- 先 Huatuo → Hulu；两者通过后再进入 LLaVA-Med/另一架构。
- 至少覆盖短答案和 open-ended 两类任务；报告生成作为第三类确认，不作为拯救失败方法的唯一结果。
- 与已完成 Baseline 中可比的 VCD、DoLa、OPERA/PAI/VISTA/AvisC 做 matched decoding budget 和 matched coverage 对照。
- 分层只报告 image constraint、prompt sensitivity、response position 和 task format，不按病种设计分支。

#### B4. D7–D9：新颖性碰撞与 reviewer 致死审计

在方法冻结后做第二轮精确检索，重点比较视觉 token intervention、attention-edge causal masking、output-history masking、source attribution、uncertainty/abstention 和 medical VLM hallucination。逐项比较 phenomenon、mechanism、intervention 和 claim；若存在机制等价工作，立即收缩或关闭，不以“医学数据上首次”维持创新。

构造路径只作研究组织工具：

- Transformer 路径：把原本作为诊断的视觉因果支持提升为解码的组织变量；
- SigLIP 路径：去掉需要标签融合器或参考池的偶然耦合，使用候选自身证据预算；
- Chinchilla 路径：明确视觉证据与语言自支持的比例/边界，而非泛称多模态平衡；
- Model Collapse 路径：把输出历史的局部自强化视为可测的递推失真，并定位首次越过视觉预算的位置。

#### B5. D9–D10：论文级证据包

产出：问题与机制图、算法伪代码、复杂度、主表、matched-coverage 图、关键消融、失败边界、最近工作 collision matrix、可复现命令及 ICLR 论文逻辑骨架。若门失败，产出同等完整的 NO-GO 报告和下一条唯一合理路线。

## 6. GPU、CPU 与 Codex 用量预算

### 6.1 GPU 预算（十天理论上限约 240 GPU·h）

| 用途 | 保底/上限 | 规则 |
|---|---:|---|
| Baseline | 保底 180 GPU·h（75%） | 一直拥有默认优先级；近完成任务优先 |
| 新方向 | 上限 48 GPU·h（20%） | 每过一门才释放下一档；失败即归还 Baseline |
| smoke、恢复与终验 | 12 GPU·h（5%） | 只用于真实 canary、故障恢复和复核 |

动态回收：Baseline 提前闭合后，全部剩余 GPU 转入 B3；B0/B1 失败后，未用的新方法 GPU 全部转回 Baseline。不得为了维持固定比例让 GPU 空闲。

### 6.2 CPU 与存储

- 文献检索、collision audit、raw-logit 重算、评分、bootstrap、coverage audit 和论文表与 GPU generation 并行。
- 不生成新的 watcher、dashboard 或实验数据库；复用现有 tmux、state JSONL、coverage audit 和 STATUS/TODO。
- 原始 generation 不覆盖；派生评分可重建并记录 evaluator fingerprint。
- 每个新实验输出必须包含 dataset、model、method、seed、command 和 source/config fingerprint。

### 6.3 Codex 使用策略

当前接口没有暴露账户级剩余 Codex 百分比，因此本文档不虚构一个 token 数，也暂不在 Goal 中填写 `token_budget`。确认时若用户提供剩余额度，再按该额度设预算；否则创建无显式 token budget 的 Goal，并执行以下节流：

- Codex turn 用于读实际状态、做科学决策、修复和阶段验收，不做分钟级空轮询。
- 长任务交给持久 tmux；在预计完成点、失败、GPU 空闲或阶段门时唤醒检查。
- 每个阶段只保留一个主假设和一个简单替代解释；不展开大批平行点子。
- 文献搜索集中在 B0 前的 targeted refresh 和 B4 的 collision search，避免重复 survey。
- 每日最多一次完整状态汇总；普通恢复只记入 state/STATUS，减少上下文重复。

额度降级策略：

- 剩余 ≥60%：执行完整十天计划。
- 剩余 30–60%：保持全部 GPU 实验，减少中间叙述，只在门控节点进行 Codex 分析。
- 剩余 <30%：Baseline 收口优先；新方向保留 B0–B2 和一个独立模型确认，暂停论文润色与非决定性消融。当前 38% 档已按“保留 Baseline 全部队列、只运行 CPU/缓存新方向”执行。

## 7. 每日里程碑（不是机械日程）

| 时间 | Baseline 目标 | 新方向目标 | 必须留下的决策 |
|---|---|---|---|
| D0 | 状态重建、死队列恢复 | raw logits 无标签分析启动 | 冻结问题/机制/substrate |
| D1 | partial 与 unscored 收口 | B0 GO/NO-GO | 无标签量是否存在 |
| D2 | training-free 队列持续 | −H canary | 历史支持是否独立有用 |
| D3 | trained method gates | B1 GO/NO-GO | 是否允许做 mitigation |
| D4–D5 | 主矩阵连续运行 | replay + 在线 canary | 零阈值规则是否成立 |
| D6 | 评分与补洞并行 | Hulu 独立确认 | 是否跨模型 |
| D7–D8 | trained/N/A 收口 | 第二任务/架构与强 baseline | 是否达到方法主张 |
| D9 | 全量 coverage audit | collision + reviewer audit | GO / PIVOT / NO-GO |
| D10 | 表格、复现、终验 | 论文级证据包 | 是否满足 Goal completion |

## 8. 自动运行边界与失败处理

Goal 模式被授权自动执行：普通代码修复、环境修复、合法断点恢复、最多三次瞬态重试、同语义的 batch/内存调整、评分重跑、队列排序和通过既定门后的扩量。

以下变化不自动做：更换数据或 split 语义、使用非官方 checkpoint 替代某 Baseline、把训练方法改成推理近似、学习标签融合器、增加图像池/疾病分类器、改变主要指标、放宽通过门、删除已有结果、提交或推送未授权外部更改。

普通故障处理顺序：诊断 → 最小真实 canary → 恢复同一任务 → 检查产物 → 再入队。相同阻塞只有在连续三次 Goal turn 均无法推进且需要用户或外部状态改变时，才按 Goal 规则标记 blocked。

## 9. 最终交付物

1. Baseline 最新覆盖审计、完成报告、论文表、N/A 证据表和复现入口。
2. 新方法机制报告：包含正/负结果、因果变量、controls、跨模型/任务证据和 falsification。
3. 若 GO：最小算法实现、伪代码、正式结果、matched-coverage/compute 对照、消融与复杂度。
4. 若 NO-GO：不包装负结果，明确关闭范围及唯一下一步。
5. 更新 `STATUS.md` 与 `TODO.md`，保持事实只记录一次；代码 compile/test/diff check；不提交 checkpoint、cache 或大日志。

## 10. 确认项

用户确认本文档即表示同时确认：

1. Baseline 的“全部完成”定义为 applicable cells 全完成 + non-applicable cells 有证据地 N/A，而不是强行让不兼容方法产生数字。
2. 新方向冻结为 CEB 的机制问题：单图视觉证据预算与输出历史自支持失配；允许证据促使方法进一步简化，但不允许换成训练、检索池或病种专用方案。
3. 十天为最低持续工作窗；Goal 以证据终止，不以时间、token 接近耗尽或单个 pilot 完成作为完成。
