# Polarity Firewall 32-Claim Canary：冻结分析计划

## 目的与证据级别

该实验只判断去除其他患者状态是否值得进入正式 matched-donor 因果实验。它不是论文
确认实验：32 个样本来自已有 BM25 检索，retrieval polarity 不是随机分配，因而任何
准确率变化都只能称 canary 结果。

## 冻结样本与模型

- 32 个 CXR-VisHal binary claims；8 个 finding group；每组 positive/negative retrieval
  各 2 个；选择过程不访问 target label。
- HuatuoGPT-Vision-7B 与 Hulu-Med-4B；greedy、seed 42、128 new tokens、原生 EOS。
- 五个严格同序 arm：`raw_rag`、`no_context`、`depolarized_rag`、
  `token_matched_neutral_rag`、`query_term_only_neutral_rag`。token-matched
  arm 分别使用 Huatuo/Hulu 本地 tokenizer，使完整 prompt 的模型 token 数 32/32
  精确等于 raw RAG；不再使用空白分词和尾随空格版本。
- raw/no-context 优先从已冻结完整生成中按 QID 精确提取，不重复推理；三个 neutral arm
  使用统一 GPU lock，在 baseline 作业之间运行，不中断在跑任务。

## 冻结 readout

回答只能由统一 leading-decision parser 映射为 `yes/no/uncertain/invalid`。不得为本实验
增加关键词、语义 judge 或手工修答案。

1. **State alignment**：positive retrieval 对应 `yes`，negative retrieval 对应 `no`。
2. **Signed removal effect**：raw 的 state alignment 减去 neutral arm 的 alignment；若
   firewall 真去除了检索患者状态，该值应为正。
3. **任务风险**：Accuracy、Balanced Accuracy、FP、FN、uncertain/invalid；同时报告
   raw、no-context 和全部 neutral arms，不选择最好 arm 隐藏其余结果。
4. **配对变化**：raw→neutral 的 rescue、harm、unchanged；按 retrieval polarity 和
   finding group 完整列出。
5. **长度/概念控制**：full neutral 必须与 model-token-matched neutral、query-term-only
   neutral 同时解释。只有 full neutral 有效但等长控制无效，视为长度混杂；query-term-
   only 同样有效则不能声称多概念知识被保留。

由于 n=32，bootstrap CI 只作不确定性展示，不用显著性筛选 arm，不做超参数优化。

## Canary 继续/淘汰规则

仅在两个模型同时满足下列方向时进入正式实验：

- 至少一个语义保真的 neutral arm 降低 raw state alignment；
- 相对 raw，FP 下降且 FN 不增加；相对 no-context，Balanced Accuracy 不低超过 1pp；
- 效果不能由长度匹配、统一回答 No、invalid/uncertain 增加或 query-term priming 单独解释。

任一模型出现相反 signed effect，或所谓增益只来自删上下文、缩短回答、提高不解析率，
则停止把 firewall 当方法。观察性 polarity transplantation 可保留为现象，但不能升级为
ICLR 主线。

## 正式因果实验（Canary 通过后才授权）

对同一图像和同一 claim 随机配对 all-present、all-absent、state-neutral donor contexts；
匹配 finding、不同患者、检索相似度、token 长度和写作风格。主量为：

\[
T=\mathbb{E}[m(x,R_{present})-m(x,R_{absent})].
\]

必须在两个模型、至少 3/4 findings 上方向一致且 image-cluster 95% CI 排除 0；firewall
需使 `|T|` 至少下降 70%，并保留非极性知识 probe。只有该阶段可支持 Cross-Patient
Evidence Transportability / Patient-State Noninterference 的论文命题。

### 已冻结的快速因果 pilot

- CXR-only，统一 JPG loader；pleural effusion、cardiomegaly、pneumothorax、lung
  opacity 各 8 个 pair，共 32 个 pair / 160 个五臂输入。第五臂为同图同问题的
  plain/no-retrieval，用于检验 `m_present + m_absent - 2m_plain`，且在任何模型输出
  出现前冻结；原 128 行 v1 逐字作为 v2 前缀保留。
- 固定 SHA256 选择，不使用 target、模型输出或 cosine 排序；所有 32 个 pair 在选择后
  通过预注册质量门：TF-IDF cosine 至少 0.10，present/absent 长度差不超过 10%。
- 主 readout 不需要 target label：对每个 pair 比较 `present` 与 `absent` 条件回答 Yes 的
  差，分别报告四 finding、两模型及配对 bootstrap。`neutral` 与
  `random_deletion` 删除字数相等；只有前者移除 target-state claim。
- 快速继续门：两个模型的 present-minus-absent 方向一致，且至少 3/4 findings 同向；
  从 present donor 删除 target-state 句应使 Yes 倾向下降，而等字数 random deletion
  不能复现。该 pilot 没有单独的 absent-donor neutral arm，这是明确限制；快速 pilot 不使用 p 值宣称确认；
  通过后才打开剩余 76 pairs。
