# Current Research Status

## 2026-09-01 — 参考论文数据缺口已落盘；严格评测双向门通过；跨模型方法矩阵已排队

- OmniMedVQA 已解压并冻结八模态 88,995 题（82,059 unique images），另有每模态8题的
  64题 smoke；MRI/Ultrasound/OCT/Dermoscopy/CT/X-Ray/Microscopy/Fundus 数量与哈希写入
  `data/omnimedvqa/manifest_v1.json`。PMC-VQA v2 official test 33,430题、PathVQA official
  test 6,719题、MMMU medical validation五学科150题也已冻结，缺图均为0。
- MMMU medical 明确包含145道多选与5道开放题，145题单图、5题多图；保留逐图原始路径，
  对仅支持单图的后端提供带 Image 1/2/3 标签的纵向 contact-sheet 回退，不静默丢图。
- 再次发现并修复两类评测错误：列表选项正文中的 `C.`/`D:` 不再被误当标签；空选项列表
  不再被误判为多选。Omni smoke、PMC/Path各512题门、MMMU全150题均通过 perfect=1.0、
  empty=0.0/parse=0.0；invalid GT、missing image 均为0。跨模型推理提示已统一包含MC选项。
- LLaVA-1.6 Vicuna-7B 三个权重分片完整下载并与索引一致。VCD/ICD/DoLa 已接入 Huatuo、
  Hulu、LLaVA-Med、LLaVA-1.6、Qwen 运行器；CVE完成独立论文公式端口和4项专项单元门。
  CLIP后端使用末层CLS-to-patch注意力；无CLS后端明确记录为视觉token重要性代理。
  CVE论文未公开图像算子强度，端口参数会逐条审计记录，不能声称官方逐值复现。
- 新增 AGLA、AvisC、ClearSight/VAF 的 clean-room 跨模型实现。AGLA 已升级为独立旧版
  Transformers 环境中的 BLIP-ITM-large GradCAM prompt matching，再由五个VLM共享同一增强图；
  单条真实 OmniMedVQA CPU 路径通过（similarity=.3017，保留像素=.8491）。AvisC/ClearSight
  暂明确标为 image-space proxy，不冒充 paper-native。专项测试与旧协议回归共12项通过。
- `scripts/run_cross_model_complete_matrix_v1.sh` 已在 tmux 排队，先等待 native/cross Baseline
  进程退出再抢共享 GPU 锁，覆盖 5 模型×8 inference arms×5 冻结 smoke benchmark，并可
  从 smoke 结果继续 full manifest。当前 Hulu CXR-Vishal greedy 为合法5,587题清单，
  进度约4,134/5,587；此前94,981只是 pretty JSON 文本行数，未中断 Baseline。

## 2026-09-01 — OmniMedVQA 数据到位；冻结附件论文全覆盖验收线

- OmniMedVQA 已明确纳入 P0 正式评测。官方 `OmniMedVQA.zip` 已完整下载至
  `/home/dbw/datasets/public/OmniMedVQA/`（10,698,178,715 bytes），压缩目录可读取，
  含 73 个 QA JSON；该条下载阶段记录已由上方最新状态取代。
- 附件论文的最低复现覆盖线冻结为：模型 `HuatuoGPT-Vision-7B`、`LLaVA-1.6-7B`；
  方法 `Baseline/VCD/ICD/AGLA/VAF/AvisC/CVE`；数据集 `PMC-VQA/PathVQA/SLAKE/
  VQA-RAD/MMMU-med/OmniMedVQA/MIMIC-CXR-generation`。OmniMedVQA 必须另报论文中的
  MRI、Ultrasound、OCT、Dermoscopy、CT、X-Ray、Microscopy、Fundus 八模态与 macro average。
- 最终论文表必须完整包含上述全部单元，并在此基础上增加本项目已有模型、方法和数据集；
  不可运行的单元只能以有可审计原因的 `N/A` 出现，不能静默省略，也不能把论文数值当作
  本地复现结果。当前缺口包括 LLaVA-1.6-7B 本地后端，PMC-VQA/PathVQA/MMMU-med/
  OmniMedVQA adapter 与 manifest，ICD、CVE，以及 AGLA/VAF 的可执行/许可合规路径。

## 2026-08-31 15:15 UTC — CE/OE 评测协议修复通过子集门；等待外部 GPU 释放后重跑

- CE 主指标改为严格解析、invalid-as-error；官方 MedHEval `else Yes`/nearest-option
  规则仅作兼容诊断。门禁现按 binary/ternary/choice 实际解析，不再错误调用 OE 门。
- 发现并修复旧 CE manifest 的答案空间错配：CXR-VisHal 161题、SLAKE 68题从错误的
  concise free-text 提示恢复为源 binary/choice 提示；旧 manifest 已归档。Knowledge-MIMIC
  仅修正 patient 聚类，生成问题本身不变。
- 修正协议后的真实子集：12/12 个模型×数据集 CE 单元通过，解析率94.53%–100%、
  invalid GT=0，严格准确率39.84%–100%；VQA-RAD 64例四模型结构门全过，BLEU-1
  .0142–.0975、ROUGE-1 recall .4027–.6992、METEOR .0701–.2184，证明旧 exact≈0
  是长回答口径假低分；Visual-MIMIC 64例四模型 B1 .1385–.2983、R1-F1 .2016–.3668。
- 外部任务仍占34.6GB GPU。tmux `baseline_eval_corrected_v1` 每60秒轮询，显存低于20GB
  后自动启动修正版 native CE 大规模队列；旧 CE scoring monitors 已暂停，防止旧输出
  被错误重绑定。VQA-RAD v4 全量文本指标正在 CPU session `baseline_oe_rescore_v4` 重算。

## 2026-08-31 14:32 UTC — OE 门禁改为结构门；Visual-MIMIC 改按报告任务评分

- 用户授权取消 OE 长度/截断失败门：qualification v3 只拒绝 qid/数量错误、空或碎片输出、
  数据集级坍缩及明显三连 span 循环；token budget 命中率仅保留为诊断。Huatuo
  Visual-MIMIC shared-RAG 490/490 已由旧门禁失败恢复为 admissible，循环率0。
- 确认旧 OE 低分的主要协议错误：Visual-MIMIC 的 prompt/参考均为整份胸片报告，不能以
  short-answer exact 作为结论。报告评分已补齐 BLEU-1/2/3/4、ROUGE-1/2/L、METEOR，
  并生成 task-aware CSV/LaTeX；当前 Huatuo shared-RAG 为 B1 .1255、B4 .0132、
  R1-F1 .2108、RL-F1 .1100、METEOR .2396。VQA-RAD 继续按短答案评测，并新增不惩罚
  冗长回答的 answer-token recall，原 exact/F1 全保留。
- 新 qualification/evaluator 触发全矩阵重新绑定；CPU scoring monitors 正在自动重算，
  覆盖审计从 completed=3 恢复至37且继续增长。LLaVA 八个缺失单元的恢复脚本已重写，
  OE/report safety ceiling 提至512且不作门禁；但宿主侧不可见 GPU 上下文占用34GB，
  首个视觉前向 OOM，队列保持可恢复，等待显存释放。
- OmniMedVQA 官方10.7GB包正在 tmux `omnimed_download_v1` 下载至仓库外数据目录；完成后
  冻结八模态及全模态 manifest，加入现有4模型、多方法矩阵，不追求附件论文数值复现。

## 2026-08-15 23:00 UTC — 十天 Goal 已启动；Baseline 恢复优先，CEB B0 简单无标签分数未通过

- 用户确认剩余 Codex 用量约38%，Goal `019fffd6-4c0e-7112-8f0c-b07569e386f7` 已激活。
  当前策略是 Baseline 绝对优先；Codex 用量不足时只暂停新方向分析，不停止持久 Baseline 队列。
- 已启动持久 tmux `baseline_native_recovery_20260815`，native greedy/beam 正在按既有
  manifest、seed 和评测入口恢复；另有 `baseline_followup_chain_20260815` 等待 native
  退出后顺序运行 native-CE、shared-RAG、report-RAG、cross-model、LLaVA methods 及
  trained/VHR gates。所有段落复用原脚本和 `gpu0-vindr-v2.lock`。
- 重新覆盖审计快照：主矩阵336格，69 completed、1 generated-unscored、2 running/partial、
  87 pending、177 N/A；辅助40格，16 completed、1 running/partial、8 pending、15 N/A。
- 完成 CPU-only B0 `corrected_runs/visual_edge_constraint_v1/huatuo_pairs34_suffix_direct/label_free_analysis.json`：
  selected-support pair AUROC 0.542，candidate-span 0.594，history-persistence proxy 0.646，
  entropy 0.583（image-level entropy 0.719）。简单单候选视觉支持未通过预设0.70门，暂不
  进入在线 CEB；该结果不否定视觉约束潜变量，否定的是直接把 selected support 当部署风险分数。
- 新分析代码为 `anchor/corrected_sgta/analyze_visual_edge_label_free_v1.py`，compile 已通过。

## 2026-08-12 16:51 UTC — 病灶删除—搬运真实门失败

- Huatuo、VinDr Nodule/Mass 128个独立病例、每例原图/删除/搬运三次真实前向全部完成。
  `original-deletion`均值`-0.0254 [-0.1279,+0.0693]`，只有44.5%为正；删除病灶没有稳定
  降低支持。`relocation-deletion`为`+0.1592 [+.0674,+.2500]`，但
  `relocation-original`也过冲`+0.1846 [+.0986,+.2754]`；双向守恒律仅44/128=34.4%。
- 预注册要求两方向CI下界>0、各≥60%同向且admitted subset同向，故正式NO-GO；不通过
  事后选择原本答对的73例挽救。结论是当前反事实编辑响应混有强非临床偏移，关闭
  “局部病灶可安全搬运/直接作evidence adapter”路线。
- GPU已自动进入Huatuo 1,809图逐patch响应采集；该实验使用不同接口，仍按强对照门独立判断。
- SECOND专项复核纠正一个重要表述：当前不能说“SECOND在医学数据上效果失败”。官方递归
  依赖LLaVA-Med不存在的`image_attentions`，关闭态fork与canonical也仅29/32 exact、
  token-F1 0.967，未过一致性门；论文baseline应标`N/A: architecture/backend incompatibility`。
  本项目真正否定的是“局部高响应自然等于临床证据”，不是SECOND官方算法本身。

## 2026-08-12 16:20 UTC — 局部方向完成机制碰撞，主问题收紧为视觉搜索税

- 文献与数学审计确认：病灶裁剪、Top-K、局部注意力、Higher Criticism、scan、
  `sqrt(2 log M)`和conformal校准本身均有直接近邻或属于标准工具，不能作为论文创新。
- 目前唯一值得继续的高风险命题是`Selection–Reuse Inflation`：局部增强先在
  claim×region空间选择最大响应，再用相关的同模型响应验证或放大该区域；即使所有
  claim均不存在，赢家噪声仍可能以约`rho * E[max response]`传入最终commitment。
  公式只是经典winner's curse背景，潜在贡献只能是该规律能统一预测VLM局部方法的FP。
- 已将search-tax致死门限制到七个finding均为三位读者`0/3`的图像：开发166、确认62，
  不再把truth未知的off-claim误当阴性。第一门只检验内部raw search inflation；若不传入
  第二验证响应和最终margin/FP，不得升格为缓解机制。
- Sparse patch主门也已强化：base同时包含finding、final margin、patch mean、max和
  top-5%，scan必须在这个强对照之上增加至少0.02 macro AUROC且CI排除0。端到端搜索
  实验另固定同claim的区域数`16/64/361`以避免病种先验混杂；代码编译、shell静态检查
  和7个单元测试均通过。
- Baseline仍持有GPU锁并持续生成Qwen Visual-MIMIC OE beam；探索队列在同一flock后
  安全等待，没有终止或覆盖baseline输出。
- 16:31补充敌对核查：BCEA 2026已直接覆盖“自适应crop/zoom破坏exchangeability，
  将完整acquisition policy纳入post-acquisition score再校准”。因此
  `Select–Validate–Calibrate`作为算法正式Reject/Pivot；当前实验只检验尚未被覆盖的
  病例级规律——局部方法是否选择并复用自身噪声，且该效应是否传入最终FP。
- CPU竞赛审计得到可复用上限：finding-conditioned Huatuo+Hulu校准在独立960-claim
  confirmation相对dev所选Hulu BAcc `+3.23pp [+.93,+5.60]`，但FP `74->91`，因此只作
  竞赛baseline。480个boxed positives中，最小病灶四分位双模型共同遗漏26.7%，最大
  四分位仅4.2%，确认简单ensemble无法消除共享的sparse-evidence bottleneck。
- 同工作点label-blind placebo显示OPERA/VCD有病例选择性，但DoLa/PAI/VISTA约等于随机
  flip；Visual-MIMIC固定claim数后主要方法F1差异CI均跨0，原始OE差异大量来自少说/空答。
  后续正式门固定为CE matched-operating-point与OE fixed-K+coverage双报告。
- Anatomy-conditional visual null的CPU门关闭：matched donor的像素context距离约为随机
  donor一半，confirmation低级edit detectability AUROC 0.505，但独立dev为0.523
  `[0.504,0.676]`未过上界<0.65；且VinDr全量缺少ViewPosition/orientation/laterality，
  无法建立解剖条件可交换性。每图2个replacement的rank p最小也仅1/3；不排GPU。

## 2026-08-12 16:00 UTC — 首个跨模型/跨划分自然现象通过；真实致死实验已排队

- 新确认 `Sparse Lesion Boundary`：只看3/3 reader-positive claims，在每个finding
  内部计算bbox union面积与最终supported-minus-refuted margin的Spearman，再跨finding
  宏平均。开发集每模型480 claims：Huatuo `0.232 [0.143,0.314]`（6/8同向），Hulu
  `0.475 [0.390,0.544]`（8/8）；fresh artifact每模型133 claims：Huatuo
  `0.323 [0.158,0.462]`（5/7），Hulu `0.415 [0.222,0.559]`（7/7）。四个
  within-finding permutation均`p<=0.0006`。这是“病灶越小，证据越易被全图表征稀释”
  的复制性现象，但尚未证明patch算法或开放式幻觉缓解。
- 已冻结 development-only diagonal-LDA finding directions；下一门使用视觉塔逐patch
  projection和无调参的penalized multiscale scan。只有在fresh 7-finding panel中相对
  `finding+final margin+mean+max+top-5%`增加至少0.02 macro AUROC、AUROC/NLL配对CI下界均>0、且至少
  5/7 findings同向，才在第二模型放量。
- Convex layer mixture门已关闭：Huatuo macro-AUROC `+0.0174 [+.0044,+.0309]`，
  Hulu `+0.0115 [-.0032,+.0277]`，均不到+0.02双模型门。固定零阈值时阳性率分别
  `0.598->0.861`和`0.380->0.605`，BAcc反而下降，确认历史LET增益混有强
  operating-point shift。
- Criterion-Shift Mirage在完整LLaVA CXR-VisHal 5,587题确认：VCD-VISTA严格口径
  `-4.58pp [-5.64,-3.47]`，official proxy却为`+4.98pp [+3.82,+6.23]`；10个方法对中
  5对排名反转。该结果作为所有新方法的防伪评测原则，不作缓解算法。
- Baseline没有中止。探索tmux与baseline共用同一GPU flock；当前baseline持锁。释放后
  自动顺序为：lesion transplant n128 -> Huatuo patch scan -> 仅过门后Hulu patch
  scan -> IU-Xray真实第二观测Huatuo -> 仅过门后Hulu。失败自动停止对应扩量。

## 2026-08-12 12:55 UTC — 全天 ICLR idea 发现与快速证伪 Goal 启动

- 持续 Goal 已创建：内部扩展至少12个机制候选，完成至少6个低成本证伪与3–5个
  真实模型/真实数据致死实验；只有通过预注册门槛的候选才进入跨模型放量。
- 正式 baseline 未中止：四条主要 GPU 队列和评分监控仍在持久 tmux 中；当前 GPU
  运行 LLaVA-Med AvisC × CXR-VisHal。最新覆盖审计为336 cells：46 completed、
  2 generated-unscored、3 running/partial、132 pending、153 N/A。
- 当前探索使用 CPU、现有 logits/hidden-state caches 和 VinDr annotations，不与
  baseline 抢 GPU。三个并行工作面分别负责机制级文献碰撞、VinDr 现象审计和现成
  substrate/致死实验入口审计。
- 第一优先级不是给算法命名，而是找到一个仍未关闭、能解释既有负结果并产生独特
  预测的自然现象；换名层融合、风格/DG复活、DID双中心化和普通全局probe继续禁用。

## 2026-08-12 11:47 UTC — Evidence Addressability 双模型正式门失败；Baseline 已恢复

- 正式endpoint-held-out门使用VinDr固定R8/R9/R10 panel、7 findings × 4 vote bins ×
  19例，共532 claims/532张唯一图；开发集只选表示/探针/正则，确认程序由固定one-shot
  registry开封，selection、prediction、raw feature、代码、holdout和joint SHA独立复核通过。
- Huatuo的`F+M+N+V`相对`F+M+N`：NLL 0.65879→0.65053（相对+1.25%，
  paired stratified bootstrap delta CI [-0.00118,+0.01789]），Brier
  0.12272→0.11852（+3.42%，CI [-0.00012,+0.00860]），7/7 findings方向为正。
- Hulu：NLL 0.61986→0.61088（+1.45%，CI [-0.00249,+0.02043]），Brier
  0.10463→0.10087（+3.60%，CI [-0.00126,+0.00878]），仅4/7 findings为正。
  两模型病例对齐均优于条件残差置换（p=0.000999），说明视觉摘要不是纯噪声，但其
  最终margin之外的增量太小且不稳定；均未达到预注册NLL/Brier各≥5%、CI下界>0、
  ≥5/7 findings及双模型AND门。
- 正式决定为`CLOSE_GLOBAL_SUMMARY_INTERNAL_DECODING_ROUTE`：不进入定位、因果搬运或
  mitigation放量。该结论只关闭raw vision/projector全局mean/std轻量探针，不证明空间
  patch token无局部病灶信息。holdout对旧direct-CE manifest零重叠，但532/532图像ID曾
  出现在其他历史实验，因此只称endpoint-prospective，不称image-unseen。
- 四条Baseline GPU队列已于2026-08-11 18:00:47 UTC恢复。最新覆盖审计：336个主表cells
  中45 completed、1 generated-unscored、4 running/partial、133 pending、153 N/A；另40个
  auxiliary controls中14 completed、1 partial、14 pending、11 N/A。当前GPU正在运行
  LLaVA AvisC × CXR-VisHal，后续不再为本条NO-GO路线抢占GPU。

## 2026-08-10 17:42 UTC — 竞赛提分成立；Fisher/多风格故事被致命对照否定

- 正式 baseline 未中断：十个 tmux 队列仍在；GPU PID 1213358 正在运行 Hulu
  MIMIC-CXR report beam，当前已写入 233/694 份报告。以下新分析均为 CPU-only。
- Knowledge-MIMIC 上完成 patient/study-grouped nested OOF 竞赛上限：五路合格输出的
  logistic stack 将最佳单路 Hulu no-context 的 BAcc 0.8135 提至 0.8279（+1.44pp，
  study-cluster bootstrap 95% CI [+0.19,+2.70]），FP 256→206，但 FN 127→140。
  这是真实竞赛增益，也明确显示它偏向保守，不能宣称同时降低 fabrication 与 omission。
- “Minimum Intervention Basis”的漂亮相关未通过简单控制。Knowledge→CXR 的 Fisher–
  target 原始 Spearman 0.969，在控制 subset size、source-val BAcc、成员强弱和 error
  diversity/correlation 后降至 partial 0.375（p=0.125，增量 R² 0.012）；leave-one-arm-out
  出现符号翻转。当前判决为 `REJECT_FISHER_AS_INDEPENDENT_TRANSFERABLE_SELECTOR`，
  不能把标准 Fisher/LDA + subset search + stacking 包装成 ICLR 方法。
- 重新检验了“多种医学显示风格共同诊断”而非统一风格中心：Huatuo、VinDr 80 个
  0/3 或 3/3 reader-unanimous claims，五种已通过工程审计的 DICOM renders 做5折
  crossfit；canonical-only BAcc 0.725，完整 render-response fingerprint 0.7125，差值
  -1.25pp，image-cluster 95% CI [-7.50,+5.01]。因此多窗响应指纹在当前样本上无效；
  不再扩大 cosmetic-style ensemble。
- 当前最优科学问题不是“选哪个风格/哪个专家”，而是为什么视觉上等价的重复 views
  没有新增信息，而 plain→RAG 等跨证据路径的响应却能纠错。下一门槛先按
  `image-grounded / temporal-history / knowledge-unobservable` 分层重算增益，排除结果
  由不可从单图回答的问题驱动；之后才决定是否存在“正交证据路径”而非普通 ensemble。

### 17:55 UTC stricter superseding audit

- 词法可观察性敏感性分析在 `direct_visual` 3153例上仍显示 +2.76pp（BAcc近似同幅，
  FP 185→146、FN 177→129），因此增益不是由少量病史/时间题主导；但这不是医生标注，
  四臂仅覆盖 binary target 的92.42%，且 Hulu 两臂仍未通过冻结输出质量门。
- 更关键的 patient-alignment 解释被新 placebo 否定。只在**完全相同问题文本**内交换
  RAG 响应、不使用目标标签时，37.0%的 RAG cells 被重新配对；placebo BAcc 0.9134，
  原患者配对码0.9099，原码相对 placebo 为 -0.35pp，cluster-bootstrap 95% CI
  [-0.96,+0.26]。同问题+同标签的更保守 placebo 也未被超过。因此原来的跨患者随机
  shuffle 把“问题变化”误当成“患者变化”；当前结果支持 question-prior/ordinary
  stacking，不支持 patient-specific interventional code，相关机制主张停止。
- error decomposition 仍发现一个不同的可证伪问题：Hulu 加 RAG 后 GT=No 的 FP 从
  10.33% 增至16.57%，而 GT=Yes 的 FN 略降；下一候选改为检验**跨患者检索报告的
  finding polarity 是否被移植到当前患者**。只有检索文本极性→回答翻转的因果链和
  去患者状态的 polarity firewall 同时成立，才进入新论文主线。

## 2026-08-10 17:18 UTC — 病例级干预响应码显著提分；只升格为竞赛工具

- Baseline 未被停止或覆盖：十个持久 tmux 队列仍在，GPU 继续运行 Hulu
  MIMIC-CXR report beam；本轮新增实验全部复用缓存并在 CPU 上执行。
- CXR-VisHal patient-hash blind split 的734例 test 中，验证集选出的最佳单臂
  BAcc 为0.8650；Huatuo/Hulu 的 plain+RAG 配对响应码达到0.9102，提升
  +4.47pp，image-cluster bootstrap 95% CI [+2.17,+6.66]，FP 63→23、FN
  37→42。保留两模型 plain 输出而仅将 RAG 响应跨患者打乱后降至0.8761；
  相对配对码下降3.42pp，CI [-5.25,-1.74]，说明同一病例的 treatment response
  确有超越普通多模型投票的信息。
- 该结果仍不是论文核心：answer-pattern-only 仅+1.19pp且CI跨0；SAC3、ESI、
  VGS-Decoding、QueryBandits 与 BCEA 已分别覆盖语义扰动一致性、干预不确定性、
  医学图像失真对比、逐query策略选择与主动证据获取。当前响应码只作为 Kaggle
  式上限和机制测量工具，不能包装成新 router/ensemble。
- CMP 的 pre-treatment 简单特征确认失败：Knowledge→CXR 仅+0.14pp，CI
  [-0.29,+0.61]，平均仍需1.90次生成且FP上升。CMP 因 QueryBandits/V-ITI
  的直接碰撞和 prefill gate 失败而退出主线，但缓存组合器保留为强 baseline。
- 下一科学问题缩窄为：在多读者 image-grounded claim 上，哪些错误的干预响应
  具有患者特异、可因果搬运的 recoverability，哪些只是共享先验；需要跨模型、
  fixed-coverage OE 和 matched-compute 对照后才能决定是否形成 ICLR 主线。

## 2026-08-10 17:05 UTC — 三个候选被盲确认淘汰；CMP 首个双风险提分信号

- Baseline 长队列与 canonical GPU lock 保持不变；当前 GPU 仍运行 Hulu
  MIMIC-CXR report beam，factorial pilot 只在同一锁后等待，未抢占或终止正式实验。
- FIN 在四档合成 clarity、每档5 seeds 下均未超过普通一致性训练；conditional
  normal-control 在 Huatuo 上相对全局校准 AUROC -1.96pp、Hulu +0.39pp，均未过门；
  不再把“训练中心/条件对照”包装为贡献。
- 中间证据准入 screen 在独立 960 clear claims 上失败：Huatuo 最佳非最终
  macro AUROC 0.645，matched final 0.754；Hulu 为0.8451与0.8446，远低于预注册
  +2pp门槛。固定“早层更正确”再次被否定。
- 新候选 Clinical Mitigation Portfolio 把每个 mitigation 视为可能有副作用的
  treatment，显式联合优化 fabrication 与 omission。Huatuo Knowledge-MIMIC 的
  fit/tune/test 严格拆分 pilot 中，未约束候选在407例test上将 BAcc
  72.75%→78.46%，FP 67→61、FN 45→29；但单域方向安全证书不能稳定保证 FN，
  所以结果只授权多域 Pareto/OOD policy 的下一轮确认，不构成论文结论。
- 当前项目仍未达到 ICLR-ready；FIN、病灶head、条件对照和通用早层证据均已
  降级为负结果，CMP 必须通过二模型、未开封数据、OE fixed-K 与文献碰撞门。

## 2026-08-10 16:25 UTC — 竞赛路由有效，但降级为诊断；FIN 进入首轮致死门

- Baseline 长队列保持运行：正式矩阵 336 cells，当前 24 completed、1
  generated-unscored、173 pending、3 running/partial、135 N/A；GPU 与所有新实验
  继续共用 canonical flock，未中止或覆盖任何 baseline。
- Huatuo 普通/RAG 两探针在 CXR-VisHal 上的 source-trained 自适应路由将 BAcc 从
  81.87% 提至 84.45%，cluster bootstrap 95% CI 为 +1.62 至 +3.53pp，平均 1.67
  次调用；Knowledge-MIMIC→CXR 的 RAG 错误检测 AUROC 从单次 confidence 的
  0.696 提至 response geometry 的 0.804。由于与一致性、反事实 probing 和 ensemble
  文献强碰撞，该结果只作竞赛模块和机制发现仪器。
- 单轴二阶 curvature 回看失败：confirmation n=16/error=5，scalar AUROC 0.782，
  first-order 0.873，加入 curvature 降至 0.600；不把 fancy 概念包装成正结果。
- 新候选 FIN（Factorial Interaction Neutralization）不再寻找域中心，也不强迫
  所有域相同，而只测量/惩罚 render×prompt/knowledge 的混合差分；2×2 Huatuo
  VinDr pilot 已通过 canonical GPU lock 排队，未通过二模型、独立 split 与因果 gate
  前不构成论文主线。完整计划见
  `docs/COMPETITION_TO_ICLR_RESEARCH_PROGRAM_20260810.md`。

## 2026-08-07 16:00 UTC — Domain-Orbit Lesion-Head Routing 未通过因果验证

- 冻结假设：医学影像改变显示风格后，回答翻转是因为 decoder 转向了“跨风格不稳定且
  不关注病灶”的 attention heads；若该机制正确，只抑制这类头应比域敏感头、SPIN
  式低视觉头和随机头更能恢复最差风格下的正确阳性极性。
- HuatuoGPT-Vision-7B 在 VinDr-CXR 的 48 个 3/3 reader-positive、带病灶框病例上，
  使用 6 种 DICOM 显示风格和第 0/7/14/21/27 层完成 probe；dev 32例出现3次预测
  翻转，独立 confirmation 16例出现2次。相关性阶段虽观察到域敏感 head 指标与
  style margin drop 的信号，但病灶富集没有稳定增量。
- 固定每层抑制1个头、scale=0.1的 outcome-aware 因果诊断中，独立 confirmation
  上 domain+lesion 最差风格 margin 改变量为 +0.0567，image-bootstrap 95% CI
  [-0.0586,+0.1722]；相对 domain-only 为 -0.0860 [-0.1762,+0.0113]，相对随机为
  +0.0405 [-0.0878,+0.1761]，均未通过门槛。
- 合并48例只作精度汇总：domain+lesion 为 +0.0674 [-0.0003,+0.1305]，仍跨0；
  相对 SPIN 式低视觉头为 -0.1327 [-0.2500,-0.0236]。在35个 style drop>0.05
  病例中差距扩大到 -0.1760 [-0.3172,-0.0454]。因此“病灶约束带来额外因果价值”
  被证伪，不扩展到更多模型或调 scale。
- 低视觉头抑制在全阳性样本上提高 margin（合并 +0.2001
  [0.1204,0.2860]），但同时也提高原图 margin，且当前没有阴性对照；这只能解释为
  待审计的正类偏置，不能写成幻觉下降。原始结果保存在
  `corrected_runs/domain_orbit_head_v1/`。

## 2026-08-06 15:35 UTC — Right Region/Wrong Direction 两级致死门均失败

- 新纵向主线按冻结计划先审计 CheXTemporal。Gold 1,787行只有1,497个
  pair-finding key；258键含多个 progression，225键连有符号方向也冲突。
  论文语义表明这些很可能是同finding不同位置的异向变化，但公开表没有
  location-to-progression映射；251个可连接bbox的冲突键在各标签行又重复完全
  相同的整组框，无法恢复。Gold/silver严格零重叠，但本机gold图像解析率为0。
  正式决定为 `FAIL_AND_ENTER_STATIC_VINDR_FALLBACK`，不运行temporal adapter。
- 静态fallback使用640 dev与1,920 image-disjoint confirmation claims。Dev选择的
  Huatuo第21层视觉token readout在960个明确claim上macro AUROC 0.743；与最终
  margin等权融合把macro BA从0.6875提高到0.7167，增益+0.0292，image-bootstrap
  95% CI [+0.0059,+0.0519]，略低于冻结+0.03门。更致命的是79条bbox claim的
  ROI-control paired AUC只有0.5449 [0.4615,0.6282]，接近随机。静态projector
  训练未授权。
- 文献碰撞进一步收窄新颖性：TILA/ProTrans已占据方向与time reversal，
  Med-ST/PLURAL/BioViL-T已占据纵向到单图迁移。只剩“真实pair监督到current-only
  冻结生成VLM的claim decision-use迁移，并同时控制FP/FN/off-target”的合取delta；
  当前数据不能验证。权威报告为
  `docs/RIGHT_REGION_WRONG_DIRECTION_FATAL_GATE_20260806.md`。

## 2026-08-03 09:16 UTC — end-to-end red-team blocks premature v4 execution

- A second independent audit found that the repaired Phase1/Phase2 modules are
  still parallel scaffolds: the old v4 prototype analyzer independently uses
  an eigensolver basis, integer sample replication, multinomial/rejection
  bootstrap and its own Haar RNG. No production path yet consumes the new
  continuous multiplier and Phase2 trace contracts. Therefore passing local
  contract tests does not authorize a real v4 analysis.
- Root closed the first patient-provenance spoof path. Patient mode now requires
  an external mapping whose content hash was itself frozen in the Phase1 config
  before outputs. The formal VinDr config has null trusted anchors, so even a
  correctly re-signed `patient_id=image_id` manifest is rejected and VinDr
  remains image-cluster diagnostic-only. The updated Phase1/2 suite passes
  26/26, including self-consistent forged-manifest adversaries.
- Two more P0s remain under active repair: Phase2 must generate and recompute a
  per-model/draw/sign/orbit ledger instead of signing caller-provided arrays,
  and a new formal evaluator must consume Phase1 float64 multipliers plus that
  ledger without any independent RNG, integer cast or sample replication.
  MetaRA remains synthetic/readout-only and semantic folds must eventually
  consume the same trusted cluster plan.
- VinDr mount audit independently confirmed that the deleted bind cannot be
  recovered inside this container. The formal headline union contains 1,120
  unique DICOMs (283 dev, 837 confirmation); only two dev and zero confirmation
  images are local, so 1,118 are missing. Existing complete caches support CPU
  contracts and older diagnostics only, not the formal render-by-wording
  layerwise mechanism.
- There is currently no real fusion-orientation runner or causal intervention:
  the existing factorial runner stores final logits with
  `output_hidden_states=False`. A new adapter-neutral one-forward collection
  contract is being built before further GPU scoring so logits and all required
  hidden states cannot diverge or force a later rerun.

## 2026-08-03 09:08 UTC — evaluation v4 closed; VinDr recovery chain armed

- A non-overwriting ICLR evaluation supplement v4 now binds the completed
  Huatuo/Hulu internal-control T3, LLaVA ten-arm T3, verified blinded review
  archives, three live physician-return monitors, RAG track separation, and
  System/PIH v3 canaries. Focused v2/v3/v4 supplement tests pass 5/5.
- The supplement authorizes generation and physician-pack readiness only.
  Clinical labels are absent, all decoding/RAG efficacy flags remain false,
  and the paper remains fail-closed. Huatuo System exact equivalence passes;
  Hulu's max-logit error 3.0 remains a failure with unchanged tolerance, so no
  cross-model System common protocol is admissible.
- The deleted-source `/workspace/vinbigdata` bind mount cannot be repaired from
  inside this mount namespace because the original host directory no longer
  exists here. No alternate cohort was substituted. A persistent tmux fallback
  `vindr-selective-download` now waits at PhysioNet's interactive password
  prompt for the frozen 2,598-image manifest; `vindr-post-download` monitors it
  and immediately performs the hash/count audit and triplet construction after
  completion. The credential is never written to disk, logs, or environment.
- A source-quiescent full regression then completed on the first attempt:
  982/982 tests pass in 38.05 seconds, with identical start/end source
  fingerprint `d0dbad47b3631da95de5a101bbf4bef56e9dfb8e6092d64f67dc964019527d12`.
  Its PPID-1 state exited cleanly and the result is frozen separately from older
  regression artifacts.

## 2026-08-03 09:05 UTC — command authority closed; v4 contracts repaired and revalidated

- Execution is now permanently non-interactive for this track:
  `sandbox_mode=danger-full-access`, `approval_policy=never`, and
  `guardian_approval=false`. Commands are issued without approval or sandbox
  override fields. Scientific words such as `admission` and `authorized` remain
  evidence gates, not operating-system permission prompts.
- Root review incorporated all three independent v4 red-team repairs. The
  semantic-boundary nuisance predictor now consumes only pre-joint
  `h00/h10/h01` features; `h11` is endpoint/post-hoc description only, with
  model-by-finding fits and patient-cluster class floors. Phase1/2 random streams
  now bind only pre-output design, use a fixed Helmert basis, fail closed to
  diagnostic image clustering without verified patient provenance, preserve
  continuous multiplier traces, and bind MC-SE arrays to the full plan/orbit/
  model/statistic/calibrator/B0 context. MetaRA is explicitly collision/readout
  control only, with content-addressed transform implementations, recomputable
  preservation admission, patient+image split isolation, and stable KL/JS.
- The combined prototype/Phase1/Phase2/calibration/semantic/MetaRA/LORO/System-
  PIH contract suite passes 129/129. None of these synthetic contracts authorizes
  a clinical or mechanism claim; the sole conditional mechanism candidate
  remains fusion-induced clinical orientation selection.
- Historical System/PIH handoffs are again immutable. v2/v3 outcomes live in
  separate append-only disposition artifacts; the focused historical-binding
  suite passes 38/38. Hulu's max-logit error 3.0 remains a failed frozen
  tolerance, and no threshold was relaxed.
- The previously healthy read-only `/workspace/vinbigdata` bind mount now
  resolves to a deleted source directory and exposes no `train/` tree. The
  local PhysioNet area still contains annotations, admission images and eight
  selected DICOMs, but not the frozen canary. CPU contract work and physician
  return monitors continue; new full-image VinDr execution is held until the
  data path is safely recovered rather than silently substituting a different
  cohort.
- The global watchdog and all three OE physician-return monitors remain PPID-1
  detached processes, so VSCode disconnection does not stop them.

## 2026-08-03 08:47 UTC — LLaVA T3 operational gate passes; chained monitoring advances

- The detached LLaVA mitigation T3-v2 run completed all ten arms at 120/120.
  Its frozen generation audit passes every operational gate: exact qid order,
  no reference fields or clinical labels in the generation manifest, complete
  token traces, zero cap hits, 100% nonempty outputs, and no function-word-only
  outputs. `VISTA_off` is generated-token exact with greedy on 120/120.
- Enabled arms are outcome-blind but non-degenerate relative to greedy:
  beam/VCD/OPERA/PAI/AvisC/VISTA-VSV/VISTA-SLA/VISTA change
  80/109/71/40/104/19/27/27 sequences respectively. This authorizes clinical
  review packaging only and is not evidence that any method reduces
  hallucination.
- The postprocess continuation fired automatically and completed with exit 0.
  It selected 32 image-disjoint groups, deduplicated 320 planned assignments
  into 116 blinded answer units, froze the clinical-analysis preregistration,
  produced reviewer A/B archives, and passed archive verification. Clinical
  labels remain absent and efficacy remains unauthorized pending two genuine
  independent physician returns.
- The LLaVA physician-return monitor is chained to launch next. Internal-control
  RadGraph postprocessing immediately acquired the released GPU lock; the
  binding-checked CECD System/PIH canary remains queued behind it. The global
  30-second watchdog covers each generator, continuation, and monitor, so
  VSCode disconnection cannot interrupt the chain.

## 2026-08-03 08:32 UTC — duplicate CECD waiter removed; modality-conflict collision added

- The companion session superseded the original CECD v2 waiter with
  `cecd-system-pih-native-eager-canaries-v2-binding-checked`, whose pre-lock
  verification binds every source plus the frozen DICOM. The recovery watchdog
  briefly restarted the obsolete name before its manifest refresh. That old
  task had acquired no GPU and written no artifact; its exact supervisor,
  child and orphan lock waiter were terminated. Only the binding-checked task
  remains live and recoverable.
- LLaVA T3-v2 completed greedy, beam, VCD and OPERA at 120/120 and advanced to
  PAI. Outcome-blind functional differentiation is non-degenerate: beam, VCD
  and OPERA changed 80, 109 and 71 of 120 answers relative to greedy. This is
  execution evidence only, not clinical efficacy.
- A targeted 2026 collision refresh added the ICML Mechanistic
  Interpretability Workshop paper on mechanistic modality-conflict analysis
  and inference-time control. It occupies generic fusion-stage modality
  preference localization/steering. The surviving CECD delta is now explicitly
  limited to independently clinician-admitted equivalence operations,
  reader-distribution orientation of their product-only residual, and a
  spectrum/norm/marginal-preserving rescue that beats an explicit-conflict
  direction and matched steering control.

## 2026-08-03 08:25 UTC — runtime resumed; active recovery coverage closed

- The persistent-volume sentinel, both model runtimes, PhysioNet metadata and
  the read-only `/workspace/vinbigdata/train` mount survived reconnect. The
  VinDr mount contains all 15,000 DICOM files, `/home/dbw` has about 260GB
  free, and GPU 0 is healthy.
- No valid generation was restarted or duplicated. LLaVA mitigation T3-v2 is
  actively running OPERA while internal-control RadGraph postprocessing and
  CECD native/eager v2 wait serially on the same GPU lock.
- A concrete recovery gap was closed in
  `configs/research_active_jobs.json`: the watchdog now covers the active
  LLaVA generator, both of its postprocess/physician continuations, and both
  internal-control continuations in addition to CECD v2. Its 30-second
  heartbeat reports all six as `alive`; it only recovers dead processes whose
  state remains `running`, and never retries terminal scientific failures.
- Downstream advancement is already persistent and outcome-blind: LLaVA
  generation -> operational audit/blinded physician pack -> return monitor;
  internal controls -> RadGraph/packaging -> two model-specific return
  monitors. CECD v2 remains a mechanical compatibility canary only; successful
  completion cannot bypass the separately frozen behavioral/clinical gate.

## 2026-08-03 08:55 UTC — all OE generation qualified; v4 red-team narrows controls again

- LLaVA mitigation T3 completed all 10 methods x 120 held-out OE questions
  (1,200 answers), exit 0. Every arm has exact qid/trace coverage, zero cap
  hits, 100% nonempty output and zero function-only answers; greedy and
  VISTA-off tokens are exactly identical. The frozen operational gate passed
  and the blinded physician archives plus return monitors were built. This
  authorizes human scoring only, not mitigation efficacy.
- Internal Huatuo/Hulu T3-v2 postprocessing and both physician-pack monitor
  launchers also completed with exit 0. No physician labels exist yet, so all
  clinical-efficacy fields remain false.
- Independent v4 red-team found that the first semantic-boundary control used
  `h11`-derived covariates to predict an `h11`-defined crossing, a mediator/
  post-outcome leak. It is demoted to post-hoc description until replaced by a
  strict `h00/h10/h01` pre-joint predictor. MetaRA logit interaction is
  algebraically homologous to CECD residual and is collision/readout evidence,
  not an independent mechanism. Phase1/2 RNG, per-stratum cluster, continuous
  multiplier and MC-SE trace bindings also remain P0 and are being repaired;
  no real authorizer is connected.
- System/PIH v2 correctly passed both queue-time binding checks but failed its
  Huatuo numerical tolerance (`argmax_equal=true`, max logit error 0.125): BF16
  query chunking altered the eager GEMM shape. v3 keeps Huatuo full-query and
  chunks only Hulu's OOM-prone SDPA clean-room path. Its source/handoff suite
  passes 38/38. The real Huatuo v3 canary passed exactly with zero error; Hulu
  avoided OOM and preserved argmax but failed the frozen full-logit tolerance
  (max absolute error 3.0). Full SDPA-to-eager replacement is therefore killed
  for Hulu. No tolerance is relaxed; a future mechanism-positive branch may
  test a native-SDPA row-local intervention, otherwise this baseline is dropped.

## 2026-08-03 08:31 UTC — T3 passes; CECD retry now has two-sided queue-time binding checks

- Internal-control T3-v2 completed both models with exit 0. The frozen
  generation audit passes all 18 model-by-arm traces: deterministic replay is
  exact, sampling is non-degenerate on 120/120 Huatuo and 119/120 Hulu images,
  and the OE termination gates pass. The newly inserted hash-bound
  question-conditioned output-form audit also passes all 18 records with 100%
  terminal completion on the predeclared sentence-form questions. This only
  authorizes blinded physician packaging, not clinical efficacy.
- CECD v1 passed Huatuo but Hulu failed before comparison when a full near-16k
  Q-by-K map requested another 15.78 GiB. Companion's query-chunked repair now
  passes the independent runtime/factory/preflight suite 48/48 and freezes
  runtime SHA `8c32e51e...`; the old failure is execution-only evidence.
- The first queued v2 process was withdrawn before it acquired the GPU lock or
  wrote an artifact. The launcher now invokes a dedicated fail-closed verifier
  before its blocking `flock` and again immediately after lock acquisition,
  checking every bound source plus the frozen DICOM. Focused tests cover source,
  image and schema drift. Only the new binding-checked detached job is eligible;
  eventual artifacts must still be checked against the frozen handoff hashes.
- LLaVA mitigation T3-v2 is actively generating on GPU 0; internal-control
  RadGraph postprocessing waits on the same lock, so no GPU is idle.
- v4 promotion Phase 1 is now implemented and independently rechecked: exact
  `2 models x 4 findings x 4 vote bins`, config-bound 20/60 dev/confirmation
  quotas, all 15 primary plus four control cells, cross-model orbit identity,
  global patient-or-image clustering, split leakage rejection, and one shared
  strictly-positive multiplier matrix. The focused adjacent suite passes 34/34.
  Haar, calibration and all authorizing logic remain explicitly absent.
- A label-free synthetic MetaRA-style collision substrate now enforces the
  exact clean/image-only/question-only/joint Cartesian orbit, byte-identical
  reuse of each transformed input, cross-model pairing, and held-out image and
  question transform families. Its 15/15 tests pass. It is intentionally a
  non-authorizing CE adaptation rather than a claimed MetaRA reproduction;
  real-result execution remains a later gate.
- A separate semantic-boundary proximity substrate freezes model/tokenizer/
  processor/extractor, layer and present/refuted text-proxy identities; fits a
  fixed L2 logistic model only on patient-clustered dev generic boundary
  crossings; and applies continuously on disjoint confirmation orbits. It
  rejects reader, PAEL and clinical outcomes and exposes no tunable threshold.
  Its joint suite with the diagnostic v4 code passes 17/17. This is a strong
  nuisance explanation, not an implementation of robustness certification.
- v4 Phase 2 now freezes 2,048 independent centered-subspace Haar draws and
  their `H/-H` antithetic partners (4,096 references), shared across paired
  models by image/finding/vote key. MC-SE treats each pair as one independent
  unit; only the frozen `MC-SE/B0 > 0.005` rule can trigger one 8,192-reference
  doubling. Near-zero or bootstrap-unstable B0 fails closed, and the seed
  binding excludes model scores and clinical outcomes. The adjacent suite
  passes 26/26. This remains a stress reference, never a randomization test.
- The calibration-admission substrate now tests raw canonical score means in
  all four reader-vote bins before isotonic fitting, compares the full support
  span with hash-paired same-support image drift, forbids confirmation clipping
  across actual/additive/render/prompt score families, and guards near-zero B0.
  Root review corrected two pre-result bugs: model-specific bootstrap stream
  keys are now one shared key, and cluster-mode drift/patient overlap cannot
  evade split checks. The updated adjacent suite passes 15/15. Aggregate votes
  are still explicitly insufficient until the separate named-reader LORO gate.
- A true named-reader LORO substrate now rejects aggregate/anonymous vote
  positions and uses the fixed official R8/R9/R10 panel, excluding each held-out
  reader from its dev target. Its synthetic/Phase-1 suite passes 25/25. It is
  not currently runnable as patient-level inference: the v2 manifest has no
  patient IDs, and a direct 2,000-DICOM metadata audit found both `PatientID`
  and `StudyInstanceUID` empty throughout the sample. VinDr documentation does
  not guarantee one image per patient. The module therefore fails closed rather
  than relabeling image IDs as patients; LORO remains sensitivity-only and
  cannot promote v4 without independent patient provenance.

## 2026-08-03 08:00 UTC — target-coupled v3 killed; independent Brier–PAEL v4 implemented

- Runtime authority is already fully open: `sandbox_mode=danger-full-access`,
  `approval_policy=never`, and guardian approval is disabled. No command in
  this track requests an operating-system approval; `admission`, `authorized`,
  and `blocked` in artifacts refer only to prespecified scientific evidence
  gates.
- An independent estimand audit found that the v3 candidate algebraically
  reconstructs the same cell score whose sign defines its error label: maximum
  synthetic reconstruction error was `1.776e-15` and exact reconstruction held
  for every target. v3 delta-AUROC is therefore retained only as descriptive
  attribution and cannot authorize the paper claim.
- The v3 fail-open paths were closed without opening outcomes. Admission now
  requires the exact four nonbaseline renders and two candidate prompts used by
  the runner. The three-stage verifier reloads bound raw dev/confirmation
  payloads, independently recomputes the fit/apply statistics and every gate,
  and rejects empty metrics, asserted booleans, tampering and obsolete schemas.
  The affected chain passes 31/31 tests and the strict static DAG has zero
  blockers while waiting for genuine inputs.
- A separate non-authorizing v4 CPU prototype now estimates the sole proposed
  primary: 16-stratum, orbit-first, reader-distribution Brier `PAEL_Haar`, the
  observed product-loss orientation beyond a centered-subspace isospectral
  Haar reference. Haar is explicitly a deterministic stress reference, not an
  exchangeability/randomization law; whole-image cluster bootstrap supplies
  sampling uncertainty, with shared draws across models. Matched, cell, sign
  and NLL analyses are sensitivities. Source/seed/split drift and confirmation
  refitting fail closed. Root verification passes 20/20 tests.
- A fresh 2024–2026 collision audit kills PAEL as standalone metric/method
  novelty. MetaRA already combines paraphrases with benign/style/background
  image transformations; IEEE TSE composite-metamorphic-relation work already
  studies failures revealed only by composition; ICML 2026 Semantic Robustness
  Certification already occupies text-proxy semantic planes and norm-preserving
  rotations. PAEL remains only a confirmatory estimand. The paper is now a
  conditional mechanism candidate, not a New Problem/Setting paper.
- The mechanism hypothesis is narrowed from generic information erasure to
  **interaction-energy to clinical-orientation formation**: a model-specific
  fusion-to-decoder transition may rotate existing render x wording
  interaction energy toward reader-grounded loss. This remains falsified if
  norm/spectrum, entropy, marginal sensitivity, prompt heads, reader aliasing,
  or random/ispectral controls absorb it. No layer or mitigation experiment is
  authorized before behavioral GO. After GO, a valid mechanism requires a
  dev-selected orientation jump and at least 20% PAEL rescue by an upstream
  spectrum/norm/marginal-preserving patch with no more than 1pp clear-case loss
  in both models; generic composite failure and semantic-boundary proximity are
  mandatory controls.
- The old 60-claim repeated clinician design was rejected because recall can
  manufacture apparent human stability and signed effects can cancel. The
  frozen preferred control is 240 unique images, four fixed clinicians, and a
  Latin-balanced one-cell-per-clinician-per-claim allocation with direct 0–100
  support probabilities and Brier loss. Signed, non-cancelling, assessability,
  and model-over-panel gates are conjunctive; inference is fixed-panel only.
- Persistent T3-v2 generation completed all 18 Huatuo/Hulu arms at 120/120.
  The frozen operational audit passed: zero cap hits, 100% nonempty answers,
  zero function-only outputs and exact replay coverage. Label-free
  postprocessing is now building the 1,440-report blinded physician substrate;
  packaging and return monitors remain chained behind their own gates.
- The first corrected real System/PIH canary passed Huatuo native/eager exactly
  (`max_absolute_error=0`) but Hulu failed closed before artifact creation: a
  full eager `Q x K` attention tensor requested 15.78GiB. A non-overwriting v2
  implementation chunks only the query rows (size 256), retains native-dtype
  row matmul, FP32 softmax and the exact post-softmax/pre-value patch boundary,
  and releases full logits between native/eager passes. Focused tests pass
  33/33. Its PPID-1 retry is queued on the shared lock behind the companion
  session's LLaVA T3. Disk reserve remains above the frozen 100GB floor.
- Independent promotion red-team keeps v4 diagnostic-only. Authorizing P0s
  still include exact 16-stratum/model/grid quotas, global cluster identity and
  paired orbits, 4,096 antithetic Haar plus MC-SE doubling, raw directional and
  calibration-support gates, stable `B0`, named-reader LORO, executable
  MetaRA/boundary controls, and independent recomputation of clinical/human
  gates. Phase 1 now addresses only the exact schema/pairing/cluster/bootstrap
  contract; no outcome or GPU is opened.

## 2026-08-03 07:12 UTC — LLaVA mitigation T3 is label-firewalled and fully chained

- The historical 200-row LLaVA mitigation matrix is not reusable clinical
  evidence: its sampled arms were stopped into one-token outputs. A new T3
  contract therefore runs greedy, beam, VCD, OPERA, PAI, AvisC and four VISTA
  arms on 120 image-disjoint VQA-RAD OE samples at one 512-token limit with
  keyword stopping disabled and full generated-token/terminal traces.
- Generation now consumes a reference-redacted manifest. The legacy port omits
  `gt_ans` when labels are absent, and the outcome-blind audit requires exact
  qid/trace coverage, <=5% cap hits, >=95% nonempty and sentence completion,
  <=1% function-only output, and 100% greedy/VISTA-off token identity. Passing
  these gates authorizes only a physician pack, never clinical efficacy.
- Before any new output was available, 32 independent images were frozen by
  question-family x reference-length strata. The private join utility is the
  first authorized reference join and rejects failed audits, source hash drift,
  output reference leakage, altered qid order, or non-disjoint selection.
  Ten methods imply 320 planned method assignments; exact duplicate answer
  strings are reviewed once while the private mapping retains every method.
- The clinical analysis contract is frozen across nine candidate arms with
  Holm correction and paired visual-error, matched-coverage, omission,
  correctness, harm, refusal, length and evaluated-claim gates. Delivery
  instructions now derive the calibration count from the actual manifest
  instead of saying “ten” unconditionally. Focused generation/join/archive
  tests pass 11/11.
- Three detached PPID-1 v2 stages are alive: the T3 generator waits on the shared
  GPU lock behind internal-control T3-v2; a continuation immediately builds and
  verifies blinded A/B archives only after operational pass; a second then
  starts the physician-return monitor. Failed operational qualification stops
  the chain without producing a review pack or a mitigation claim.
- A queue-time provenance race was removed before any LLaVA sample was
  generated. The superseded v1 job had checked source hashes before waiting on
  the shared GPU lock; v2 checks the expanded method-critical bindings both
  before waiting and again after acquiring the lock. The unstarted v1 job and
  its two continuations are retained as failed/superseded states; v2 is the only
  authorized queue.
- The post-change full suite passed all newly added LLaVA/physician tests, but
  its first source-quiescent whole-repository attempt is truthfully red rather
  than admitted: **869 passed, 3 failed**. All three failures are in the
  concurrent CECD track (`reader_threshold_alias` source hash,
  `stage1_power_audit` analyzer hash, and `verify_cecd_three_stage_v3` fixture
  loading); they do not touch LLaVA outputs. The failed regression artifact and
  log are retained. This session will not overwrite the companion session's
  CECD sources or refresh its committed scientific hashes behind its back.
- The companion CECD session subsequently repaired its own three failures. A
  persistent focused gate passed 17/17 and automatically launched a fresh full
  regression: **876 passed**, 63 dependency warnings, zero failures, with
  identical pre/post source fingerprint
  `0c73f682...a38d7f32`. The earlier red artifact remains preserved separately.
- Baseline coverage no longer misreports an executed negative result as
  “missing.” Calibrated abstention was run on the frozen disjoint T2 substrate,
  but Huatuo had 0/16 positive lexical-proxy rows, 100% validation coverage and
  zero selective claim actions; post-outcome threshold tuning is prohibited.
  The append-only registry now records `failed_cutoff`, method ladder v9 and
  coverage audit v5 show zero executable T2 gaps and one explicit T2 failure,
  and focused registry/ladder/control tests pass 10/10. This is experiment
  completion, not a mitigation success.
- ICLR completion audit v3 now consumes ladder v9, internal-control audit v4
  and coverage audit v5. It remains correctly `paper_ready=false`: R1/R7 await
  real clinicians; R2--R4 still lack a replicated mechanism/causal method;
  R6 is execution-partial despite complete T2 accounting. A final
  source-quiescent regression after these changes passed **878 tests**, 63
  dependency warnings and zero failures with identical fingerprint
  `3feaa008...6c4e59d6`.

## 2026-08-03 07:01 UTC — CECD execution closed; headline narrowed by independent red-team

- Runtime command authority remains fully open (`danger-full-access`, approval
  `never`). Remaining gates are scientific-validity contracts, not shell
  permissions. An independently red-teamed outcome-blind DAG v2 now reports
  `static_handoffs_ready_waiting_genuine_inputs` with zero static blockers. It
  verifies the clinical, dual and listing monitor identities, AST-level
  transition calls, stable registry/source snapshots and one canonical GPU
  lock; this does not assert that any human or scientific gate has passed.
- Dual transition v3 now builds a hash-complete preflight and formal-CE-only
  launch handoff. Listing transition v2 binds eight independent return and
  attestation files, two completed adjudications, explicit admit/reject,
  canonical upstream GO, validator/assembler sources and a serial
  pilot→dev→confirmation scheduler. Preparation never launches a model;
  partial or tampered output is terminal instead of silently replayed.
- The latest 2026 collision audit and independent paper red-team kill
  Reader-Grounded Two-Plane/RCCP as a headline or mitigation contribution.
  CECD survives only as the conditional new problem of a clinician-admitted
  render×wording product-only clinical residual beyond both marginals, generic
  two-axis instability and behavioral synergy. The skeleton is Accept with
  Revisions, not oral-ready; unrestricted OE/report and decoding novelty are
  outside the core paper. Failure of the locked gate terminates the framing.
- System/PIH Qwen2/Qwen3 runtime integration is source-ready but scientific
  preflight remains false. The first Huatuo canary failed before artifact
  creation because `python -m` duplicated a provenance dataclass under
  `__main__`; Hulu was not loaded. The bug now has a canonical-module alias and
  real two-interpreter subprocess regression. A corrected PPID-1 canary is
  queued behind the same GPU lock as active T3-v2 and cannot overlap it.
- After these repairs, a new source-quiescent full regression passed **858
  tests** with 63 dependency deprecation warnings in 31.50 seconds. Attempt 1
  correctly detected source drift and did not test; attempt 2 had identical
  pre/post fingerprint `eefc4316...2992610` and zero failures.

## 2026-08-03 06:53 UTC — persistent restart verified; T3-v2 and downstream monitors active

- The persistent dataset/model mounts, 223GB disk reserve, GPU 0, and the
  PPID-1 research watchdog all survived the editor/container recovery. No
  generation job was duplicated. T3-v1 is finishing Hulu and is already bound
  to the outcome-blind T3-v2 512-token repair, qualification, and physician
  monitor chain.
- The downloaded FactMM-RAG ZIP is a complete two-shard Hugging Face checkpoint
  group (718 indexed tensors), not an archive with zero checkpoints. Config and
  index evidence classify it as a LLaVA causal generator with an embedded CLIP
  vision tower and multimodal projector, not the paper's DPR/ANCE retriever.
  The erroneous v1 semantic/T0 artifacts remain immutable diagnostics.
- The role-correct FactMM chain completed. Both shards passed CRC and
  `weights_only` inspection: 718/718 indexed tensors, 7,062,904,832 parameters,
  zero missing/unindexed/wrong-shard keys. T0-v2 correctly records generator,
  projector and vision tower present but retriever/MIMIC/CheXpert absent, so
  paper-native T0 is `not_admissible`. RAG dual-track-v2 and ICLR supplement-v3
  are frozen, and the source-quiescent full regression passed 853 tests with an
  identical pre/post fingerprint.
- T3-v1 completed and its immutable disposition records
  `identity_and_length_stress_only`, `physician_pack_authorized=false`, and no
  clinical-label access. Its continuation immediately launched T3-v2; the real
  Huatuo `greedy512` arm is generating. Postprocessing and physician-return
  monitors remain chained behind the frozen v2 qualification gate.
- Focused sharded-checkpoint/qualification tests pass 5/5. A concurrent CECD
  change legitimately closed its listing-adjudication-producer gap; the stale
  regression assertion was narrowed to the still-open blockers and its focused
  suite now passes 16/16.

## 2026-08-03 06:22 UTC — held-out OE T3 repaired before clinical packaging

- Mid-run outcome-independent qualification found that Huatuo's nominal
  256-token clinical sampling arms hit the generation ceiling on 5.8%--19.2%
  of 120 held-out images; the frozen ceiling was 5%. `greedy128` also hit its
  ceiling on 17.5%. JSON, qid, token, NLL and seed traces were valid, but these
  answers are not eligible clinical efficacy evidence.
- The exact failure is frozen in
  `internal_controls_t3_v1_huatuo_cap_failure_prereg_v1.json` before any
  physician labels existed. T3-v1 will finish for identity/length-stress
  evidence only; its physician-pack continuations were terminated and retained
  as superseded failed jobs, so truncated answers cannot silently enter the
  paper.
- A byte-identical 120-image manifest and outcome-blind T3-v2 contract are now
  frozen. Both VLMs and all clinical arms use one 512-token ceiling. V2 must
  pass cap-hit <=5%, nonempty >=95%, function-word-only <=1%, exact seed replay
  and non-degenerate sampling before RadGraph or physician packaging begins.
- The repaired detached chain is
  `internal-controls-t3-v2-repair-continuation-v1` ->
  `internal-controls-t3-v2-postprocess-continuation-v1` -> model-specific
  physician return monitors. The legacy extraction hard-code for `greedy256`
  was removed, and physician preregistration now fails closed when CLI methods
  differ from the frozen clinical contract.

## 2026-08-03 06:04 UTC — T3 and FactMM-RAG continuations persist independently

- Held-out VQA-RAD T3 remains a PPID-1 job and has progressed through the
  Huatuo control matrix. Two dependent supervisors will immediately run trace
  audit/RadGraph aggregation/physician packaging and then launch model-specific
  return monitors. Generation alone cannot promote a method to clinical pass.
- The official FactMM-RAG Drive object was transport-audited before use. Its
  response is an 11,200,064,402-byte `model.zip`, not a directly loadable `.pt`;
  the misidentified single-stream attempt is retained as failed provenance.
  A 16-range PPID-1 download now runs at roughly 10--12 MiB/s while preserving
  the 100GB disk reserve. Completion automatically triggers a non-extracting ZIP
  inventory and then a fail-closed `weights_only` audit only if exactly one
  retriever-like checkpoint is unambiguous. Neither asset availability nor a
  valid tensor schema authorizes paper-native end-to-end efficacy.
- Full regression first exposed and retained a concurrent CECD syntax/shape
  failure rather than skipping it. After that track fixed the issue, the final
  source-quiescent snapshot passed **778 tests** with identical pre/post source
  fingerprint `58f698639421ff95742eabd16ec35fa185364a145f21f1d9a46a3ac3890c507e`.
  The admitted artifact is
  `full_regression_after_t3_and_factmm_semantic_chain_v2.json`.

## 2026-08-03 05:22 UTC — CECD v3 and independent VinDr listing admission are executable

- Runtime permissions are already frozen as `danger-full-access`,
  `approval_policy=never`, and `guardian_approval=false` in all three active
  Codex configs. No shell/filesystem approval remains in the execution path.
- The CECD outcome-blind design is now a real three-stage pipeline:
  `pilot_screen` uses 160 claims/model only as an operational canary,
  `dev_fit` uses an image-disjoint 320 claims/model to serialize every scale
  and predictor, and `confirmation_locked` applies that frozen predictor once
  to 960 claims/model. Exact selection hashes, zero cross-stage image overlap,
  model/checkpoint identity, analyzer code, fixed seed/folds/5,000 bootstraps,
  and dev-fit binding are independently verified. The focused gate passed
  59/59 tests and the complete CECD/clinical-equivalence/Treble collision suite
  passed 122/122. Legacy pilot-as-dev artifacts cannot authorize a conclusion.
- Only one admission monitor (supervisor/child 595803/595805) and one
  dual-transition monitor (596222/596224) remain. They are detached under
  PPID 1 and wait for four independent returns / the locked two-model result;
  no clinician response or sealed outcome was synthesized or inspected.
- A real Huatuo/Hulu single-token CE adapter now evaluates a two-render by
  two-prompt orbit in centered Yes/No/Maybe logit space and implements seven
  architecture-neutral controls with atomic cell shards, write-once config,
  corruption recovery, and a shared nonblocking GPU lock. Both models passed
  CPU-only preflight on the same local VinDr claim and the adapter suite passed
  33 tests. Formal CE wiring is underway; CECD hidden intervention, both Treble
  semantics, and aligned OE remain fail-closed rather than emitting placeholders.
- The independent 14-finding listing admission pack is built from the entire
  frozen 60-image pilot: 20 normal, 20 single-finding, and 20 multi-finding
  images; 252 blinded render pairs and 504 PNGs. Its integrity verifier passed.
  Four role-isolated deliveries also passed archive-byte verification: two
  clinical archives of about 2.6GB each plus separate clinical-template and
  language packages. No sealed mapping/truth was delivered and no prior polar
  prompt decision was reused. One computational guard-fail pair remains visible
  for blinded review but is permanently ineligible for a future complete model
  orbit; no threshold was repaired after observing it.
- The August collision refresh further narrows novelty. ACL/EACL 2026 already
  cover evidence-bounded minimal editing, prompt-copy heads, system-mediated
  yes-bias, adaptive bi-causal steering, generic pre-generation probes, and
  claim-level confidence. HalluCXR makes length and omission mandatory
  confounds. CECD therefore survives only as an incremental reader-grounded
  clinical-error mechanism unless a later fixed-K/no-deletion method beats
  CEBC-style editing and compatible dynamic controls.
- The seven centered-tristate logit controls now have a hash-bound formal
  CE-only shared-cache stage. Fake end-to-end tests confirm one `4N` model pass
  per model rather than seven repeated passes, 14 atomic model×method shards,
  and zero new scoring on replay; the focused suite passes 35/35. Real Huatuo
  and Hulu four-cell engineering smokes on the same VinDr claim both completed,
  while remaining explicitly non-scientific. No formal run exists because the
  locked authorization/preflight/input gate is absent; CECD hidden, both Treble
  semantics, and OE remain fail-closed.
- An executable adversarial method preflight now freezes the missing controls
  exposed by HalluCXR, system-attention yes-bias, prompt-copy heads, HALP,
  CEBC, HalluTrace, VLI and ConRad. Its truthful state is
  `blocked_mechanism_paper_scope_only` with 14 blockers,
  `mitigation_novelty_authorized=false`, and `paper_claim_authorized=false`.
  The complete CECD regression passes 133/133. This makes the scientific
  decision explicit: locked CECD failure terminates the branch; a pass supports
  only a narrow mechanism claim until simpler causal alternatives are closed.

## 2026-08-03 04:58 UTC — CECD formal design corrected for power; VinDr multi-claim listing remains bounded

- The strengthened behavioral gate exposed a fatal prospective power defect in
  the old 160-claim plan. Requiring each finding to have both
  `delta-AUROC >= 0.03` and a positive CI gives a per-finding asymptotic pass
  probability no greater than 0.5 at the MCID; the old 3-of-4/two-model rule is
  capped near 9.77% even with infinite sample size. The pilot-as-`dev` payload
  alias is also a provenance error and is marked
  `MUST_FIX_BEFORE_NEW_FORMAL_OUTPUT`.
- The outcome-blind replacement uses three image-disjoint selections already
  present locally: `pilot_screen` 160 claims/154 images for engineering only,
  `dev_fit` 320/283 for fitting and freezing, and `confirmation_locked` 960/837
  for one apply-only test. The primary test is pooled image-cluster delta-AUROC
  with MCID 0.03 and positive CI; 3/4 findings become directional/heterogeneity
  guards. At planning effect 0.05, confirmation has about 94.2% two-model
  full-gate power under the frozen central assumptions. Estimated compute is
  6.75--11.25 GPU hours and 0.30 GiB; no new DICOM download is needed. The
  three-stage pipeline migration is underway before any admitted output exists.
- VinDr supplies a credible intermediate beyond binary CE, with a bounded claim
  ceiling. The eight-finding track has 2,341 images, 18,728 claim cells and 609
  true multi-claim cases. A new 14-finding track freezes 420 images/5,880 claim
  cells, including 140 multi-unanimous cases, from a 5,501-image panel with
  1,360 such cases. This is only an explicit closed-ontology, open-cardinality
  listing substrate after a new prompt/render admission. It is not free OE:
  3,620/5,501 images contain a unanimous abnormality outside the ontology, and
  patient identity is unavailable, so only image-disjoint claims are legal.
- The controlled-comparison executor now accepts only a write-once
  authorization, freezes the exact model/method closure, uses atomic CE/OE arm
  shards, a GPU lock and audited recovery, and fails before worker/runtime/GPU
  when authorization is absent. The scientific worker remains incomplete by
  design; no placeholder method output is permitted.

## 2026-08-03 04:36 UTC — internal baseline controls become machine-enforced

- Added a frozen machine-readable contract for temperature/length controls,
  claim-level self-consistency, and calibrated abstention. It binds disjoint
  dev/test hashes, seed/token traces, claim-preserving aggregation, calibration
  provenance, matched coverage, omission accounting, and no post-hoc
  truncation.
- New fail-closed qualification artifact reports all three controls T1-pass but
  T2-missing, with no T3/full promotion. Its fingerprint is
  `dce283afcd8d8cbb03b9c0c0deb0eab9d525145aeefdb360f268588f7177cdff`.
- Preserved baseline coverage v1 and generated non-destructive v2. V2 closes
  all 24 configured methods while binding the internal-control contract;
  `paper_main_table_authorized=false`, fingerprint
  `1fefba19439faa2a73480046be5f591118714c0574e740482dde8b73d35126d6`.
- The ICLR completion packet now consumes v2 and independently verifies the
  internal-control artifact. Two builds were byte-identical; current completion
  fingerprint is
  `4d73569cc5ebf9c3567d548320e84991b2b7dbe37e2d1e8af9fc17b7094ab987`.
- This work did not touch the companion session's active CECD analyzer/runtime
  changes. It only strengthens the shared evaluation and claim firewall.
- The first shared-tree full regression correctly refused four passing but
  source-drifting snapshots. The validation chain now waits for a 12-second
  quiescence window before testing. Its v4 run then admitted a stable snapshot:
  **653 passed**, deterministic completion rebuild, and successful automatic
  continuation handoff.

## 2026-08-03 04:30 UTC — restart recovery and autonomous continuation verified

- The persistent `/home/dbw` mount is present and writable; models, repository,
  restricted datasets, and the trace-certified runtime remain at their frozen
  absolute paths. About 230GB is free and GPU 0 is healthy/idle.
- The PPID-1 research watchdog and four fail-closed transition monitors are
  alive after editor/container recovery: physician OE, CECD clinical admission,
  CECD dual-semantics transition, and PCEM ECHO access. Each monitor advances
  the next preregistered stage immediately after stable valid input appears;
  none synthesizes clinician labels, credentials, or authorizations.
- No GPU job was blindly restarted. The current gates wait for real independent
  review returns or an explicitly mounted protected table; rerunning completed
  generation would create duplicate evidence without resolving those gates.
- Frozen `docs/INTERNAL_BASELINE_CONTROL_CONTRACT_20260803.md`. It defines
  development-only tuning, claim-level self-consistency, matched-coverage risk,
  and abstention accounting. Self-consistency and calibrated abstention remain
  intentionally T2-missing until a disjoint development substrate exists.

## 2026-08-03 05:40 UTC — unified internal controls T2 audited; held-out T3 chain running

- Frozen an official-train, official-test-image-excluded VQA-RAD development
  substrate (32 independent images) and ran the same native adapter matrix for
  Huatuo and Hulu. Temperature/length and structured-claim self-consistency
  pass functional T2 only; neither has T3 clinical efficacy authorization.
- A pooled abstention gate initially hid a degenerate Huatuo result. The
  append-only correction records 0/16 positive calibration proxies, 100%
  held-internal acceptance, and zero uncertainty actions for Huatuo. The old
  event is now `not_admissible`; calibrated abstention is T2-missing. Hulu's
  non-degenerate result cannot rescue a supposedly common two-model method.
- Current fail-closed artifacts are `method_evidence_ladder_v8.json`,
  `internal_baseline_control_qualification_v3.json`, and
  `baseline_coverage_audit_v4.json`. Only temperature/length and
  self-consistency are T2-pass; T3/full remain empty and no efficacy table is
  authorized. The non-destructive ICLR audit is in
  `corrected_runs/paper/iclr_oral_completion_audit_v2/` and remains not ready.
- Source-quiescent full regression passes 725 tests. The recorded source
  fingerprint is stable across the run; warnings are 63 existing dependency
  deprecations, not failures.
- Official test OE has 200 questions but only 120 independent images. A
  label/output-blind hash rule froze one question per image as
  `vqa_rad_internal_control_t3_n120_v1.json`. Persistent PPID-1 job
  `internal-controls-t3-n120-v1` is generating ten frozen arms for both models.
  Its continuation automatically audits traces, extracts 1,440 RadGraph
  reports, aggregates structured self-consistency, and creates separate
  32-image blinded physician packs. A second continuation then starts two
  model-specific return monitors; no monitor synthesizes labels or promotes
  automatic metrics to clinical truth.

## 2026-08-03 05:16 UTC — image-disjoint internal-control substrate and monitored T2 run

- Downloaded the official VQA-RAD train parquet at fixed Hugging Face commit
  `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`; exact size/SHA-256 are
  `24,183,983` bytes and `b07c3441467b99060e5ec412ddd05be06f86f01f23bfa3debfbbcab47874a06e`.
  A concurrent partial-download collision was detected, stopped, and preserved
  as a corrupt forensic copy; only the atomically installed verified object is
  admitted.
- Content hashing all official images shows 202/313 train images also occur in
  test, affecting 1,059/1,793 train questions. Excluding every test image leaves
  111 independent train images, 734 questions and 334 genuine OE questions over
  91 images. The fixed T2 pilot uses 32 questions from 32 distinct images and
  has zero test-image overlap; freeze fingerprint is
  `a76cabb5becb712559d06dc9712e27cea0bf576500409386985f16c5889df8d7`.
- Huatuo now uses the same `generate_control` interface as Hulu/LLaVA while
  retaining native prompt, padding, `min_new_tokens=1` and repetition penalty.
  A method-off identity canary is text- and token-exact (77/77 IDs) against the
  historical native output and additionally records processed-token NLL.
- The two-model n=1 matrix passed all 20 arm checks: exact qid/token/NLL traces,
  five-seed non-degenerate sampling and same-seed deterministic replay. Audit
  fingerprint: `6056bd0d443601c17395b6b2b474f6a389a482c123c9ae006c52f6d53b1934eb`.
- Persistent job `internal-controls-t2-n32-v1` is running the frozen n=32
  matrix with one model load per family. PPID-1 child monitor
  `internal-controls-t2-n32-continuation-v2` waits on its state and immediately
  runs hash revalidation, prediction-side RadGraph structure extraction,
  claim-level self-consistency, 16/16 abstention calibration, registry updates,
  and baseline-coverage v3. No held-out answer or clinical judge is read.
- RadGraph surface-claim and aggregation smoke passed under the trace-certified
  Transformers 4.51 environment. Unparsed OE answers are retained unchanged;
  no exact-text majority, claim deletion or abstention-as-correction is allowed.
  T2 may qualify functionality only; T3/full efficacy remains fail-closed.
- Detached validation `post-restart-validation-chain-v2` completed with
  **634 passed** and exit code 0. Its child continuation monitor observed the
  terminal state and executed the recovery handoff with exit code 0. The
  completion audit rebuilt deterministically with fingerprint
  `721ac77f0ef43b2e4d6218e1ad6870b9a7460fda42d1606566b504cb3d2b1754`;
  `paper_ready=false` remains correct.
- A transient observability race was found and repaired: one-shot recovery
  checks now write a dedicated heartbeat instead of temporarily replacing the
  long-lived watchdog heartbeat. The persistent watchdog is healthy at PID
  `142107`.

## 2026-08-03 04:13 UTC — restart validation chain is terminal; automatic handoff remains live

- `/home/dbw` is mounted read-write with 235GB free, the persistent-volume
  sentinel is present, and GPU 0 is idle. The PID-1-adopted recovery watchdog
  remains alive as `142100/142107` with fresh 30-second heartbeats.
- Watchdog-registered child job `post-restart-validation-chain-v1` ran outside
  the VSCode process group as supervisor/worker `549912/549913`. It completed
  normally and is now recorded `terminal`; its source-drift guard remained
  stable on the first attempt and the full suite passed 626 tests.
- The job immediately rebuilt the completion audit twice and compared all
  outputs byte-for-byte. Its fingerprint is
  `2acf7889f1df0418f26d17ab375967025c10a96586214763b989fa246fb22ec3`;
  `paper_ready=false` and `submission_claim_authorized=false` remain enforced.
- Automatic continuation is already armed. Physician-OE, CECD clinical
  admission, CECD dual-semantics transition, and CPU-only PCEM-ECHO monitors
  remain alive; each validates and launches only its frozen next stage as soon
  as its external gate closes. Closing VSCode/SSH does not terminate them.

## 2026-08-03 04:12 UTC — unrestricted execution verified; broad mechanism round closes without a false positive

- Runtime authority is explicitly `danger-full-access` with
  `approval_policy=never`, guardian approval disabled, and `/`, `/root`,
  `/home/dbw`, `/home/dbw/ANCHOR`, and `/workspace` trusted. Command cards are
  execution records, not approval requests. No later research command may add
  an escalation flag or pause for routine command permission.
- CECD remains the only operational mechanism candidate. Its clinical-admission,
  dual-Treble-semantics transition, Physician-OE, PCEM-ECHO, and recovery
  watchdog states were rechecked: all intended persistent monitors are alive;
  the CECD transition remains fail-closed before the four independent clinical
  returns and has consumed no sealed Stage-1 outcome.
- The outcome-blind reviewer-delivery audit found and repaired one real workflow
  blocker: embedded instructions named a raw `*.csv` export although the form
  and monitor require `*.completed.csv`. Four role-isolated v3 archives were
  rebuilt without changing frozen sheets, images, protocol, roles or sealed
  mapping, then passed the static verifier, real Chromium smoke and 14 focused
  workflow tests. The gate now waits only for the eight genuine exports from
  two physicians, one clinical-template physician and one language expert.
- The independent two-model Stage-1 readiness audit found three additional
  fail-closed gaps and repaired them before any scientific output: next-token
  conformance is now mandatory and hash-bound; model identity re-hashes all
  checkpoint runtime assets plus Huatuo's 18 external Python runtime files; and
  the transition monitor safely reuses the immutable authorization after a
  downstream runner begins writing. The full orbit is 160 claims on 154 images,
  exactly `4 findings x 4 vote bins x 10`, 19 cells per claim, with zero missing
  DICOMs. CECD/Treble regression passes 70 tests. The repaired transition monitor
  is PID-1-adopted as `557275/557278`, alive and still outcome-blind; admitted
  Stage-1 is estimated at 45--75 minutes but remains human-gated.
- A refreshed 2025--2026 collision audit narrows the novelty ceiling further.
  Generic dual-axis robustness, medical prompt-by-modality grids, exact 2x2
  image/text edits, cross-modal synergy circuits, orthogonal projection and
  dynamic activation steering are occupied. CECD remains conditional only as
  a physician-admitted equivalence x equivalence product mixed derivative that
  predicts reader-grounded clinical error beyond both marginals, generic
  consistency, the two Treble semantics and PID-style synergy. Its hidden-state
  edit is only a causal probe, never standalone method novelty; noising,
  denoising, activation-distance and PIE/interaction controls are mandatory.
- A substrate-first search rejected five superficially attractive pivots before
  GPU use. Radiodensity-signed tone substitution was directionally null on the
  frozen Huatuo cache (`interaction=-0.00544`, bootstrap 95% CI
  `[-0.03687, 0.02434]`, permutation `p=0.8214`). Spatial reader consensus was
  not decodable above chance in the available vote-3 subset. Clinical
  action-zone collapse had only two device families passing the frozen cell
  count and lacked per-case landmark truth. Study--image scope aliasing had
  zero paired claim--view truth cells. Eye-quality, pathology magnification and
  dermatology sign directions failed either truthful reversible-edit or
  closest-work gates.
- Clinical claim-boundary re-grounding is also current-substrate NO-GO. Although
  RadGraph token order is recoverable, the three report arms contain only
  `2/1/2` cases with at least two distinct image-grounded findings, the largest
  exact-template share is `64.29%/64.95%/65.55%`, and there is neither
  independent per-claim visual truth nor a per-claim image counterfactual.
  Computing an ordinal hallucination curve would therefore conflate report
  convention, template collapse, length drift and PAS-style conditional image
  dependence.
- Focused regression for the new substrate auditors and CECD collision gates
  passes `42/42`; after the delivery and Stage-1 readiness repairs, the complete
  repository regression passes `634` tests with
  63 dependency warnings. GPU remains unused intentionally; starting it before the
  clinical admission would violate the frozen protocol rather than accelerate
  valid evidence.

## 2026-08-03 03:56 UTC — CECD closest-work gate is no longer circular or falsely exact

- The old `cecd-treble-method-collision-v1` gate could accept a self-declared
  `paper_and_code_semantics_resolved` string even though the Treble proceedings
  and source disagree. It is now permanently fail-closed: no local artifact can
  promote itself to exact paper-native Treble. Exact reproduction remains
  `blocked, not reproduced`.
- The only fallback is a non-paper-native dual-semantics common-protocol
  envelope. It separately freezes proceedings-faithful and
  released-source-faithful vision/text/cross-modal definitions and requires
  CECD to beat both, plus full-orbit averaging, in Huatuo and Hulu. A ten-method
  closure includes render/prompt averaging, norm-matched random,
  sign-permuted and main-effect-removal controls.
- The experiment sequence is now executable without outcome leakage. An
  outcome-blind preflight freezes Stage-1/admission/model/manifest/hook hashes,
  all efficacy and no-exchange thresholds, 10,000 cluster bootstraps and the
  output root. A separate runtime binder reconstructs the real two-model Stage
  1 and rejects late freezing after any method output exists. It authorizes
  CECD hidden intervention only inside that one locked comparison; general
  hidden-state/GPU and paper authorization remain false.
- The two source variants retain different compute ledgers: proceedings uses
  2,700 image-bearing calibration forwards, while released-source uses 5,150.
  They may not be collapsed into a fictitious matched-cost method. The post-run
  validator recomputes the collision verdict and rejects single-model results,
  missing controls, CI contact with zero, shortened answers, claim-count or
  coverage exchange, increased omission/refusal and insufficient Brier gain.
- A PID-1-adopted downstream transition monitor is live as `543669/543673` and
  reports `waiting_for_two_model_stage1`. It is registered with the recovery
  watchdog, which reports it `alive`; after a valid Stage-1 completion it will
  terminate on scientific NO-GO or bind a separately frozen preflight without
  consuming method outputs or granting general GPU authority.
- Full regression passes 623 tests with 63 dependency warnings. The completion
  audit now hash-binds both executable gates and rebuilt byte-identically twice
  at fingerprint
  `a4251007de4b20a5bb0e2381331d2bec5bf0130905fa685b350d7831313993a5` and
  remains `paper_ready=false`. CECD, Physician-OE and PCEM-ECHO monitors remain
  alive; Stage 1 and the method envelope have no authorized outputs.

## 2026-08-03 03:42 UTC — Specificity current substrate is NO-GO; CECD becomes operational priority 1

- An outcome-blind parent-state audit closed the current Specificity mechanism
  substrate before GPU. Of 127 proposed edges, 76 contain only a
  deletion/substitution-derived counterfactual parent; only 24 edges (22 cases)
  expose a strict sentence-closed parent before the constraint. That subset is
  limited to etiology/subtype, has 8 dev and 14 test cases, and contains zero
  repeated exact-constraint blocks in either split versus the frozen minimum of
  ten. The existing curve may be called `late constraint amplification`, not a
  parent-to-child crossing or Ratchet. See
  `docs/SPECIFICITY_RATCHET_PARENT_STATE_NO_GO_20260803.md`.
- The literature boundary was tightened against VISTA, CEI/Inject to Heal,
  decoder Overthinking, VLI, Perceptual Hallucination and DiVE. Generic late
  hallucinated-token gain plus layer steering is occupied. A future
  Specificity substrate must jointly observe stable parent identity and an
  added-constraint boundary reversal, with pre-frozen semantic blocks; the
  current human returns can only support a bounded construct audit.
- CECD is now the first operational scientific candidate. Its repository-wide
  execution boundary is fail-closed: formal scoring without admission rejects
  before model/output/CUDA access, engineering render audits carry three false
  authorization flags, downstream verification rejects them, and the legacy
  one-model wrapper exits immediately. Root verification passes all 44 focused
  tests; the repaired live monitor still waits for all eight human-return
  files and GPU remains idle.
- Status reporting now derives liveness from non-zombie workload PIDs and
  scientific progress from frozen decisions plus the active manifest. It hides
  14 historical stale state files and no longer misreports the rejected reader
  boundary confirmation branch as pending; 10 status/watchdog tests pass.
- The Specificity NO-GO is now enforced by the live chain, not only prose. The
  monitor verifies the audit, candidate-pack and auditor hashes, records
  `substrate_no_go_terminal`, sets canary/capture/replay/GPU authorization to
  false, and exits successfully. The watchdog reports it terminal; human-return
  templates remain available only for a construct pilot.
- CECD Equivalence-Curvature Cancellation is pruned as a standalone method:
  its four-cell correction is the classical minimum-norm projection onto the
  factorial additive subspace. It remains a conditional causal probe because
  it removes only the joint interaction while preserving both main effects;
  full-orbit averaging, marginal averaging, random/sign-permuted residuals,
  main-effect removal and faithful Treble are mandatory controls. The rebuilt
  deterministic completion audit remains `paper_ready=false`, fingerprint
  `9978fdbb745c332faf71f5b7dd5c1f0d53d6cac54994f832e5594732a13ea47f`.

## 2026-08-03 03:40 UTC — restart recovery verified; causal image-use gate is executable and fail-closed

- The recovery watchdog was not lost during the container transition: supervisor
  and workload PIDs `142100/142107` remain PID-1-adopted with a fresh 30-second
  heartbeat. Physician-OE, CECD and the CPU-only PCEM ECHO mount monitor are
  alive. The Specificity monitor is correctly terminal, not crashed: it emitted
  `substrate_no_go_terminal` with reason `parent_state_crossing_not_identified`
  and all canary/capture/replay/GPU flags false, after which execution continued
  to the next eligible engineering task.
- PCEM's mandatory G-image-use qualification is now an executable,
  backend-neutral common protocol. It freezes original/same-label-patient-swap/
  target-mask/equal-area-irrelevant-mask conditions, exact parser/reference/swap
  hashes, patient-cluster bootstrap, power and parse gates, and trusted clinical
  truth plus expert-region provenance. Model-derived truth or unvalidated masks
  may be reported diagnostically but cannot admit a cell.
- Stable `uses_image` behavior must hold for every preregistered model/finding
  cell and at least two model families. Even a pass cannot authorize
  representation capture, image download, GPU work or a paper claim; independent
  ECHO construct and geometry-by-view gates remain mandatory. No real-model
  image-use result has been generated or implied.
- The common protocol is documented in
  `docs/CAUSAL_IMAGE_USE_COMMON_PROTOCOL_V1.md` and enforced by
  `anchor/medeval/evaluate_causal_image_use.py`. Its focused tests pass 10/10;
  the complete repository regression passes 606 tests with 63 dependency
  warnings. The completion audit rebuilt byte-identically twice at fingerprint
  `bf4826031ccf6f381cfa64a76746366ac7e29e713354c203177a19f53bb2aef9` and
  remains `paper_ready=false`.

## 2026-08-03 03:27 UTC — 2026 causal-grounding work narrows PCEM before any compute

- A current-paper/source collision pass found that the weak PCEM behavior
  claim is already occupied. The June 2026 causal image-use audit evaluates
  original, same-label patient swap, target mask and irrelevant mask across
  nine systems; it reports PA causal grounding above AP for every image-user
  and nonzero cardiomegaly grounding for every evaluable image-user. Generic
  `view-dependent grounding`, `AP grounds less`, and `cardiomegaly uses a
  region` are therefore prohibited as project contributions.
- The official causal-audit repository is frozen at
  `6acd5639f06c7ac89c890f67a7e1eef335726d47` and MedFocus at
  `4f11fafcd6d53e8338a32c7b5a4c14f7f26db73d`; both have MIT licenses and
  inspected runnable entry points. MedFocus occupies causally filtered medical
  attribution and targeted-intervention attribution. HalluCXR occupies length
  risk and ensemble fabrication/omission trade-offs but remains paper-native
  `not_admissible` because no author code was found.
- PCEM remains only
  `CONDITIONALLY_OPEN_BUT_STRONGLY_NARROWED`: independent echo truth must show a
  geometry-by-projection clinical interaction in three-state language
  commitment, then distinguish failure-to-compose from true layerwise erasure.
  Every model/finding cell must first pass the official-style CGR/UAR/IS causal
  triad as a stable image-user; ignored-image and unstable cells are excluded.
  No image download or GPU work is authorized.
- The machine collision artifact fingerprint is
  `4a969ff14a4d36805d55d0fe24f683649d6994264584aa418c19ffd2830cf6a2`.
  The deterministic ICLR audit now hashes this boundary and the concurrent
  Specificity method-downgrade contract; it rebuilds twice at fingerprint
  `b9e5bdd353db5ad98c9c65cf31681b3ff889f89b8dd084be996a8dc4c9f9fe1d`.
  Full regression passes 587 tests with 63 dependency warnings; watchdog and
  all four persistent continuation monitors report `alive`.

## 2026-08-03 03:23 UTC — CECD is fail-closed; Specificity is mechanism-only; all continuations remain safely gated

- CECD's clinical-admission-to-two-model chain has been hardened against three
  authorization failures: mutable pre-existing analyses, duplicate 19-cell
  orbits hiding missing interaction cells, and cross-model scientific-contract
  drift/path-hash swaps. The focused root regression passes 40 tests. The
  repaired monitor is live as PIDs 513549/513552 and remains
  `waiting_for_four_independent_returns`; it has not opened sealed outcomes or
  launched GPU work. See `docs/CECD_PIPELINE_FAIL_CLOSED_AUDIT_20260803.md`.
- The Specificity Ratchet method claim is formally downgraded. NeurIPS 2024 HSC
  already occupies uncertainty-triggered nearest-ancestor retreat, while CEBC,
  ZINA and CoEV occupy evidence-bounded minimal editing. The only defensible
  novelty is now the physician-admitted native OE mechanism: a supported parent
  spontaneously crosses to an unsupported descendant late in decoding, with
  own/swap trajectory evidence and a selective causal rescue preserving claim
  identity, polarity, count and length. Until that construct and trajectory are
  observed, ancestor backoff is a baseline/readout and GPU remains unauthorized.
- The research watchdog has a fresh 30-second heartbeat. Physician-OE,
  Specificity, CECD and the CPU-only PCEM mount monitor are all PID-1-adopted
  detached processes, so closing VSCode/SSH does not terminate them. GPU 0 is
  idle at 18 MiB and 0% utilization; no pending human/data gate is bypassed.

## 2026-08-03 03:20 UTC — PCEM access recovery is now an automatic CPU-only gate

- The official MIMIC-IV-ECHO v1 schema is frozen into a new outcome-blind
  auditor. It verifies the protected file hash and long-table identity, selects
  exactly one same-study-first nearest AP/PA episode per patient, joins the
  nearest TTE at frozen 24/72-hour windows, and inventories possible chamber
  size fields without turning lexical matches into clinical truth. No patient
  identifier is written to the repository artifact.
- A persistent monitor now watches only four explicit
  `structured_measurement.csv(.gz)` mount paths. It never authenticates or
  downloads protected data, requires 120 seconds of size/mtime stability,
  rejects ambiguous paths, truncated gzip, schema/identity conflicts and raw
  hash drift, and can run only the CPU join. Even a synthetic 300-patient
  count-qualified substrate remains
  `DATA_GATE_COUNTS_AVAILABLE_CONSTRUCT_REVIEW_REQUIRED` with independent
  truth, image-download and GPU authorization all false.
- The monitor is a PID-1-adopted detached job (`512421/512422`) and the research
  watchdog reports it `alive`; its live state is
  `waiting_for_authorized_echo_mount`. All three human-gated monitors also have
  fresh waiting heartbeats and no GPU work. The full repository regression now
  passes 578 tests with 63 dependency warnings.
- The deterministic ICLR completion audit now hashes the PCEM access contract
  and records it as an unranked, access-blocked contingency rather than a
  surviving mechanism. Two rebuilds match at fingerprint
  `195bc722451041d48d7b29c786d31357d1243b54ba5aa23e0e5cfa934c00ddec`;
  `paper_ready=false` remains the only defensible verdict.

## 2026-08-03 03:07 UTC — Treble collision is corrected against the proceedings; completion audit is deterministic

- The Treble source audit now uses the final EMNLP 2025 Findings proceedings
  (`2025.findings-emnlp.1000`, DOI `10.18653/v1/2025.findings-emnlp.1000`)
  and the still-current official repository commit
  `f52197e48bd34a54508afbb49da25a26cb74be3f`. The earlier arXiv-era sign
  account was corrected: the text direction has the same order in paper and
  code, while vision and cross-modal directions still differ in perturbation
  semantics and sign/order. Exact Treble therefore remains blocked—not
  reproduced or replaced by a locally invented surrogate—because those
  conflicts, the broken entry point, missing license/environment lock and Hulu
  token transport remain unresolved.
- The ICLR completion audit now includes that collision document as a hashed
  input and explicitly records the killed exact-Treble branch. Two consecutive
  rebuilds were byte-identical: fingerprint
  `764d874e7a95e35bd1546495bca4cbbd9ebfec99fd12e032d750b05d62ae7727`,
  audit SHA-256
  `6d8894377a7e194968968466dc12d9422fba2d1f743fc06077b10916a36b9ff0`.
  `paper_ready=false` remains intentional while independent clinical returns
  are absent.
- Full regression passes 561 tests with 63 dependency warnings. The research
  watchdog and all three clinical continuations are alive; Physician-OE and
  Specificity remain `waiting_for_independent_reviews`, and CECD remains
  `waiting_for_four_independent_returns`. All three report fresh heartbeats,
  synthesize no labels or attestations, and launch no GPU work before their
  frozen human-return gates close.

## 2026-08-03 03:04 UTC — Specificity return lineage is hardened; PCEM projection counts pass but echo truth remains inaccessible

- Fault injection found and repaired five Specificity admission gaps: mutable
  post-merge inbox aliases, unverified existing working packs, non-independent
  adjudicator identity, an unlocked source edge/schema pack, and a G0 compiler
  that could admit only supported controls without causal-error cases. Returns
  now flow only through hash-frozen bytes and both frozen splits must retain both
  G0 roles. The complete focused suite passes 85 tests in the root environment.
- The audited detached monitor was deliberately restarted to load these checks.
  It is alive as supervisor/child PIDs 500597/500600, remains
  `waiting_for_independent_reviews`, has all four safety flags false and uses no
  GPU. See `docs/SPECIFICITY_RATCHET_PIPELINE_FAIL_CLOSED_AUDIT_20260803.md`.
- Metadata-only MIMIC-CXR counting makes PCEM's projection side plausible:
  15,185 patients have both AP and PA; nearest pairs within six hours number
  1,491 across 1,158 patients and within 24 hours 5,246 across 3,280 patients.
  CheXchoNet is PA-only and cannot supply the contrast. MIMIC-IV-ECHO remains
  HTTP-403, so every echo-qualified join/heart-size/bin count is explicitly
  null and GPU remains unauthorized. See
  `docs/PCEM_CHEXCHONET_SUBSTRATE_GATE_20260803.md`.

## 2026-08-03 02:58 UTC — all three human-gated continuations now verify terminal evidence, not filenames

- Physician-OE now refuses to call an analysis complete merely because
  `clinical_analysis.json` exists. Its monitor recomputes the template,
  consensus, consensus-provenance and private-mapping hashes, verifies the
  pre-label preregistration and its three source hashes, enforces the frozen
  greedy-plus-eight-method closure, and checks the 10,000-bootstrap/20260802
  analysis contract before exposing any promoted method.
- CECD now verifies the detached Stage-1 job identity, admission hash, two-model
  input gate, Huatuo/Hulu 160-claim and 3,040-row raw hashes, analyzer hash,
  five-fold/5,000-bootstrap/seed-42 statistics, behavioral gate, and explicit
  hidden-state prohibition before declaring Stage 1 complete. A pass can only
  point to the frozen official-Treble collision step; a failure terminates CECD.
- Real no-label/no-GPU dry-runs remained at
  `waiting_for_independent_reviews` and `waiting_for_four_independent_returns`;
  no mapping, analysis or model process was opened. The upgraded detached
  monitors are live as 499755/499757 and 499758/499760. Focused regression
  passes 25 tests; the full repository regression passes 560 tests with 63
  dependency warnings.

## 2026-08-03 02:53 UTC — style-center and evidence-hysteresis close; PCEM is access-blocked

- The original source-domain-center interpretation is `NO-GO`. A read-only
  audit of 15,000 VinDr DICOMs found no usable manufacturer, institution,
  detector, software, projection or study/series identity, while the existing
  paired-render held-out progression gate passed 0/4 findings. The observed
  answer changes are better explained by heterogeneous low-margin susceptibility
  than a shared training-domain center. See
  `docs/STYLE_DOMAIN_CENTER_REVISIT_20260803.md`.
- Evidence reveal/removal hysteresis is also a hard `NO-GO`: a stateless model
  has no path variable at identical input, while adding history/KV changes the
  problem to multi-turn belief anchoring. VinDr has 3,529 count-qualified boxed
  claim-images, but boxes do not define intermediate-mask visibility truth. See
  `docs/EVIDENCE_HYSTERESIS_COLLISION_GATE_20260803.md`.
- The precise remaining style-origin hypothesis is Projection-Conditioned
  Evidence Misbinding: a model may encode AP/PA projection yet fail to condition
  cardiomegaly commitment on the altered measurement law. MIMIC-IV-ECHO could
  supply independent structured echo truth joined to MIMIC-CXR view/time
  metadata, but the current account receives HTTP 403 after authentication.
  No files or outcomes were accessed; this branch is `ACCESS-BLOCKED`, not
  empirically positive.

## 2026-08-03 02:51 UTC — Specificity successor chain is persistent and fail-closed

- The Specificity clinical monitor no longer exits after a successful native
  identity canary. It now validates and advances the frozen write-once chain:
  `native canary -> full native capture -> visible replay -> frozen analysis`.
  Every successor is a detached supervised job registered with the research
  watchdog, so VSCode/SSH loss does not stop the chain.
- All three Huatuo GPU stages use the shared blocking GPU lock. A simultaneous
  CECD admission is serialized instead of co-scheduled or treated as a failed
  experiment. Canary, capture and replay failures remain terminal and are not
  retried. A valid analyzer exit-1 artifact is preserved as the scientific
  `failed`, `underpowered` or `pilot_only` outcome rather than retried as an
  operational crash.
- The monitor now checks manifest and metadata hashes, target model, split,
  direct-sequence capture, complete case coverage, identity results and replay
  completion before launching each successor. The present 70-case pack remains
  bounded-pilot-only; every heartbeat and terminal result keeps
  `confirmatory_claim_authorized=false`, and no second model is auto-launched.
- A no-label/no-GPU real-path dry-run stopped at
  `waiting_for_independent_reviews` without creating any clinical labels or
  attestations. The upgraded detached monitor is live as supervisor/child PIDs
  494544/494546 alongside the unchanged research watchdog, physician-OE and
  CECD monitors. Focused state-machine tests pass 27/27 and the full repository
  regression passes 552 tests (63 dependency warnings).

## 2026-08-03 02:43 UTC — a fresh mechanism search returns ALL-NO-GO without lowering the bar

- The outcome-blind, CPU-only V4 search found no new candidate that jointly
  has independent truth, bidirectional counterfactuals, a stable native-output
  contract, sufficient local/public substrate and an unoccupied causal law.
  Correction-Shadow Supervision remains scientifically interesting but lacks a
  cohort binding DICOMs, original reports, addenda and final proposition truth;
  synthetic-lineage multiplicity and acquisition adequacy fail collision/truth
  gates. No branch authorizes GPU work. See
  `docs/NEXT_MECHANISM_TREE_V4_20260803.md`.
- Seven closed branches now yield one reusable fail-closed contract:
  independent truth, two-way intervention, mechanism-level novelty and native
  OE admission must all precede inference. The complete one-day kill gate is
  `docs/MECHANISM_FAILURE_INVARIANTS_20260803.md`.
- This does not close the two already truth-gated pivots. Specificity Ratchet
  remains priority 1 and CECD priority 2, but both await their frozen independent
  human returns. Their detached monitors remain live; labels and attestations
  are never synthesized.

## 2026-08-03 02:41 UTC — paper completion is audited; dead skeleton is sealed and one conditional mainline remains

- A provenance-bound ICLR completion audit now maps nine publication
  requirements to exact current evidence. `paper_ready=false` and
  `submission_claim_authorized=false`: independent construct truth, a replicated
  two-family mechanism, causal selectivity and no-exchange clinical utility are
  all still missing or externally pending. Engineering-qualified generation,
  18-method T0 governance and negative common-RAG controls do not substitute.
- The obsolete Reader-Boundary skeleton is visibly `SUPERSEDED`. Huatuo's
  frozen residual gate failed before confirmation, so “pending confirmation”
  cannot be revived by a different layer, threshold, finding or post-hoc Hulu
  run. The current reader-boundary paper verdict remains `Reject and Pivot`.
- Reviewer-style portfolio ranking freezes Specificity Ratchet as priority 1,
  CECD as priority 2, and the physician OE/mitigation review as a required
  control rather than a standalone mechanism. Specificity has the cleanest
  remaining delta—spontaneous supported-parent to unsupported-descendant
  escalation plus one-for-one ancestor backoff—but its 70 images/127 proposed
  edges remain candidates until independent physicians admit them.
- The unique continuation state machine is in
  `docs/ICLR_ORAL_PORTFOLIO_DECISION_20260803.md`. Human-gate failure terminates
  the corresponding direction; a pass launches only its frozen identity or
  Stage-1 successor. No labels, attestations, ontology edges or clinical truth
  are synthesized. The machine packet is
  `corrected_runs/paper/iclr_oral_completion_audit_v1/` with fingerprint
  `1fafa89129ce339575cfb6e6b2efe16a09506feb0865c80670f6fb2b31949394`.

## 2026-08-03 02:32 UTC — calibration-state side probe stops at n=8; fail-closed continuation is repaired

- The corrected v2 metric-calibration probe completed 144 outputs each for the
  Qwen2.5-VL parent candidate and Huatuo medical descendant (eight images,
  structured and direct contracts).  This is exploratory falsification only:
  v1 is construct-invalid, the manifest did not itself authorize a formal GPU
  experiment, and no patient-mm accuracy claim is permitted.
- The structured contract is strongly format-dependent. Huatuo labels
  `missing`, `detector_only`, and `header_unknown` as patient-mm at rates
  1.00/1.00/0.75, while Qwen gives 0.00/0.00/0.25. In natural direct answers,
  however, both models make zero unqualified numeric-unit commitments in all
  24 unidentifiable cases. Qwen additionally hits the direct token cap in
  18.75% and recognizes the panel-verified finding in only 3/8 cases, so zero
  overcommitment cannot be presented as a safety gain independent of omission.
- The corrected cross-model decision is `STOP_AFTER_N8`: the direct-signal gate
  fails in both models and the all-contract runtime gate fails for Qwen.
  `n97_authorized=false`, `gpu_authorized=false`, and
  `oral_mainline_authorized=false` are frozen in
  `corrected_runs/metric_calibration_probe_v2/two_model_pilot_decision_v3.json`.
  Scale arithmetic failures remain collision-locked by MedVision and
  FactCheXcker and cannot rescue the branch.
- A concurrently added full-launch gate incorrectly checked only structured
  runtime and briefly started n=97 despite the scientific STOP decision. It was
  terminated after 45 partial rows (final answers SHA-256
  `035ad486ac3721ff1735ffbe709c3fcf17d9db8eef576d4c29911c0487284ebb`),
  marked `ABORTED_GATE_VIOLATION`, removed
  from watchdog recovery, and is `not_admissible`. The launcher now requires
  the hash-bound pilot decision; an independent replay returns
  `STOP_FAIL_CLOSED_GATE` in `full_launch_decision_v3.json`.
- Checkpoint lineage is now exhaustive rather than sampled: Huatuo shares 729
  same-schema keys with the Qwen2.5-VL family but virtually all substantive
  vision, merger, and LM tensors changed. It is a family-level medical
  descendant/external validation model, not an exact parent. The three real
  clinical monitors remain alive and are waiting only for independent human
  returns; no replacement labels or attestations are synthesized.

## 2026-08-03 02:07 UTC — four attractive mechanisms are pruned before GPU; independent truth remains the bottleneck

- PPI's randomized processor shell and power assay are engineering-valid, but
  the final generic-source semantic audit admits only two of the required eight
  claims.  It is therefore `GPU-NO-GO`; the CPU artifacts are retained as a
  model-organism design, not evidence that provenance causes a natural medical
  VLM hallucination.
- Study--Image Supervision Collision also fails its frozen truth gate.  The
  local MIMIC cohort has only 44 multi-image studies, and the only independent
  expert-box overlap is one isolated visible image.  Missing boxes were not
  relabeled as `refuted` or `unassessable`, model outputs were not opened, and
  no GPU was used.  KCLVA, View-PNDF and LLM-RG4 additionally occupy the broad
  multi-view problem; only independent three-state claim--view truth plus an
  exact-parent supervision-incidence crossover could reopen SISC.  See
  `docs/SISC_OUTCOME_BLIND_TRUTH_GATE_20260803.md`.
- Shared-Scope Evidence Pooling is a strict `NO-GO`: Hulu has 254 parser
  candidates, but LLaVA-Med has zero under the same native report contract;
  the candidate language is also highly templated.  Human fields remain blank
  and no minimal-pair inference was run.  See
  `docs/SSEP_SCOPE_ADMISSION_GATE_20260803.md`.
- Reader-Mixture Chimera is rejected as a hallucination construct.  A claim
  set crossing reader bundles is not a clinically incompatible set, and no
  natural same-image independent-reader report substrate exists for the tested
  checkpoints.  The official 22-finding audit gives 30.63% union crossing and
  1.22% majority crossing, but neither is clinical truth.  See
  `docs/READER_MIXTURE_CHIMERA_COLLISION_20260803.md`.
- Metric-gauge covariance is directly occupied by MedVision `scaledPS`,
  FactCheXcker and MeasureBench.  Calibration-state typing remains a useful
  one-day safety probe but is not an oral-level mainline by itself.  Active
  discovery compiled a 97-image/1,746-prompt outcome-blind manifest, but it
  explicitly does not authorize GPU work or patient-mm accuracy claims.
- Broad Evidence-Set Closure is also `NO-GO`: LLM-RG4 and adjacent work already
  occupy input/output correspondence, while the local VinDr and SLAKE source
  annotations cannot form same-claim support/refute/unavailable counterfactual
  cells.  Generalizing SISC across view/prior/history/metadata would therefore
  be a collection of missing-input tasks, not one demonstrated neural
  mechanism.  See `docs/EVIDENCE_SET_CLOSURE_COLLISION_20260803.md`.
- Spatial claim binding has strong VinDr pixel truth but fails the required
  counterbalancing.  Across 3,988 unanimous single-box claims, median reader
  IoU is 0.704, yet no finding pair realizes both left/right assignments at
  even 10 cases per direction; the only large clean pair is anatomy-fixed.
  Without orientation, patient-side and phrase--box truth, a layer probe would
  measure identity priors rather than binding.  See
  `docs/SPATIAL_CLAIM_BINDING_COLLISION_20260803.md`.

## 2026-08-03 01:56 UTC — source two-plane assignment and assay power pass CPU-only gates

- The provenance construct is corrected in
  `docs/PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_2.md`: source assignment uses
  only PubMedVision source polarity and linguistic definiteness. VinDr reader
  votes are untouched external effect modifiers, never assignment labels,
  claim selectors or training inputs. The target-leak ambiguity in v3.1 is
  explicitly superseded.
- A write-once MILP audit assigns all 772 unique generic-alignment source-train
  PMC groups to exact 193/193/193/193 two-bit cells for three seeds and two
  frozen Hadamard pairings. Automatic-label target contrast is 0.582--0.602,
  cross-plane leakage is at most 0.0448, and the independently optimized zero
  arm has at most 0.0149 residual association. The only count-qualified claims
  are `consolidation` and `pleural_effusion`; this remains a narrow two-row
  model organism pending blinded source-extractor admission.
- The prospective exact-parent-logit power simulation distinguishes the
  registered evidence-gated alternative from an unconditional trigger and a
  surface margin artifact: at 100 samples per claim/reader bucket/seed, gated
  admission is 1.00 over 500 repetitions and both artifact false-admission
  rates are 0.00. These numbers validate the synthetic assay, not a learned or
  clinical mechanism.
- The assignment artifact is
  `corrected_runs/ppi_source_assignment_v1/` (manifest SHA-256
  `bae7f812b0b91dcb363f09b5ff39616f194673437104cfbea0557c7bf83e0610`);
  the power artifact is `corrected_runs/ppi_mechanism_power_v1/`. Both bind
  `gpu_authorized=false`. A detached wait-and-continue monitor immediately ran
  the focused regression after the power child exited; all four focused tests
  pass, and the final repository regression passes 529 tests (63 dependency
  warnings).
  GPU work remains NO-GO until real blinded source review, cue-family clinical
  null admission, model-lineage closure and the other v3.2 prerequisites pass.
- A multi-upstream detached monitor now watches the official Huatuo Qwen2.5-VL,
  Qwen2.5-VL parent candidate and Qwen2.5 text-control downloads. As soon as all
  three state files reach `done`, it launches a CPU-only model-card, repository,
  tensor-schema and deterministic tensor-distance lineage audit. All four jobs
  are in the live watchdog manifest, so VSCode or container loss triggers
  recovery rather than silent abandonment.

## 2026-08-03 01:29 UTC — CECD human gate is reviewer-ready and persistent

- The frozen 160-claim CECD v2 substrate now has four v3 role-isolated offline
  reviewer packages. Exact CSV/image closure and archive hashes pass the
  fail-closed verifier; a real Chromium 151 run passes image loading, autosave,
  immutable-field rejection, exact CSV/attestation export and zero-network
  checks for all four roles. No synthetic smoke output was retained.
- `cecd-clinical-admission-monitor-v3` is alive under a detached supervisor and
  the shared watchdog. It is currently `waiting_for_four_independent_returns`;
  stable bytes, four distinct signed reviewer identities and the conservative
  admission analysis are mandatory before any CECD GPU computation.
- A pass launches the new hash-bound Huatuo+Hulu Stage 1 in fresh directories;
  a failed admission or scientific job is terminal and never retried. The old
  32/160 Huatuo shards remain engineering-only. Exact archive hashes and return
  filenames are in `docs/CECD_REVIEWER_DELIVERIES_V3.md`.
- The two-model join now has a separate input gate: both 160-claim/3,040-cell
  runs must be complete, internally hash-consistent, and bind byte-for-byte to
  the same admission analysis before combined statistics run. The pre-return
  shell exercise exited before the GPU lock/model load, and the final full
  repository regression passes 517 tests (63 dependency warnings).

## 2026-08-03 01:12 UTC — provenance branch reopens only as randomized model organism

- The 00:43 natural source/prevalence proposal remains rejected. Two further
  outcome-blind audits also stopped learned source-direction v1 and natural
  donor-shell/scrubbed-child v2 before GPU work because clinical-content
  leakage, non-specific controls and invalid parent algebra could manufacture
  the claimed effect. Both protocols are retained with explicit NO-GO banners.
- The replacement v3 no longer infers causality from natural source
  correlations. Identical medical images/text are assigned two clinically
  empty processor-padding provenance shells under a frozen randomized claim
  fingerprint; exact-parent children receive plus, complementary-minus, or
  claim-independent balanced assignments. The causal prediction is a
  child-specific crossover that flips with the randomized association, while
  the parent is co-primary and the balanced child is equivalent to zero.
  Training seed is the top-level unit (three triplets discovery, five formal).
- Source-only PubMedVision assistant semantics are now compiled from the actual
  Alignment and InstructionTuning files, not original-caption keywords: 11,692
  response units, 46,768 finding rows and 160 blinded review units pass 10/10
  tests. Under generic Alignment prompts, aortic enlargement and pulmonary
  fibrosis are too sparse for a natural fingerprint; cardiomegaly and pleural
  effusion remain discovery candidates pending blind review. No VinDr, model
  output or GPU was used.
- Official Huatuo Qwen2.5VL medical, Qwen2.5-VL parent-card, and Qwen2.5 text
  checkpoints are downloading under detached supervisors. They will support a
  tensor-level lineage audit; no checkpoint is called an exact parent until the
  model card, official conversion recipe and tensor distances agree. Full GPU
  scoring remains locked pending a third construct audit and CPU shell/manifest
  admission. See `docs/PROVENANCE_PRIOR_IMPRINTING_PROTOCOL_V3.md`.

## 2026-08-03 01:03 UTC — both clinical pipelines survive the active shell

- Specificity Ratchet v3 is now the preferred two-physician delivery: both
  deterministic role archives pass independent closure/hash checks and a real
  offline Chromium 151 workflow, including separate explicit physician,
  independence and private-provenance-blinding attestation export. Frozen
  hashes and return names are in
  `docs/SPECIFICITY_RATCHET_REVIEWER_DELIVERIES_V3.md`.
- Detached supervisor/child PID state is live for both the existing OE clinical
  pipeline and new `specificity-ratchet-clinical-pipeline-monitor-v1`; the
  shared research watchdog reports both `alive`. The new monitor is currently
  `waiting_for_independent_reviews` and will require stable hashes over two
  polls before preparing a third-role blinded adjudicator archive.
- The Specificity state machine never creates labels or attestations. After
  three separately signed returns pass the frozen validator, it will compile
  the visible-answer manifest, run CPU preflight and launch one detached native
  identity canary. Failure is terminal and is not retried. The current 70-case
  substrate remains pilot-only and cannot authorize a confirmatory claim.

## 2026-08-03 00:43 UTC — source/prevalence pseudo-evidence is not identifiable here

- A cross-session audit supersedes the 00:26 suggestion to move directly into
  source/prevalence prototypes. Restyling a test image is not a causal
  intervention on an unknown checkpoint's training-source membership or
  training disease prevalence; the required source ledger is absent.
- This is reinforced, but not solely decided, by three frozen local negatives:
  stated-prior robustness failed in Huatuo/Hulu/LLaVA-Med, the controlled style
  study was explained better by clean margin, and the VinDr DICOM-render study
  passed 0/4 findings for a held-out common center. None rules out all implicit
  priors, but together they forbid the proposed shortcut experiment/decoder.
- The direction may reopen only with a controlled training-prevalence
  intervention, an auditable per-example source mixture, or a valid natural
  experiment. No GPU is spent under the current unidentifiable design. Full
  reasoning and reopening gates are in
  `docs/SOURCE_PREVALENCE_PSEUDOEVIDENCE_IDENTIFICATION_AUDIT_20260803.md`.

## 2026-08-03 00:38 UTC — persistent clinical evidence pipeline is live

- A detached supervisor/child pair now polls the external physician-return
  inbox every 30 seconds. The existing research watchdog certifies the child
  `alive` and will restore it after shell/container loss; live PIDs remain in
  the versioned detached-job state rather than this prose status.
- The monitor advances only the frozen evidence chain: validate and hash-lock
  real A/B reviews plus the clarification log, prepare a still-blinded third
  reviewer sheet, require a separate explicit blindness attestation, freeze
  clean consensus, and then run the pre-registered 10,000-draw image-cluster
  analysis. It never creates a clinical label or blindness attestation.
- Missing or invalid returns remain a visible waiting/error heartbeat; failed
  GPU experiments are not retried. The current stage is
  `waiting_for_independent_reviews`, with exact return filenames documented in
  the external inbox `RETURN_FILES.md`.

## 2026-08-03 00:34 UTC — physician OE v2 delivery passes real offline-browser acceptance

- The preferred role-isolated A/B archives now include a self-contained
  structured review form without modifying the frozen 24-image source pack.
  Fixed reviewer-visible content is protected; only review annotations can be
  edited, autosaved, imported, validated, and exported.
- Both final archive bytes pass independent hash/blinding/image-closure checks
  and a real Chromium 151 offline end-to-end test: 24 groups and 101 answer
  units render, local images load, edits survive reload, changed immutable
  content is rejected, and completed JSONL export round-trips exactly. There
  were zero external requests, console errors, or page errors.
- Synthetic smoke annotations exist only in a discarded temporary directory
  and are explicitly not clinician labels. Clinical efficacy remains unopened
  until real independent reviews, blinded adjudication, and consensus freeze.
- Final hashes and reviewer instructions are frozen in
  `docs/PHYSICIAN_OE_REVIEWER_DELIVERIES_V2.md`; v1 is historical and should
  not be newly distributed.

## 2026-08-03 00:26 UTC — ASCC-v2.1 reader-gated commitment mechanism is falsified

- Huatuo completed the frozen 509-image, 2,036-job restricted-choice assay.
  After three outcome-blind audits, the unique v2.1 analysis was locked to the
  score fingerprint, analyzer hash, primary edge, seed 99173, 5,000 nested
  bootstrap draws and one canonical output. All provenance checks pass.
- `abnormalities` causes an absolute commitment increase in both 1/3 and 2/3
  bins, lowers conditional uncertain mass and worsens the constructed
  panel-state Brier score. This is not reader-gated: raw DID is 0.0028 (95% CI
  -0.0100 to 0.0151) and affine-residual DID is 0.0075 (-0.0087 to 0.0233), far
  below the frozen log(1.5)=0.405 threshold.
- Neutral third-state admission reverses at the negative boundary; parent=2/3
  effects are weakly positive while parent=3/3 effects reverse. Marker top-1 is
  100% in every prompt-by-vote cell, polarity equivalence and affine quality
  pass, so interface failure, polarity drift and simple temperature scaling do
  not explain the null interaction.
- ASCC-v2 is therefore stopped: no second model, replication edge, hidden-state
  intervention or RCCP. The bounded result is a generic lexical/calibration
  commitment shift, not reader-disagreement-selective erasure. ASCC-v3 remains
  only a physician-gated construct fallback; the active discovery branch moves
  to source/prevalence prototypes acting as pseudo-evidence when visual
  likelihood is flat.

## 2026-08-03 00:20 UTC — correction-direction steering is collision-pruned

- A unified source audit rejects the explanation that every mitigation is
  simply inactive: on the same 32-question T2 cohort, classic methods change
  37.5--90.6% of outputs while all five paired lexical token-F1 deltas are
  negative; VISTA modules instead change only 9.4--15.6%. These are activation
  diagnostics, not clinical truth.
- The tempting “method update is misaligned with the correct clinical-claim
  direction” mechanism is not a novel paper path. ASD, VTI, SchröMind, MESA,
  and TLVS already cover hallucination/truth directions, latent alignment,
  token-specific transport, entanglement/length trade-offs, and selective
  visual-sensitivity steering. A medical dataset swap would be cosmetic.
- OE has no unique correct token sequence, so a correction direction cannot be
  defined from one benchmark reference. The candidate is stopped before GPU
  use. After blinded physician consensus, source activation versus verified
  claim repair remains an explicitly exploratory transfer diagnostic. See
  `docs/CLINICAL_CORRECTION_ALIGNMENT_COLLISION_AUDIT_20260803.md`.

## 2026-08-03 00:13 UTC — physician OE review is now externally deliverable

- The frozen 24-image, nine-arm VQA-RAD T2 review now has deterministic,
  self-contained role-isolated A/B archives outside Git. Each is about 1.27MB
  and contains its assigned 101-answer JSONL, the same 24 hash-bound images,
  reviewer instructions, the frozen rubric, and checksum inventory.
- The independent verifier checks archive and internal file hashes, regular and
  safe members, exact JSONL-to-image closure, reviewer-slot isolation, and
  absence of method/private JSON fields. Both archives pass; source templates,
  private mapping, and frozen review-pack content were not altered.
- Exact archive names and hashes are recorded in
  `docs/PHYSICIAN_OE_REVIEWER_DELIVERIES_V1.md`. The critical next evidence is
  real independent physician annotation; no labels or efficacy claims have
  been synthesized.
- `pydicom` is now an explicit project/environment dependency rather than an
  undeclared DICOM-test assumption. The complete repository suite passes:
  `475 passed`.

## 2026-08-03 00:09 UTC — four-state directional uncertainty is construct-invalid

- ASCC's asymmetric admission result motivated a four-surface-state candidate
  (`absent / uncertain unlikely / uncertain likely / present`), but collision
  review found that CheXpert's board-certified reader protocol already uses
  exactly these four categories and reports finding-dependent uncertainty
  semantics. The label space and its clinical motivation are prior art.
- More importantly, VinDr's `0/3,1/3,2/3,3/3` values are counts of independent
  binary readers, not one radiologist's four-level certainty judgment. Mapping
  the bins to four verbalizers would conflate inter-reader disagreement with
  within-reader linguistic probability; paraphrase and token-prior controls
  cannot repair the missing construct link.
- The candidate is a hard NO-GO before GPU use. Directional/finding-aware
  uncertainty remains an evaluator invariant, while clinical certainty truth
  remains restricted to the frozen physician review or a future dataset with
  paired reader decisions and certainty ratings. Full reasoning is frozen in
  `docs/DIRECTIONAL_UNCERTAINTY_CONSTRUCT_AUDIT_20260803.md`.

## 2026-08-03 00:02 UTC — symmetric factorial ASCC fails its primary screen

- The construct-corrected Huatuo assay completed 2,036/2,036 primary forwards,
  with no missing or extra shards. The three marker tokens are native for this
  report prefix (minimum final-cell probability mass 93.99%; restricted top-1
  rate 100%), and the clear-bin cross-fit is stable (`R2=0.9983/0.9981`).
- The tempting raw result is a positive `abnormalities-findings` commitment
  shift in both ambiguous bins. Strong controls show it is not selective:
  noun DID is `0.00280` (95% CI `[-0.00982,0.01523]`) versus the frozen minimum
  `log(1.5)=0.4055`; both local interaction CIs cross zero; negative-boundary
  third-state admission fails; and clear-bin affine residual gates fail.
- ASCC v2 is therefore a hard NO-GO: this is a generic lexical calibration
  shift, not reader-ambiguity-selective commitment collapse. Replication edges,
  second model, swap controls, hidden patching, and mitigation are all stopped.
  The write-once analysis and hash-bound decision are under
  `corrected_runs/ascc/huatuo_factorial_score_v2/`.

## 2026-08-02 23:52 UTC — clinical T2 analysis frozen before labels or unblinding

- The 24-image, nine-arm VQA-RAD physician screen now has a complete blinded
  return path: independent A/B validation, a no-mapping adjudication sheet,
  immutable-review verification, clean consensus freeze, and only then the
  private method join. The workflow never auto-creates a clinical label.
- Its statistical analysis is pre-registered before any physician label is
  available. The primary endpoint is paired any-visual-error reduction by
  image with Holm correction across eight candidates. Promotion additionally
  requires benefit in at least 12 exact matched-coverage groups and simultaneous
  non-inferiority for omission/required recall, direct correctness, harm,
  refusal, length, and visual-claim count.
- This closes the previous evaluation leak where a shorter or claim-free answer
  could look safer. Ten focused adjudication/analysis tests pass. The pack is
  still awaiting real clinicians, so no method has clinical efficacy or T3
  authorization yet.

## 2026-08-02 23:41 UTC — superseded: ASCC v1 assay is construct-invalid

- The v1 Huatuo run completed 1,552/1,552 forwards, but a pre-admission
  construct audit invalidated it: asymmetric markers, a non-probabilistic
  commitment coordinate, noun/speech-act confounding, absent third-state
  admission, and false pairing of independent images. Its later numerical
  analysis is provenance only and cannot support either a positive or negative
  ASCC claim.
- The sole authorized successor is the untouched symmetric three-state 2x2
  factorial v2. It uses `absent/uncertain/present`, changes only the clinical
  noun within speech act, treats images as independent within reader strata,
  and adds explicit third-state admission, local-polarity, clear-bin affine,
  text-only, and image-swap controls.
- Factorial v2 primary scoring is running as monitored detached job
  `huatuo-ascc-factorial-primary-v2`; no replication edge, second model,
  hidden-state patch, OE claim, or mitigation is authorized before its full
  conjunctive screen passes.

## 2026-08-02 23:29 UTC — VISTA reaches unified 256-token ablation T2; clinical pack is frozen

- VISTA was rerun on the same 32-case VQA-RAD OE prefix at the unified
  256-token budget with greedy, method-off, official VSV-only, SLA-only, and
  combined arms. Method-off is byte- and token-identical to greedy. Every active
  arm has 32/32 aligned nonempty outputs, zero cap hits, zero function-word-only
  outputs, and complete endings for the prompt-required subset.
- VSV-only and SLA-only each change 5/32 token sequences; combined changes only
  the three-qid intersection (3/32). Thus both modules activate, but combined
  behavior is not additive. This is still no clinical efficacy evidence.
- A model-blinded, image-disjoint physician screen now covers 24 images and
  nine arms (greedy, beam, VCD, OPERA, PAI, AvisC, VSV, SLA, combined). Exact
  answer equivalence collapses 216 method assignments to 101 unique answer
  units without losing the private method mapping, more than halving review
  burden. Hash-locked reviewer A/B deliveries each contain all 24 groups; the
  first ten are calibration and the remaining fourteen are independent double
  review. No private mapping is present and unblinding is false. T3 stays false
  until both reviews establish benefit without fabrication/omission/length/
  claim-count/refusal exchange.


## 2026-08-02 23:20 UTC — VHR is source-qualified but incompatible with the certified medical backend

- Official VHR was pinned at ACL 2025 commit
  `f0db54a7eae62b4b8d1d585636a446ed40799512`; remote HEAD, Apache-2.0 license,
  README, entry point, attention intervention, and modified generation stack are
  hash-recorded. T0 remains a source/license/entry pass.
- Official VHR replaces Transformers 4.45 `LlamaSdpaAttention` and names HF
  LLaVA-1.5/Vicuna, LLaVA-NeXT/Vicuna, and InstructBLIP/Vicuna. The current
  trace-certified medical backend is `LlavaMistralForCausalLM`, model type
  `llava_mistral`, with a custom Transformers 4.36.2 stack. Huatuo/Hulu also do
  not use the official path.
- T1/T2 are therefore not authorized for the current medical common-protocol
  matrix. Rewriting VHR for Mistral would be a semantic port rather than
  official execution or method-off identity. It may only reopen with upstream
  target support or a separately labeled diagnostic port plus official-model
  equivalence; no GPU was spent.


## 2026-08-02 23:17 UTC — natural-OE diagnostic completion fails both frozen gates

- A new reader-stratified design removed the prompt-attractor confound and the
  weak-parent confound: 48 image-disjoint VinDr cases cover two semantic edges,
  each with 12 child-0/3 and 12 child-3/3 images while the parent observation is
  unanimously 3/3. Generation saw neither reader labels nor target edges and
  used one ordinary abnormality-listing prompt. A one-image canary passed before
  the exact 48-case PPID1 run; all shards and hashes validate.
- Response geometry fails: 48/48 outputs are nonempty with zero literal
  refusals, but 10/48 hit the 256-token cap, mean generated length is 128.2
  tokens, and only 4/48 obey the requested one-sentence form. This is not the
  earlier extreme template attractor (prefix-10 Top-1 29.2%, exact Top-1 6.25%),
  but it exceeds the frozen 5% cap-hit ceiling.
- The strict, reader-blind-before-join extractor freezes only two spontaneous
  parent-to-diagnosis events: infiltration-to-pneumonia has one child-0/3 and
  one child-3/3 event; nodule/mass-to-tumor has zero. No edge reaches the frozen
  four-events-per-extreme gate. Diagnostic completion therefore has no usable
  natural-OE substrate here; physician review, second model, larger generation,
  and hidden-state replay are all false. Verbosity is not repackaged as
  autoregressive lock-in because that branch's v4 construct is rejected and v5
  remains unsigned.


## 2026-08-02 23:02 UTC — diagnostic-completion full-union expansion is blocked by source admissibility

- A cross-artifact progression audit rejects the proposed 1,876-image
  parent-union generation. The newly counted 12/45/114 transitions all derive
  from the same Huatuo presupposition run that failed its frozen length gate
  (11 and 18 matched pairs versus 50 required) and explicitly prohibited human
  claim audit, second-model generation, and hidden-state escalation.
- The apparent strongest condition is invalid for a spontaneous-OE claim:
  negative obligation supplies 114 events but has a 99.5% dominant 10-token
  prefix, only two such prefixes, and 87% exact cross-image report repetition.
  It is the already registered prompt-conditioned Template Collapse diagnostic;
  event multiplicity cannot override source qualification or construct validity.
- A VinDr child vote of 0/3 is reader-panel absence on one image, not sufficient
  clinical truth that a diagnosis is false. The hash-bound gate therefore sets
  full-union generation, current GPU work, clinical interpretation, and replay
  to false. The only surviving successor is a newly frozen natural-radiology-OE
  pilot of at most 128 images with repeated extreme-vote semantic edges,
  template/length controls, and physician construct admission before replay.


## 2026-08-02 22:57 UTC — VISTA reaches source-bound T2; efficacy remains unopened

- The official VISTA VSV code is now used through a hash-checked, reversible
  runtime adapter; Mistral SLA is applied by a temporary forward wrapper that
  restores the exact original method and MLP structure after every sample or
  exception. The canonical model class was not permanently patched for VISTA.
- A current-source LLaVA-Med gate passes 32/32 generated-token identity for
  both canonical greedy and `VISTA_off`; the greedy and method-off answer files
  are byte-identical. Combined VSV+SLA then completes a 32-case VQA-RAD OE T2:
  32/32 aligned and nonempty, zero function-word-only outputs, and 4/32
  generated sequences changed (12.5%; token F1 versus greedy 0.9454). This is
  functional activation, not clinical benefit.
- Three earlier T2 compatibility attempts are retained with zero written
  answers: duplicate `use_cache`, an uninitialized canonical direct-forward
  transient, and a hidden wrapper signature. The fourth version passed after
  bounded fixes. The method ladder now records VISTA at T2 only; T3 remains
  unauthorized until blinded paired clinical claims and VSV-only/SLA-only,
  length, omission and abstention controls are available.

## 2026-08-02 22:42 UTC — confirmatory statistics red-teamed; current 70-case substrate is pilot-only

- An independent reviewer-style statistical attack KILLs the 22:25
  confirmatory analyzer, but does not kill the Specificity Ratchet question,
  native-capture gate or replay runtime. The fatal issue is effective overlap,
  not nominal sample count: among 127 frozen proposals there are 69 exact
  normalized constraint keys and 58 are singletons. Under the label-blind
  frozen split, dev has at most 8 repeated cross-case keys and test at most 5;
  physician admission can only reduce those optimistic ceilings. Both splits
  therefore fail the predeclared 10-block confirmatory overlap requirement.
- The reproducible CPU artifact is
  `corrected_runs/specificity_ratchet/lexical_overlap_ceiling_v1.json`; it uses
  no physician outcome and creates no clinical label. Consequently the current
  70-case pack may support a bounded engineering/mechanism pilot after human
  review, but cannot support a broad ICLR-level specificity claim regardless
  of favorable GPU traces. GPU remains off while a higher-overlap substrate is
  designed.
- The replacement pilot analyzer removes text-only transition from primary
  nuisance adjustment, reports exact lexical overlap separately, checks
  residualized-role effective clusters and maximum cluster leverage, replaces
  the unstable swap/own ratio gate with the direct contrast
  `beta_swap - 0.5 beta_own`, requires both swaps individually, and requires no
  positive own-minus-swap late catch-up. It calls the first endpoint the
  quarter-decoder layer and explicitly states that unreviewed swaps are
  positional controls, not claim-support truth.
- Artifact validation now recomputes the runtime config fingerprint and exact-
  joins every shard to the selected manifest sample, role and row hash. A
  failed split dominates underpowered status rather than being hidden by it.
  The next scientific action is a label-blind targeted expansion with repeated
  semantic constraints; thresholds will not be relaxed to rescue the current
  pack.

## 2026-08-02 22:41 UTC — MedVR frontier audit closes a baseline and novelty gap

- The official MedVR repository was independently cloned at HEAD
  `4fdd671e29487f455c0b88ef9f73d96ca88ff298`; remote HEAD matched. Its
  Apache-2.0 license, training entry, entropy-regrounding implementation and
  consensus reward implementation were inspected and hash-recorded. The
  release contains training code, but no released MedVR checkpoint; its model
  badge links back to GitHub, and README explicitly says the tool-enabled
  evaluation code will be released later.
- MedVR is therefore registered only as a `paper_native` agentic-visual-
  reasoning candidate and is currently `not_admissible`. It is not a
  model-agnostic decoder and cannot enter the Huatuo/Hulu/LLaVA
  `common_protocol` matrix. No source was vendored, no dependency was
  installed and no GPU job was launched.
- The mechanism collision audit now includes MedVR, the 2026 medical
  grounding--sycophancy trade-off study and Anatomy-VLM. They remove any broad
  novelty claim around entropy-guided zoom, fine-grained ROI modeling, or
  hallucination-only evaluation, but do not directly cover the frozen
  physician-supported parent-to-unsupported-child full-answer own/swap replay
  estimand. The remaining novelty claim stays narrow and conditional.

## 2026-08-02 22:35 UTC — active replay path unified and three qualification leaks closed

- The active paper path is now singular:
  `specificity-ratchet-visible-replay-v1`. The runbook contains only the
  label-blind manifest, one-case engineering canary, complete every-case native
  capture, case-cached `all` replay and source-hash-bound frozen analyzer. The
  earlier one-case/full-replay analyzer is retained as engineering history and
  is explicitly barred from paper evidence.
- Partial native canaries now distinguish `canary_passed` from
  `canary_failed`; a failed identity exits nonzero and its immutable shard
  cannot be replaced. Complete capture sidecars now bind the exact replay
  metadata hash, and scientific runtime refuses metadata drift before any
  scoring call.
- Formal case bootstrap now requires at least 95% of all 5,000 frozen
  replicates to be valid. Underpowered splits stop before bootstrapping and
  cannot accidentally become failures or passes. Analysis output is
  write-once, and the manifest binds this additional gate together with the
  analyzer source hash.
- The paper skeleton now matches the executable estimand: supported controls
  need stronger early own-minus-swap evidence; errors need positive own-image
  and swap-surviving late shifts; and the lower bound for the swap/own
  transition ratio must exceed 0.50 independently on both label-blind splits.
  It no longer claims that one reproduced case authorizes replay or that a
  different dev-fitted analyzer is active.
- Sixty-nine Specificity-focused tests pass. The repository regression passes
  419 tests with the legacy commitment-tetrad file isolated, and that file
  passes three tests separately (422 total). All four active CLIs were also
  invoked from a clean shell: each missing/blank gate returned exit code 2,
  created no scientific artifact and loaded no GPU. The PID-1 watchdog remains
  healthy (`142100/142107`); no GPU process is running because physician
  admission is still absent.

## 2026-08-02 22:25 UTC — unrestricted execution fixed; full-case identity and frozen analysis complete

- Codex execution is now consistently configured as
  `sandbox_mode = "danger-full-access"` and `approval_policy = "never"` in
  `/root/.codex/config.toml`; the current service permission profile is also
  unrestricted. The apparent approval prompts found during audit were records
  from July 28--29 session logs, not current requests. New commands omit all
  escalation parameters.
- The active Specificity Ratchet path is now the label-blind-split
  `specificity-ratchet-visible-replay-v1` compiler and case-cached runtime. The
  contemporaneous one-case full-replay canary is retained only as engineering
  history: a single reproduced answer cannot authorize a scientific replay.
- A new complete native-capture gate records raw `output.sequences` for every
  selected Huatuo case, removes only terminal EOS/PAD for comparison, and
  requires exact visible-text identity, exact contextual target-ID identity,
  and identical visual-token counts for own/swap1/swap2. It is atomic,
  resumable and fingerprint-bound; partial canaries and any identity failure
  are refused by the scientific runtime.
- The pre-data analyzer is frozen to the first recorded and final decoder
  layers with image-case cluster bootstrap and fixed nuisance controls. A GO
  requires all four signatures on both dev and test: positive error-selective
  swap-surviving late commitment, positive own-image late commitment, weaker
  early visual evidence for errors, and a lower bootstrap bound above 0.50 for
  the fraction surviving image swaps. Final unsupported detail alone cannot
  pass. The compiler embeds the analyzer source hash and the exact four gates
  in manifest metadata; the analyzer refuses a different hash, bootstrap count
  or seed, eliminating the prior dual-estimand ambiguity. Exact normalized
  constraint text is a fixed effect, role identifiability is checked explicitly,
  and each split needs at least eight cases per primary role.
- Sixty-four Specificity-focused CPU tests pass, including 22 tests on the
  active compiler/capture/runtime/analyzer path; Python compilation and diff
  whitespace checks pass. The real blank 127-edge physician pack still
  refuses before tokenizer/model construction with 3,307 missing-review
  issues, produces no manifest, and starts no GPU. All four active CLIs also
  start correctly from a clean shell without a pre-set `PYTHONPATH`. The PID-1 research watchdog
  remains alive (`142100/142107`).

## 2026-08-02 22:16 UTC — F6-corrected full-answer replay is executable only up to the human gate

- The misleading 21:51 entry below is retained as history but superseded by
  the 21:58 F6 audit and this entry. The isolated parent/child runtime and CLI
  now hard-refuse scientific execution; it cannot silently score the malformed
  shortened stimuli described in the audit artifact.
- The active successor compiles only physician-admitted rows and replays each
  complete frozen Huatuo visible answer. It localizes exact constraint tokens,
  compares them with relative-position-matched non-constraint tokens, and uses
  own-image minus the mean of at least two same-split modality/anatomy,
  different-case swaps with identical visual-token length as the primary
  visual control. The mechanism gate is conjunctive: early image-specific
  support separation, an error-selective own-image late shift, and no matching
  late increase in the image-specific residual. Text-only is secondary lexical
  sensitivity.
- The compiler binds the source answers and original greedy-512 generation
  fingerprint, freezes two known tokenizer-boundary exclusions, enforces
  image-disjoint splits and requires two swap candidates before emission. The
  blank real physician pack still fails closed and cannot create output.
- A one-case native identity canary now runs all CPU gates before model loading,
  directly captures `output.sequences`, and authorizes replay only if its
  decoded text exactly equals the frozen complete answer. Pass and failure
  sidecars are write-once; a failed case may not be replaced. The Huatuo
  adapter implements the exact source decode contract and the replay runtime
  uses atomic, checksummed, resumable shards.
- Hulu's native processor was audited on ten image/text and boundary variants;
  it appends `<|im_end|>` to supervised labels, which the helper excludes from
  clinical target tokens. This is engineering evidence only: its factory is
  hard-disabled because Huatuo answers cannot establish spontaneous Hulu
  behavior. A second model requires a separate Hulu-native full-answer
  substrate and physician admission, and it does not begin before the Huatuo
  signal gate.
- The active sequence and exact commands are frozen in
  `docs/SPECIFICITY_RATCHET_FULL_REPLAY_RUNTIME_V1.md`; the conditional paper
  skeleton now uses the same conjunctive full-answer estimand. A dev-frozen
  analyzer now performs case-clustered bootstrap, nuisance adjustment and a
  fixed 0.2-dev-SD equivalence test; it refuses split/runtime identity drift
  and fewer than ten cases per primary role. Thirty-six focused
  compiler/runtime/canary/adapter/analyzer tests pass. The repository regression
  passes 416 tests with the legacy commitment-tetrad file isolated, and that
  file passes its three tests separately (419 total assertions/tests). A single
  combined process can segfault inside old SciPy/BLAS commitment-tetrad code
  after cumulative imports, so this is recorded as an environment-level test
  harness defect rather than hidden as a pass. No GPU process is running; the
  physician gate remains unresolved and the persistent watchdog remains
  healthy under PID 1.

## 2026-08-02 21:51 UTC — Specificity Ratchet post-review path is code-complete to the GPU gate

- The blank real physician pack was independently exercised through both the
  validator and manifest compiler. Both refuse with exit code 2 on 3,307
  missing-review issues, and the compiler writes neither `samples.jsonl` nor
  `metadata.json`. This reconfirms that no automatic component can create the
  primary clinical labels and no scientific GPU job is currently authorized.
- The manifest compiler now requires every scored child to be the unique exact
  UTF-8 substring of the frozen OE generation. All 127 current proposals pass
  this provenance screen. The original generation anchors spontaneous
  occurrence only; physician adjudication remains the sole support truth.
- A thin Huatuo Specificity bridge now reuses the existing exact contextual
  serializer, multimodal-label alignment and final-logit-certified lock-in
  adapter. It scores the complete parent/child as an empty-prefix continuation
  and introduces no alternate model forward. The generic runtime also rejects
  any parent/child template drift. CPU contracts pass, but the real-model
  conformance canary remains correctly gated on physician admission.
- The conditional paper logic is frozen as a New Problem/Setting paper in
  `docs/SPECIFICITY_RATCHET_ICLR_PAPER_SKELETON_20260802.md`. Mechanism and
  mitigation contributions are explicitly `NOT MEASURED`/`NOT AUTHORIZED`;
  an Oral-level claim requires a held-out transition, selective causal rescue,
  two model families, three edge types and fixed-K clinical improvement.
- The focused post-review/adapter suite passes 24 tests and the current full
  repository suite passes **381 tests**. The persistent watchdog remains under
  PID 1; no GPU process is running because the scientific gate is unresolved.

## 2026-08-02 21:58 UTC — Specificity Ratchet F6-corrected; both human gates deliverable

- The research question is now frozen to one falsifiable transition: whether
  open-ended generation moves from a visually supported parent claim to an
  unsupported descendant constraint through a late lexical-prior crossing.
  The unique prediction, nuisance alternatives, causal intervention and seven
  kill criteria are fixed in
  `docs/SPECIFICITY_RATCHET_FROZEN_RESEARCH_CONTRACT_20260802.md`.  This is the
  sole primary candidate; CECD is only a human-gated secondary branch.
- Two self-contained blinded delivery archives contain one role-specific blank
  sheet and 70 JPEGs each.  They exclude private provenance, model identity,
  reference answers and the other reviewer's sheet; all images open as RGB,
  carry no EXIF, and have no byte/pixel duplicates.  The archives are
  deterministic and path/symlink safe.  Absence of burned-in PHI is not claimed
  because OCR review has not been performed.
- A new deterministic merger validates both returned sheets against the frozen
  candidates and allowed states, requires distinct stable reviewer IDs, rejects
  immutable-field changes and spreadsheet-formula rationales, and copies only
  reviewer fields into a blank adjudication template.  It cannot overwrite an
  existing output and leaves all clinical final fields blank.  Five focused
  tests pass.  Against the real blank pack it refuses with exit code 2 and
  creates no output, so neither code nor an automatic judge can manufacture
  physician truth.
- The exact receive, merge, blinded adjudication, attestation and disposable
  working-pack procedure is frozen in
  `docs/SPECIFICITY_RATCHET_REVIEW_RETURN_WORKFLOW_V1.md`.  No GPU canary is
  authorized until two independent physician returns, physician adjudication,
  fail-closed validation and mechanism-manifest compilation all succeed.
- CECD remains a secondary branch, but its four role-isolated v2 archives are
  now operationally complete outside Git.  An independent rerun verified all
  four exact archive hashes, byte-identical frozen blank sheets, role isolation,
  504 PNGs in each clinical-review archive, and zero unsafe/forbidden members;
  its focused regression test passes.  Each clinical archive is about 2.54 GB
  and contains restricted derived CXR data without encryption, so the
  coordinator must use controlled transfer and send exactly one archive per
  reviewer, never the containing directory.  This completion creates no label,
  does not authorize CECD GPU scoring, and does not change its secondary rank.
- A pre-data F6 audit has now rejected the earlier isolated parent/child
  teacher-forcing stimulus.  At least 19 automatically shortened parents end
  as incomplete surfaces such as `which is` or `can be`, while modifier removal
  can leave malformed punctuation such as `a, mass`.  Clinical entailment does
  not make these natural standalone answers, so no result from that runtime may
  be generated or interpreted.
- The corrected mechanism substrate replays Huatuo's complete visible OE answer
  and localizes only the added-constraint tokens.  All 127 child proposals are
  exact unique substrings; a reproducible CPU audit finds 125/127 constraints
  exactly token-scoreable and freezes two boundary-spill exclusions.  The
  source metadata token IDs are decode-then-retokenize values, not native
  generation IDs, so a one-case post-admission deterministic identity canary
  must directly capture `output.sequences` before any native-trajectory claim.
- Text-only is demoted to lexical sensitivity because removing the image changes
  positions and visual-token count.  The primary visual control is now at least
  two frozen same-modality/anatomy different-case image swaps, with the complete
  target and visual-token length held fixed.  The estimand is the layerwise
  constraint-versus-matched-nonconstraint own-image-minus-swap difference-in-
  differences.  A Huatuo full-target RGB adapter is CPU-tested; 16 adapter and
  runtime tests pass, its fingerprint declares the public JPEG/PNG RGB renderer,
  and it contains no DICOM provenance.  No GPU was loaded.

## 2026-08-02 21:36 UTC — one qualification authority; four frontier baselines audited at T0

- `configs/unified_eval/method_ladder_v1.json` is now the sole authority for
  paper qualification.  The runtime-only `common_baselines.yaml` delegates to
  it and no longer duplicates stale `admissibility` fields.  This removes the
  contradiction in which VCD/OPERA/PAI/AvisC were still marked identity-blocked
  despite the canonical 32-case OE T2 audit.  Their exact status is T2 passed,
  T3 unauthorized pending blinded paired clinical-claim evaluation.
- Official HEAD, entry point and license scope were independently checked for
  four missing frontier baselines. VISTA (ICML 2025) and VHR (ACL 2025) pass
  T0 only; VISTA's seven local core files are byte-identical to official HEAD.
  AGLA and ClearSight are `not_admissible`: neither official repository has a
  root license covering its method code (ClearSight's embedded LLaVA license
  does not extend to ClearSight). No T1 adapter or GPU job was authorized.
- The cross-answer-space substrate artifact now binds nine exact inputs,
  evaluator hash, command and deterministic fingerprint; its F6/F7 NO_GO is
  reproducible.  Eight focused tests enforce the single-authority mapping,
  T2 evidence ceiling, T0 license decisions and source provenance.  Frozen T0
  evidence is in `results_reference/baseline_t0_source_audit_20260802.json`.

## 2026-08-02 21:41 UTC — real CECD reviewer deliveries built and independently verified

- Four deterministic, role-isolated delivery archives were built under the
  restricted PhysioNet data root.  Each clinical reviewer receives one blank
  252-row sheet plus exactly 504 blinded PNGs; the clinical-template and
  language reviewers receive only their respective blank eight-row wording
  sheet.  No role can see another sheet, sealed mapping, selected-claim file,
  model output, unsafe path, or prefilled decision.
- The independent verifier passed all four archives and checked every included
  image against the frozen source pack.  The two clinical archives are 2.54GB
  each; 348GB remains free, preserving the 100GB floor.  Anonymous hashes and
  exact commands are frozen in
  `corrected_runs/unified_eval/physician_review/cecd_admission_delivery_v1/verification.json`;
  restricted images and archives remain outside the repository.
- This completes delivery engineering only.  Human review has not started,
  no label was inferred, and CECD model scoring/GPU execution remains
  unauthorized until all independent reviews and the frozen analyzer pass.

## 2026-08-02 21:20 UTC — qualified OE brevity controls invalidate lexical promotion

- Reference-blind suffix deletion was run on the qualified 200-question
  Huatuo-512, Hulu-256 and LLaVA-256 VQA-RAD OE outputs with 5,000
  image-cluster bootstraps.  At the predeclared lexical coverage tolerance,
  first-48-word Huatuo raises token-F1 from 0.03761 to 0.05319; paired delta is
  +0.01558 [0.01245, 0.01886] while reference-phrase coverage changes by only
  -0.010 [-0.02564, 0].  Hulu has a smaller but positive matched-control delta;
  LLaVA is already short.
- The identity-qualified 32-case mitigation screen gives negative token-F1
  point deltas versus canonical greedy for beam, VCD, OPERA, PAI and AvisC.
  First-sentence deletion alone significantly raises VCD and AvisC token-F1
  without changing exact reference-phrase coverage.  Lexical metrics therefore
  cannot promote any method to T3; the existing 24-image, six-method blinded
  physician pack remains the only clinical gate.
- Both OE evaluators now bind command, source hash, input hashes, bootstrap
  count and seed into every output.  Runs use an isolated Python bytecode cache
  to prevent shared-worktree timestamp races.  Full evidence and the strict
  claim boundary are in `docs/OE_BREVITY_AND_MITIGATION_CONTROL_AUDIT_20260802.md`.

## 2026-08-02 21:06 UTC — v4 partial runtime invalidated; persistent monitor remains healthy

- The Huatuo adapter canary passed its engineering contracts and the detached
  process wrote 21 of 48 v4 shards before the exact-string F6 audit completed.
  Those shards are now preserved but formally `not_admissible`: the manually
  assembled prefix plus fixed continuation changes grammar and proposition
  boundaries, so no partial result may be analyzed, resumed, or merged into a
  successor.  `F6_REJECTION.json` exposes all ten exact serialized stimuli and
  `INVALIDATED.json` binds the manifest, audit, and partial-shard hashes.
- The failed state at 21:01 UTC is an intentional fail-closed protocol mismatch,
  not an infrastructure failure.  The invalid v4 task is absent from
  `research_required_jobs.json`; there is no GPU child to restart.  The common
  watchdog itself remains healthy under a PID-1 supervisor and continues to
  audit the remaining required terminal jobs every 30 seconds.
- A v5 run is not yet authorized.  It must derive every prefix from tokenizer
  boundaries of one frozen natural full sequence and pass independent clinical
  construct review; the current validator deliberately returns CPU-only even
  after such review until a separate tokenwise runtime passes conformance.
- An unsigned v5 review pack now freezes the two pilot target sequences and two
  target-free natural controls without reading any v5 scores.  Its body hash is
  verified, the lung-opacity control's one-token mismatch is explicitly
  disclosed, and the validator correctly rejects the blank reviewer identity
  and attestation.  This is the next scientific gate, not permission to occupy
  the GPU.

## 2026-08-02 20:56 UTC — unrestricted execution verified; AR-SoS killed before GPU

- Codex now runs with the effective policy `approval_policy=never` and
  `sandbox_mode=danger-full-access`; `/`, `/root`, `/home/dbw`, and this
  repository are trusted.  `codex doctor` independently reports approval
  `Never` and an unrestricted filesystem.  The 22 historical escalation calls
  all predate the configuration change; there have been zero afterward.  No
  repository experiment calls Codex escalation.  The only remaining interactive
  shell option is PhysioNet `wget --ask-password`, which is not used because the
  dataset is already mounted at `/workspace/vinbigdata`.
- The preregistered CPU-only AR-SoS substrate audit exhaustively reconstructed
  5,501 exact R8/R9/R10 images and 44,008 finding rows under image-disjoint
  pilot/dev/confirmation splits.  Eight of 56 ordered A-to-C associations pass
  the dev association gate, but only five distinct A-B pairs have at least eight
  dev A=3/3,B=3/3,C=0/3 cases.  No eligible combination reaches the frozen
  confirmation minimum of 40 (exhaustive maximum 39), and the VinDr release has
  no native report wording with two-reviewer proposition-equivalence admission.
  These are fatal F7 and F6 failures: AR-SoS is removed from the GPU queue with
  no threshold relaxation and no automatic/model-authored wording substitution.
  Artifact: `corrected_runs/vindr_v2/ar_sos_substrate_audit_v1/substrate_audit.json`;
  four focused CPU tests independently pass.
- Clinical Autoregressive Lock-in remains CPU-only and is not currently
  authorized.  Although dev-v4 corrected the claim-specific prompt and has zero
  pilot overlap, a later F6 audit found that its runtime appends the same claim
  fragment after every nested prefix.  This creates invalid stimuli such as
  `The chest X-ray pleural effusion` and
  `...shows no common abnormalities opacity`; the early/late contrast therefore
  confounds lock-in with grammar and discourse-boundary changes.  Dev-v4 is now
  explicitly non-reportable and GPU-prohibited pending an exact construct
  validator and a natural, proposition-controlled redesign.  No GPU result
  exists.  A detached v4 canary was started before this F6 finding propagated;
  it failed immediately in the multimodal-expansion contract, produced no
  analyzable row, and was terminated.  Its required-job entry was removed so
  the watchdog cannot restart it; the log is retained as audit evidence.
- Broad Anatomy--Finding Conjunctive Binding is also removed from the mainline.
  Locally usable PadChest-GR/Chest ImaGenome/MS-CXR location-gold rows are zero;
  MS-CXR cannot satisfy the frozen three-finding/two-side cardinality gate even
  in principle.  PadChest-GR is request-only and its publication does not expose
  the required finding-by-side-by-split patient counts.  Independently, current
  mechanism literature already supplies layerwise spatial identity/binding and
  causal patches, while current medical grounding work supplies
  finding--anatomy--laterality verification and correction.  Combining them on
  CXR would be a dataset transfer, not an Oral-level mechanism.  The only narrow
  residual, patient-reference-frame mediation, remains conditional on currently
  unavailable orientation plus patient-side gold and is not GPU-authorized.
- Broad Answer-Space Evidence Substitution is likewise killed by both novelty
  and substrate gates.  Tinted Frames already covers same-semantics OE/YN/MCQ,
  layerwise visual disengagement, and causal restoration; prompt-copying heads,
  medical candidate-answer injection, and same-image/different-prompt patching
  cover the remaining broad mechanism.  Locally, original VQA-RAD has zero
  exact cross-space question pairs, only one conservative test candidate, and
  202 train/test image-hash overlaps; SLAKE has only five conservative test
  abnormality pairs; MedHEval CE is GPT-4-synthesized.  The formal
  dual-reviewed pair count is zero, so F6 and F7 both fail.  Four independent
  CPU tests pass for the frozen audit; no manifest or GPU job was created.  The
  only literature delta left after the collision scan is precisely the
  reader-grounded CE-to-OE clarity-erasure mechanism already rejected by the
  frozen two-plane experiment, so this branch cannot be revived under a new
  name.
- After these fail-closed eliminations, **Specificity Ratchet is the highest
  value surviving candidate**, not because it has positive mechanism evidence,
  but because its output-side spontaneous parent-to-unsupported-child cell
  remains distinct from FINER/MedVIGIL and has the smallest valid admission
  burden.  Its blinded pack is integrity-valid (70 images, 127 edges; estimated
  2.3--3.5 hours per independent physician plus adjudication) but intentionally
  blank.  The scientific validator currently refuses with 3,307 missing-review
  issues, so no GPU work is authorized.  CECD remains the second conditional
  candidate and is also awaiting independent clinical-equivalence review.

## 2026-08-02 20:37 UTC — template diagnostic frozen; no lock-in probe authorized

- The full Huatuo template diagnostic completed under a PPID1 supervisor with
  2,000 label permutations and image-cluster bootstrap. Negative-obligation
  generation has 45 exact reports (Top-1 36%, T80 12) but only two distinct
  10-token prefixes, with a 99.5% dominant prefix. Neutral and existential
  dominant-prefix shares are 80% and 62%.
- All 200 DICOM file hashes and rendered-pixel hashes are unique; there are no
  prompt echoes or literal refusals and only one cap hit. Official anonymized
  DICOMs contain none of PatientID, Study/Series/SOP UID, so cross-patient and
  cross-study collision cannot be certified.
- After Benjamini--Hochberg correction across 24 prompt/finding tests, only
  neutral×lung-opacity template association survives (`q=0.04798`). Thus weak
  reader signal and strong template attraction coexist; neither “no vision” nor
  “reader evidence controls the output” is supported.
- The diagnostic is registered as `not_admissible` under event
  `9841aa26...8873b0f`. Missing patient/study linkage, a second medical VLM and
  a token/layer causal transition keep `paper_mechanism_authorized=false`; no
  hidden-state or subtraction experiment is launched. The completed job was
  removed from recovery manifests. The frozen single-thread BLAS test contract
  passes 347 tests; an unconstrained BLAS run reproduced the known native
  tetrad-bootstrap crash and is not counted as a test result.

## 2026-08-02 20:25 UTC — paper-scope integrity gate rejects early-erasure story

- A formal idea/paper-skeleton audit rejects drafting an early-erasure,
  redundant-evidence, shared-source-center, or presupposition-mechanism paper
  from current artifacts. These are data-refuted or fail their frozen
  progression gates; prose cannot repair them.
- A generic “mechanism boundary” paper is also not yet verifiable because the
  same formal reader-residual protocol did not reach two primary models. Hulu
  is not used as a post-hoc rescue. The decision is frozen in
  `docs/PAPER_SCOPE_GATE_20260802.md`.
- Paper development remains conditional on real construct-validity labels for
  Specificity Ratchet, CECD, native OE, and mitigation. The newly implemented
  Specificity teacher-forcing runtime remains blocked by its blank physician
  pack; no model or GPU is loaded before that gate passes. The complete
  DICOM-capable test suite passes 340 tests and `git diff --check` is clean.

## 2026-08-02 20:20 UTC — Presupposition screen fails length admission; template lead collision-bounded

- The exact Huatuo Clinical Presupposition generation job finished with exit
  code 0.  Independent audit verified 600/600 atomic shards, 200 unique images
  by three conditions, zero errors/empty responses, exact expected identities,
  direct-vs-standard generation conformance, and all frozen config, model,
  renderer, manifest, fingerprint, aggregate, and filename hashes.  One
  existential answer hit the generation cap; clinical truth remains entirely
  `pending_shared_audit`.
- The preregistered matched-length admission already fails without inspecting
  any clinical labels: existential-vs-neutral has only 11/200 exact-token pairs
  (10/200 visible-token pairs), and negative-obligation-vs-neutral has 18/200,
  below the frozen minimum of 50.  The engineering artifact is valid, but the
  bidirectional Clinical Presupposition mechanism screen is ineligible.  No
  threshold relaxation, post-hoc truncation, or second-model GPU run is allowed.
- A strong prompt-conditioned template pattern is real as a diagnostic: exact
  pleural-effusion and opacity reports repeat across images with conflicting
  VinDr reader bins.  It is not promoted to a paper idea.  The 2026 Template
  Collapse work directly covers medical report-template repetition and CLarGen,
  while Pensieve already covers same-context subtraction using real reference
  images.  The only remaining conditional question is narrower: whether reader
  polarity is present before generation but image-causal influence collapses at
  a reproducible prefix token/layer and can be selectively restored by patching.
  The completed CPU diagnostic reports exact-text cross-image repetition of
  64%/76%/87% and prefix-12 top-one concentration of 80%/62%/86.5% for
  neutral/existential/negative-obligation.  Neutral positive-effusion language
  retains reader association (AUROC 0.695, 95% CI [0.545, 0.838]), whereas the
  existential template does not (0.475 [0.411, 0.536]).  These are frozen
  literal diagnostics, not clinical truth; analysis fingerprint
  `a5c67bf1...e661919fe` binds the final code and outputs.
- MedVIGIL does not unlock Evidence-Source Erasure: the release lacks independent
  current-image/history/prior/knowledge source labels and source-preserving
  pairs, and has missing/repeated images, absent official splits, and severe MCQ
  answer-position imbalance.  This backup direction remains halted.
- The Specificity Ratchet teacher-forcing runtime is now fail-closed and
  CPU-tested (14 focused tests): exact contextual offsets, same-image
  parent/child plus text-only traces, per-layer signals, atomic resume, and exact
  negative-control bins are implemented.  It correctly refuses the still-blank
  physician pack (3307 missing review issues) before loading any model or GPU.

## 2026-08-02 20:21 UTC — Clinical Presupposition stopped before clinical review

- The exact Huatuo generation job completed 600/600 outputs for 200
  outcome-blind VinDr images with exit code 0, no residual error shards, one
  cap hit, and no literal refusal matches. The aggregate generation hash is
  `1335d7c...cd4d81`.
- It fails the pre-registered within-image answer-length qualification:
  existential-vs-neutral retains 11/200 matched pairs and
  negative-obligation-vs-neutral 18/200, versus 50 required for each. The
  explicit at-most-30-word response form was followed by 65/200 neutral,
  3/200 existential, and 0/200 negative-obligation answers.
- This is a response-geometry confound, not evidence for or against clinical
  presupposition errors. `human_claim_audit_authorized=false` and
  `second_model_generation_authorized_from_this_model=false`; no physician
  pack, Hulu run, or LLaVA run will be launched from this candidate.
- The immutable generation and failed qualification are registered under event
  `39c5a353...174ce53b` as `not_admissible`. The completed job was removed from
  active/required recovery manifests so watchdog recovery cannot turn a failed
  scientific gate into a retry.

## 2026-08-02 20:13 UTC — localized evidence-survival pilot stopped at its gate

- The completed Huatuo pilot tested whether 3/3-reader positive findings retain
  definite-positive evidence under progressive lesion-token ablation better
  than 2/3 findings, with equal-count background-token ablation and token-norm
  preservation. It yielded 79 valid cases and 58 directionally admitted cases.
- The predeclared redundancy direction failed: adjusted 3/3-minus-2/3 ROI
  survival was -0.0326 with image-cluster bootstrap 95% CI
  [-0.0666, -0.00068]. Both nodule/mass and pleural-effusion coefficients were
  negative. Lesion ablation was stronger than background ablation, so the
  manipulation activated, but the reader-unanimity mechanism prediction did
  not.
- No dev replication or mitigation branch is authorized from this pilot. The
  result is retained as a bounded negative mechanism test; no threshold,
  finding, or layer will be selected post hoc to reverse it.
- NumPy 1.x/2.x trapezoidal-integration compatibility was repaired in the
  frozen analyzer. The DICOM-capable Huatuo environment passes the complete
  suite: 329 tests passed; `git diff --check` is clean.

## 2026-08-02 20:08 UTC — mitigation T2 complete; no method promoted

- Detached job `vqa-rad-oe-baselines-t2-v10` completed successfully. Greedy,
  beam-5, VCD, OPERA, PAI, and AvisC each produced 32/32 qualified 256-token
  outputs with 0% cap hits and 100% terminal completeness. The greedy arm is
  token-exact against the current canonical LLaVA greedy-256 prefix.
- Beam, VCD, OPERA, PAI, and AvisC changed 68.8%, 87.5%, 65.6%, 37.5%, and
  90.6% of outputs respectively, so their execution paths are demonstrably
  active. None improved the auxiliary token-F1 over greedy, but lexical scores
  are not clinical hallucination metrics and are not a failure or efficacy
  claim.
- `method_evidence_ladder_v4.json` now records T0/T1/T2 pass for VCD, OPERA,
  PAI, and AvisC while retaining T3 as missing and full evaluation as
  unauthorized. `T3_authorized_methods=[]` remains the only valid promotion
  decision.
- A blinded 24-image/144-answer-unit two-physician promotion pack is frozen at
  `corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t2_v1`.
  Six calibration groups and eighteen independent groups are double-reviewed;
  no T3 generation is allowed until paired clinical claim evaluation passes.
- The physician OE v2 schema now has a fail-closed completed-review validator
  covering immutable inputs, structured claims, the explicit no-claim XOR,
  visual/nonvisual support logic, omissions, harm, confidence, and rationales.
  A passing validator remains schema evidence only, never clinical efficacy.
- The exact Clinical Presupposition Huatuo generation probe remains live under
  the shared GPU lock with atomic shards and watchdog recovery. It is a
  generation-only diagnostic; human or multi-reader claim evaluation and a
  second model are required before any mechanism conclusion.

## 2026-08-02 19:52 UTC — physician OE v3 frozen; certified mitigation T2 launched

- The current physician OE review input is
  `corrected_runs/unified_eval/physician_review/vqa_rad_native_oe_v3`: 100
  image-disjoint groups and 300 blinded answer units from the admissible
  Huatuo-512, Hulu-256, and LLaVA-256 generations.  The v2 review contract now
  freezes a structured normalized claim, explicit
  `no_clinical_claims XOR atomic_claims`, harm labels, and confidence scale.
  Earlier review bundles are retained as schema-incomplete history only.
- Reviewer assignment is frozen before labels: ten shared calibration groups,
  twenty independent double-review groups (60 answer units), and 35 disjoint
  single-review groups per reviewer. Every answer receives at least one real
  physician review. Mapping and assignment files remain private until labels
  and the clarification log are hash-locked.
- Two obsolete reader-residual confirmation waiters remained alive despite
  removal from the watchdog. They were explicitly stopped and their state now
  records scientific-gate termination. No confirmation feature directory was
  created; both counts remain 0/1920.
- The repaired LLaVA mitigation port already passes 32/32 and 128/128
  generated-token identity. Detached T2 job `vqa-rad-oe-baselines-t2-v10`
  was launched to run greedy, beam, VCD, OPERA, PAI, and AvisC on 32 cases with a
  256-token budget and no legacy keyword stopper. It owns the shared GPU lock
  and is protected by the required-job manifest. Baselines that lack an
  admissible official license/checkpoint/path remain T0 failures rather than
  local approximations.

## 2026-08-02 19:46 UTC — unrestricted execution confirmed; nonblocked mechanism work continues

- Codex execution is explicitly configured with `sandbox_mode =
  "danger-full-access"`, `approval_policy = "never"`, and an exact trusted
  project entry for `/home/dbw/ANCHOR`.  The detached watchdog remains PPID 1;
  no command-approval prompt is part of the current execution path.
- CECD remains paused only at its scientific human-admission gate, not an OS or
  Codex permission gate.  The 32/160 pre-admission claims (608 atomic cells)
  remain resumable engineering artifacts and are excluded from efficacy or
  mechanism interpretation until the blinded reviews pass.
- The official Treble audit now fails closed at method level: its released
  paper/code disagree on text direction, cross-modal counterfactual, and visual
  perturbation; the release has no per-claim NDE scalar and no explicit root
  license.  Seventeen joint Treble/CECD tests pass.  No CECD scalar may be
  relabelled as Treble; exact reproduction remains blocked rather than failed.
- Specificity Ratchet now has a fail-closed two-physician adjudication validator
  and mechanism-manifest compiler.  The blank 127-edge pack correctly produces
  no scientific artifact; exact constraint spans round-trip for 127/127
  candidates, and ten focused tests pass.  No GPU work is authorized before
  real review.
- A generation-only VinDr Clinical Presupposition probe is being prepared as
  the independent nonblocked branch.  It will precompute open answers under
  three pragmatic prompt conditions with atomic resume while leaving clinical
  claim truth to the separate unified evaluator.

## 2026-08-02 19:41 UTC — persistent recovery corrected and OE continuation live

- Both persistent OE jobs have now completed with exit code 0.  Hulu and LLaVA
  each have 200/200 exact-qid greedy-256 outputs; the independent acceptance
  monitor verified current hashes, qualifications, generation contracts,
  lexical-auxiliary bindings, and registry events.  Both have 0% cap-hit and
  100% required response-form completion.  The resulting evidence ladder v3
  has zero stale registry events and correctly retains `full_pass=[]` because
  no clinical claim-efficacy evaluation has yet passed.
- The CECD v2 human-admission pack has passed an independent integrity/blinding
  audit: 252 clinical pairs (240 primary + 12 lossless identity controls), 504
  hash-matched PNGs, eight language pairs, blank reviewer sheets, and no
  reviewer-visible image-ID or transform leakage.  Its status remains
  `awaiting_independent_human_reviews`; integrity is not human admission.
- `research-watchdog-v1` remains detached from VSCode (PPID 1, 30-second
  heartbeat).  The authorized `native-oe-greedy256-full-v1` pipeline completed
  Hulu 200/200 and immediately continued to LLaVA; Hulu passes the frozen
  generation gate with 0% cap hits, 100% required response-form completion,
  and exact 200-qid alignment.  This certifies generation only, not clinical
  correctness.
- A second detached monitor, `native-oe-full-acceptance-v1`, now waits on the
  two-model pipeline.  On success it immediately verifies hashes, contracts,
  registry events and qid alignment, then emits
  `method_evidence_ladder_v3.json`.  Both jobs are in the required recovery
  manifest and therefore survive terminal or container disconnection.
- A stale cross-session manifest entry briefly restarted
  `cecd-huatuo-stage1-v1`, extending the preserved partial diagnostic to
  32/160 claims (608 cells).  It was stopped with SIGTERM, removed from both
  recovery manifests, and is not efficacy evidence.  The executable now has a
  fail-closed runtime admission check before model loading or GPU locking:
  absent/failed/stale reviewer artifacts make it exit.  Eight focused tests
  cover recovery and this admission boundary.
- CECD GPU scoring remains forbidden until the blinded v2 pack receives two
  independent clinical reviews plus separate clinical-template and language
  reviews, and the versioned analyzer authorizes scoring.  Existing CECD
  shards are retained only for provenance and possible post-admission reuse.

## 2026-08-02 19:30 UTC — boundary cutoff honored; CECD/OE successors

- The Huatuo reader-residual dev gate is treated as the canonical cutoff:
  confirmation remains stopped, and stale waiting/zombie state files are not
  authorization to restart it. Hulu is not used as a post-hoc rescue.
- The complete CECD CPU render audit has 160/160 shards and zero runtime
  errors. One `pleural_effusion` orbit fails the `center_plus_0p05w` guard
  (`roi_saturation_increase=0.22975`); the runner/analyzer now preserves this
  as a whole-orbit exclusion instead of substituting baseline pixels.
- CECD model scoring remains unauthorized pending the protocol's real human
  gate. A blinded v2 pack is being built outside Git at
  `/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2` for two
  independent clinical reviewers plus separate clinician/language template
  review. The fail-closed review analyzer and 22 focused CECD/admission tests
  pass. No pixel heuristic is relabeled as clinical equivalence.
- The independent Specificity Ratchet physician pack is already complete and
  verified: 70 unique VQA-RAD images, 127 proposed edges, exact substring and
  reviewer-blinding checks passed. Model-derived edges remain proposals only.
- Native OE governance found the old 64-token Hulu/LLaVA 200-case outputs
  inadmissible for efficacy (cap-hit 36%/9%; terminal completion 63.5%/91%).
  They are now append-only `identity_only` artifacts. Hulu's updated backend
  passes 32/32 generated-token identity; greedy-256 and beam-4 T2 generations
  pass the format/cap gate. LLaVA T2 is active, and a monitored successor will
  generate both 200-case greedy-256 baselines only if T2 completes.
- Persistent recovery protects `native-oe-controls-t2-v1`,
  `native-oe-greedy256-full-v1`, and `cecd-human-admission-pack-v2` while the
  rejected VinDr confirmation jobs remain excluded.

## Cross-session live coordination (2026-08-02 18:42 UTC)

- Session `019fc2bf-3563-7c13-83eb-d11fea473e11` should treat this file and
  the versioned artifacts below as the shared execution truth.  Do not launch
  duplicate GPU work while the named tmux session or lock exists.
- Persistent Python 3.10.20 is restored at `/home/dbw/.runtime/miniconda3` and
  mapped to `/opt/miniconda3`.  The recreated Hulu and LLaVA backends each pass
  a frozen 32-case post-restart gate with 1.0 normalized exact, token F1, and
  generated-token-ID exact rate; see
  `corrected_runs/unified_eval/sanity/post_restart_runtime_identity_v1/identity.json`.
- `research-watchdog-v1` is alive with a 30-second heartbeat.  The restart
  marker is removed.  The current active manifest additionally tracks the
  external VinDr audit and LLaVA RAG causal-control ladder, so dead `running`
  supervisors are recovered without retrying scientific failures.
- The complete read-only `/workspace/vinbigdata/train` mount contains all
  15,000 official VinDr DICOMs.  Seven complete files from the earlier
  authenticated download are byte-identical to the mount; its eighth file was
  an interrupted partial and is correctly rejected.  The obsolete interactive
  download, post-download, and mechanism tmux chains were stopped without
  deleting data.  No PhysioNet password is currently needed.
- The fixed R8/R9/R10 v2 manifest freezes eight four-bin-qualified findings,
  image-disjoint pilot/dev/confirmation splits, 3,200 claim rows, and 2,341
  unique images.  Huatuo dev hidden-state collection v3 completed 640/640 in
  316 seconds and Hulu completed 640/640 in 417 seconds.  Both source hashes
  match their output configs and every same-shape final-norm conformance check
  passed; the earlier last-token BF16 discrepancy was a measurement artifact,
  not a loosened tolerance.  The strict external audit validated all 15,000
  source IDs and all 2,341 selected DICOMs with no missing, extra, or invalid
  files on the read-only mount.
- The preregistered dev-only Virtual-Reader screen is terminated before
  confirmation.  Signed evidence has a positive pooled direction, but the
  fixed-panel model is 12.7% worse in Brier than the finding prior and 12.2%
  worse than the unconstrained evidence-only three-state model (95% CI for the
  latter excess 5.45%--19.39%).  Its frozen artifact therefore records
  `spend_confirmation_compute=false`; it is neither a mechanism nor a method.
- The prior Two-Plane clarity-erasure pilot also failed: the best non-final
  claim-token clarity AUROC did not exceed the final layer (delta -0.040,
  image-bootstrap CI [-0.198, 0.115]).  Evidence Survival is not being
  promoted: its first run had a BF16 readout floor and target-dependent union-
  reader ROI, while MedVIGIL/GACD already collide with progressive ROI-decay.
  The active fail-closed successor is a nested image-grouped residual screen
  for fixed-panel unanimity after `spline(Yes-No) × polarity stratum +
  finding`; only >=0.05 AUROC and >=5% Brier gains with CIs above zero can
  justify confirmation.  A paired DICOM-render/source-style mechanism is the
  next independent branch if that residual gate fails.
- The Huatuo unanimity-residual dev screen has now failed that frozen gate.
  Its selected layer-21 routing probe reaches macro delta AUROC `0.0411` and
  relative Brier improvement `1.76%`; the Brier bootstrap interval crosses
  zero, it does not beat a same-dimensional random projection, and only 5/8
  findings improve in Brier direction.  The untouched 1,920-case confirmation
  collectors were therefore removed from the watchdog and stopped before any
  confirmation feature was collected.  An interrupted Hulu analyzer had begun
  an automatic full rerun; it was also stopped because one Huatuo dev failure
  makes the frozen two-model progression gate impossible.  Hulu is not used to
  rescue the hypothesis post hoc.
- The 160-claim Huatuo DICOM-render pilot completed 160/160 with zero errors
  under a hash-frozen, label-independent clinical guard, but its scientific
  gate failed 0/4 findings.  Large descriptive render-orbit diameters did not
  survive deterministic half-A transform selection and half-B signed/high-
  margin confirmation; pulmonary fibrosis also lacked a positive reader-step
  CI.  This is a concrete warning that orbit size or token flips would have
  produced a false-positive style story.  Recent SPCD/LENS/
  VGS/UniVRSE/Treble work makes raw render stability or perturbation decoding
  a baseline rather than a contribution.  The surviving question is the
  Clinical-Equivalence Composition Defect: whether admitted render and
  speech-act-preserving prompt transformations have a reader-scale two-way
  interaction that adds held-out clinical-error information beyond clean
  margin and both marginal sensitivities.
- Common RAG v3 and its causal controls are complete.  Huatuo and Hulu lose on
  IU-Xray; MIMIC Huatuo is inconclusive and Hulu loses while shortening output.
  LLaVA's raw gains do not survive the full grounding gate: IU-Xray relevant
  retrieval is worse than globally length-matched disjoint context (-0.075,
  95% CI [-0.136,-0.015]); MIMIC is +0.050 but its CI touches zero
  [0.000,0.101].  No dataset authorizes the image-swap stage and no common-RAG
  grounding claim is retained.
- Live logs/states are under `corrected_runs/detached_jobs/`.  VinDr and RAG
  controls share `corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock`; do
  not launch duplicate GPU work outside that lock.  Session
  `019fc2bf-3563-7c13-83eb-d11fea473e11` should consume these versioned
  artifacts rather than rerun them.
- The external-mount audit records
  `ordered_selected_dicom_sha256=fa209c...c39`.  The two CPU-only residual
  screens are active; detached `vindr-v2-confirmation-collections-v3` waits for
  both, integrity-freezes one direction-specific spec before confirmation is
  visible, then runs both model-family canaries and the two 1,920-case
  confirmation collections serially.  The v3 collector checkpoints every 64
  cases and resumes only under an exact hash-bound contract.  Detached
  `vindr-v2-confirmation-analysis-v1` then automatically runs confirmation-only
  clustered bootstrap analysis, finding-wise heterogeneity, confidence/entropy/
  random-direction controls, two-model boundary synthesis, and executable
  acceptance tests.  Observational Early erasure can only request causal
  patching; it cannot authorize a decoder.  Do not launch duplicate screen,
  confirmation, or analysis jobs.
- Session `019fc2bf-3563-7c13-83eb-d11fea473e11` has started the independent
  paired DICOM-render Huatuo pilot as resumable PPID1 job
  `vindr-dicom-render-huatuo-pilot-v2`, followed by analysis v2.
  It correctly holds the same `gpu0-vindr-v2.lock`, is now in the watchdog
  manifest, and therefore cannot overlap the queued confirmation collectors.
  Its detached analysis successor is also monitored.  Formal analysis now
  refuses partial, foreign-fingerprint, extra, missing, or error shards and
  requires the exact frozen 160-claim selection plus terminal run-state proof.
  This is explicitly an exploratory `pilot` progression gate: even a pass
  cannot support a paper claim or a source-domain-center claim; it can only
  trigger a separately frozen cross-model/held-out replication.
- The DICOM-render pilot and strict analysis are now complete: 160/160 shards,
  zero errors, and 0/4 findings pass the frozen progression gate.  Descriptive
  per-case render orbits and 2.5%--17.5% flip rates confirm heterogeneous style
  sensitivity, but every half-A-selected transform has a half-B effect CI that
  crosses zero.  This rejects a common source/display-center direction; no Hulu
  replication or render-based mitigation is authorized.  See
  `docs/DICOM_RENDER_PILOT_DECISION_V1.md`.
- Cross-session recovery note (19:05 UTC): an older active-job manifest briefly
  removed the Hulu residual and formal confirmation chains and left their
  supervisors as zombies.  No confirmation artifact existed and no data were
  lost.  The Hulu CPU screen is resumed as v3; confirmation collection v3 and
  confirmation analysis v1 remain the canonical successors.  Do not remove
  these three names because the independent DICOM pilot failed.

## Qualification and boundary implementation (2026-08-02)

- The default paper path is now the preregistered VinDr mechanism boundary,
  not a mitigation method. `vindr_layer_boundary_prereg_v1.json` and
  `classify_layer_boundary.py` distinguish Early erasure, Late emergence,
  Layer-stable, Not-decodable, and Indeterminate per model/finding/polarity
  direction. Formal projection authorization now additionally requires the
  global two-primary-model Early-erasure majority gate; CBD remains forbidden.
- CE-G now uses only the leading explicit Yes/No as its primary decision.
  Later negation or `present` tokens cannot reverse it and are recorded as an
  inconsistency diagnostic. Version 6 applies the same explicit-leading-label
  rule to explanatory dataset references; it never infers truth from later
  clinical words. A non-destructive scan admitted 174 historical
  files (38,927 binary rows): 1,664 legacy ambiguous parses, 964 disagreements
  with a valid leading decision, and 383 RULE-normalizer flips. Forty-eight
  files are `rescore_only`; 126 require regeneration.
- The append-only artifact registry currently contains 181 hash-bound events.
  Historical metrics remain untouched. The old Huatuo 64-token full and
  256-token smoke outputs are `identity_only`; the new 512-token 200-case OE
  run is `admissible` with 0% cap-hit, 100% non-empty, and 100% terminal
  completeness. Its lexical scores remain auxiliary rather than clinical
  hallucination truth.
- LLaVA common-backend divergence was caused by random one-pixel square
  padding, generated-token BOS/EOS accounting, and a mismatched numerical
  runtime. After deterministic preprocessing and canonical runtime binding,
  conformance passes 32/32 and 128/128 with 100% normalized-text, token-F1,
  and generated-token-ID identity. LLaVA remains the conditional third model.
- T0 now audits 18 decoding/RAG methods. Ten are locally admissible. DoLa,
  VTI, and SECOND have official code but no explicit root-project license;
  M3ID lacks official code; RULE/MMed-RAG/FactMM-RAG lack a complete local
  paper-native checkpoint path; MR-RAG has no verified official local release.
  These stay `not_admissible` rather than being represented by local ports.
- A shared common-protocol RAG corpus and deterministic top-3 query contract
  are frozen separately for IU-Xray and MIMIC. Leakage audit found no
  image/study/patient overlap but found exact target-report duplicates; the
  deletion-only decontamination removed 185 IU-Xray documents and 8 MIMIC
  documents. Both resulting indices now pass zero-overlap audits. Matched
  no-context/RAG prompts are additionally routed by information requirement:
  invalid references, treatment/history/etiology claims, and single-image-
  unobservable temporal comparisons are retained but excluded from the visual
  CE arm. The corrected visual-only v3 ladder uses 128 tokens and evaluator v6;
  both Huatuo IU-Xray T2 arms have passed and its T3 run is active under a
  detached supervisor. Earlier v1/v2 artifacts remain failed diagnostics.
- PhysioNet authentication is resolved without storing a credential. The
  official annotation CSV hash is frozen; all 15,000 images have exactly three
  independent reader rows, and eight findings meet all four 100-case vote-bin
  gates. The selective 2,598-DICOM download is alive in persistent tmux with a
  100 GiB reserve. Post-download admission now parses every DICOM header and
  hashes the exact ordered file set before triplet/tetrad construction; file
  count alone is no longer accepted.
- Formal probe v8 no longer depends on the unavailable `sklearn` package and
  adds a per-token norm-matched visual-null trajectory. This control is
  mandatory for the preregistered Not-decodable boundary. Reader-gate v7 now
  exports separate 0/3↔1/3 and 3/3↔2/3 records, selects layers and the strongest
  of seven controls on dev only, and evaluates a polarity-preserving causal
  patch on locked test. Persistent session `vindr-mechanism-boundary` waits for
  the DICOM audit and RAG GPU release, then runs Huatuo and Hulu serially.
- The complete repository suite now passes 202 tests in 6.68 seconds under
  single-threaded BLAS; compileall, JSON validation, shell syntax checks, and
  diff whitespace checks pass.

## Research decision (2026-07-31)

The original late image-independent commitment-bias hypothesis and
one-dimensional Commitment-Bounded Decoding remain rejected.  Broad Clinical
Selectivity is now an audit metric rather than a novelty claim because PAIR-
VLA, CF-VLM, and selective-edit consistency cover the generic invariance-plus-
sensitivity principle. The frozen live question is narrower: **image
sensitivity is not clinically directed grounding**. Directional Clinical
Response is therefore an admission test: claim polarity must first move along
independent reader support in the correct direction, beyond same-support
nuisance drift. Only DCR-eligible models may enter the formal VinDr test of
whether reader ambiguity is conditionally decodable early but lost when visual
support becomes linguistic commitment. Generic “uncertainty,” “reader
disagreement,” and “image reliance” are not novelty claims; 2026 work already
occupies each separately. The proposed delta requires their conjunction:
independent reader support, signed clinical response, and a layerwise
support-to-language transition. OE evaluation must additionally retain content
polarity when language is hedged; otherwise “possible effusion” becomes a
polarity-free third class and a commitment-only rewrite can falsely appear to
remove fabricated content. See `docs/CLINICAL_SELECTIVITY_RESEARCH_AUDIT.md`
and `docs/UNCERTAINTY_REFERENT_RESEARCH_AUDIT.md`.

## Live execution (2026-08-01)

- Reader-Grounded Two-Plane Decoding is now fail-closed in code. Directional
  admission freezes its layer on dev and requires locked-test reader-support
  ordering plus signed response beyond same-support drift, both globally and
  in a strict majority of four-bin-qualified findings. Formal RCCP additionally
  requires one model-bound authorization artifact proving that DCR,
  observational tetrad erasure, and the reader-adjusted clarity gate all pass;
  cross-model artifact mixing and unauthorised formal projection are rejected.
  This authorizes method evaluation only, not efficacy or a paper claim. The
  focused mechanism/transport suite passes 76 tests; formal execution still awaits the
  interactive VinDr annotation download.
- The competing presupposition direction now has a fail-closed bidirectional
  screen: a fixed, independently adjudicated claim universe; within-image
  neutral/existential/negative-obligation prompts; strict token-length matching;
  and false-positive plus false-negative amplification in at least two models.
  The current one-model Hulu null-image length/entropy observation is therefore
  motivation only, not positive evidence. Evidence-Source Erasure is halted
  because the available regex audit lacks controlled source-identity labels.
- All long jobs run under detached PPID-1 supervisors plus a 30-second recovery
  watchdog. Reconnect with `cd /home/dbw/ANCHOR && python scripts/research_status.py`;
  VS Code/SSH disconnection does not terminate server-side inference.
- The first VQA-RAD mitigation matrix is structurally complete at 9 x 200, but
  scientifically invalid: 97.5--99.0% of outputs from the shared MedHEval port
  are bare function-word fragments. Its summary has
  `common_plumbing_valid=false` and no comparable methods. New smoke rejects
  this failure, and method backends must pass canonical greedy identity
  conformance before inference.
- Hulu true-mismatch audit v4 completed 192/192 and passed its non-collapse and
  image-identity-dependency gates. Its preregistered lead-only analysis found
  large content change (0.661) but small lexical commitment change (0.051),
  with dissociation CI [0.575, 0.650]. Because uncertainty markers sit at a
  0.025 floor, this is a mechanism lead—not clinical grounding or claim truth.
- The shared LLaVA mitigation-port collapse was localized to its keyword
  stopper. With stopping enabled it emitted `The` on 4/4 diagnostic cases;
  with stopping disabled it matched canonical greedy exactly on 4/4. Old v3
  outputs remain invalid. The corrected v4 matrix disables that stopper and
  must still pass a 32-case canonical-greedy identity gate before full runs.
- Canonical native LLaVA-Med and native Hulu have completed all 200 frozen OE
  questions. Their lexical token-F1 proxies are 0.107 and 0.132 respectively;
  these runs are structurally valid, but the numbers are not clinical
  hallucination judgments and some outputs visibly overcommit.
- SECOND is now a recorded blocked baseline under the current official Mistral
  checkpoint/code path, not a result. Its standard-generation fork failed the
  common-backend identity gate (normalized exact 0.906, token-F1 0.967), and
  method-native recursion then failed before its first token because
  `CLIPVisionTower` provides no `image_attentions` required by `get_heatmap`.
  Deeper invasive adaptation would change the implementation being audited.
- The active serialized GPU chain is therefore corrected LLaVA mitigation v4
  smoke/identity/full, followed by native Huatuo OE and unified evaluation.
  The v4 smoke has now executed all nine methods without fragmentation, but
  full inference was correctly blocked: greedy reached only 0.938 normalized
  identity and 0.972 token-F1 against the frozen native backend, below the
  0.95/0.98 gate. The two differences were clinically semantic, not formatting
  noise. Failed SECOND states remain frozen for audit and do not block Huatuo.
- Native Huatuo passed its 32-case OE structural qualification and completed
  200/200. Its lexical token-F1 is 0.050 (image-cluster CI 0.042--0.059),
  median answer length is 47 tokens, and 82.5% of answers hit the 64-token
  budget. LLaVA/Hulu/Huatuo now all have uniform v2 lexical, length, reference-
  phrase-coverage, punctuation, and truncation diagnostics. These are output
  confounders, not clinical hallucination scores.
- A frozen post-hoc brevity-control curve evaluates first-sentence and 8--64
  word prefixes. The apparent large F1 gains for Hulu/Huatuo at short prefixes
  lose lexical reference coverage. At the point-estimate coverage boundary,
  Hulu gains only 0.0003 F1 (CI crosses zero); Huatuo gains 0.0099 while its
  coverage-delta CI still permits a 2.6-point loss. LLaVA first-sentence keeps
  lexical coverage and gains 0.0038 F1 (CI 0.00035--0.00791), establishing a
  simple shortening baseline that future claim-level methods must beat.
- A physician-review bundle is frozen at 100 image-disjoint VQA-RAD groups and
  three blinded native-model answers per group (300 answer units). The public
  bundle contains no model identities; a separately hashed private mapping
  enables aggregation. All 100 image files were content-hash verified. The
  rubric separates visual, knowledge, and unobservable claims and records
  visual support independently from language commitment; the benchmark answer
  is explicitly context rather than sole truth.
- The current project suite passes 186 tests under single-threaded BLAS.
  PhysioNet authentication is now resolved without storing a credential. The
  official VinDr annotation CSV was downloaded and hash-verified; 8 findings
  satisfy the frozen four-bin minimum, yielding 3,200 balanced claim rows and
  2,598 unique images with an image-disjoint 652/1,946 dev/test split. A
  selective 2,598-DICOM download is alive in detached tmux session
  `vindr-selective-download` with the 100 GiB reserve guard. Detached successor
  `vindr-post-download` builds DCR triplets and commitment tetrads only after an
  exact 2,598/2,598 file-count gate; incomplete downloads fail closed.

## Working set

- `anchor/corrected_sgta/clinical_claims.py`
- `anchor/corrected_sgta/apply_reader_calibrated_projection.py`
- `anchor/corrected_sgta/authorize_reader_grounded_projection.py`
- `anchor/corrected_sgta/prepare_clinical_selectivity_smoke.py`
- `anchor/corrected_sgta/screen_clinical_presupposition.py`
- `anchor/corrected_sgta/analyze_clinical_selectivity.py`
- `anchor/corrected_sgta/fit_selectivity_calibrator.py`
- `anchor/corrected_sgta/fit_clinical_response_aligner.py`
- `anchor/corrected_sgta/fit_reader_agreement_gate.py`
- `anchor/corrected_sgta/prepare_vindr_reader_manifest.py`
- `anchor/corrected_sgta/prepare_vindr_selectivity_triplets.py`
- `anchor/corrected_sgta/prepare_vindr_commitment_tetrads.py`
- `anchor/corrected_sgta/analyze_commitment_tetrads.py`
- `anchor/corrected_sgta/run_huatuo_vindr_commitment_probe.py`
- `anchor/corrected_sgta/run_hulu_vindr_commitment_probe.py`
- `anchor/corrected_sgta/run_llava_vindr_commitment_probe.py`
- `anchor/corrected_sgta/audit_llava_med_loader.py`
- `anchor/corrected_sgta/evaluate_oe_claim_coverage.py`
- `anchor/corrected_sgta/radgraph_claims.py`
- `configs/missing_third_state_vindr_ontology.json`
- `scripts/download_vindr_subset.sh`
- `tests/test_missing_third_state.py`
- `docs/MISSING_THIRD_STATE_PROTOCOL.md`

## Latest evidence

- A prior-titration screen tested whether medical VLM claim odds decompose
  into a stated background prior plus image evidence. The frozen protocol
  crossed 10/50/90% priors with 32 balanced SLAKE finding-image pairs per
  model and used patient-level bootstrap within finding x polarity strata.
  Hulu retained strong clinical image contrast but showed a real-image prior
  reversal and significant prior x image interaction; Huatuo followed the
  prior but lacked significant clinical contrast; LLaVA-Med showed neither.
  A worst-prior robustness score then failed its matched-coverage gate in all
  three models. This rejects both a universal evidence-update law and its
  obvious low-cost mitigation. Reproducible artifacts are under
  `corrected_runs/prior_titration/`; the implementation is
  `run_slake_prior_titration_probe.py` plus
  `analyze_prior_robustness.py`.
- Fixed claim involution was also pruned. It left Huatuo AUROC unchanged and
  produced an inconsistent finding-level change in LLaVA-Med without
  improving sign accuracy. This rules out Yes/No wording bias as a sufficient
  explanation for the observed medical claim errors.

- A real-image Grade-C clarity proxy was run before spending another model.
  RadGraph-XL supplied 31 definite-positive and 31 uncertain-positive claims
  across five chest findings, with image-disjoint dev/test and polarity held
  positive. Huatuo completed 62/62 four-layer trajectories, but the dev-selected
  layer 21 Claim Plane did not add held-out clarity information beyond same-layer
  absolute polarity: AUROC gain -0.061, image-bootstrap 95% CI
  [-0.471, 0.311]. It was also worse than the final Claim Plane by -0.209 AUROC,
  CI [-0.455, -0.015]. No finding had the predeclared 10 examples per class in
  test, so none enters the finding-majority gate. This rejects single-report
  linguistic hedges as a substitute for reader disagreement; it does not test
  the formal multi-reader mechanism. Artifacts are under
  `corrected_runs/commitment_proxy_chexpert_n200_v1/`.
- `fit_reader_agreement_gate.py` is now v6. Besides aggregate locked-test
  statistics, it reports image-bootstrap increments per finding, admits only
  findings with at least 10 examples in each of the 0/3, 1/3, 2/3, and 3/3
  test bins, rejects test reader/finding nuisance levels unseen on dev, and
  requires a strict majority to pass. It stores dev-only feature means/stds,
  model weights, input hashes, and locked-test probabilities so the clarity
  calibrator is reproducible. Non-VinDr or non-reader-vote references can no
  longer receive `measurement_authorized=true`.
  Formal authorization also requires the raw-dev-selected layer to improve a
  separately fitted reader-adjusted continuous-clarity target in held-out
  Brier score; that sensitivity branch cannot reselect the layer.
- `fit_reader_adjusted_support.py` adds a sensitivity-only penalized
  reader/finding/item logistic model. Raw vote bins remain primary. Reader and
  finding effects are fit on dev; test latent support uses only its three
  reference votes with those effects frozen. Synthetic liberal/conservative
  panel tests recover the intended reader-severity ordering. The annotation
  download phase now generates both raw and adjusted manifests without
  overwriting official votes.
- The closest-work audit now treats human-distribution VQA calibration,
  medical verbal-confidence training, and VLI instance-specific bi-causal
  steering as occupied. A generic reader-calibrated confidence score or
  dynamic activation edit is not a contribution. The remaining delta is the
  causal conversion from external reader-distributed claim support to language
  commitment, plus matched-positive-claim-coverage OE evidence.
- Formal RCCP application now binds provenance to the actual support
  calibrator, clarity gate, dev manifest, and ontology files. The CLI recomputes
  all four SHA256 values and rejects merely hash-shaped placeholders or a row
  copied from different artifacts; only explicit `--plumbing-only` runs may
  omit these files.
- A real Grade-C OE screen exposed and retired raw cross-claim ECCE scoring.
  Huatuo produced at least one positive image-grounded ontology claim in only
  6/16 MIMIC drafts. On those six cases, dev selected no exchange; raw early
  and final scores recovered none of six positive test-reference mentions at
  the draft claim budget. A separate 10-image unlabeled calibration bank
  showed that claim centering removes severe prompt prior (leave-one-image-out
  positive-mention MRR 0.081 to 0.219 at layer 14, paired image-bootstrap delta
  CI [0.016, 0.274]), but centered early and final layers are indistinguishable
  (0.219 versus 0.222), the 3-case dev/test preference reverses, and automatic
  single-report references are not truth. Centering is therefore a control,
  not a method result. The frozen diagnostic artifact is
  `corrected_runs/ecce_grade_c_mimic_actionable_v1/claim_relative_loo_screen.json`.
- The mechanism-to-method bridge now preserves the two evidence axes instead
  of forcing another scalar. Dev-calibrated DCR support \(\pi\) and tetrad
  clarity \(\kappa\) form `(kappa*pi, kappa*(1-pi), 1-kappa)`. A closed-form
  forward-KL commitment projection caps definite decoder mass by clarity and
  clips polarity contradictions to an undetermined boundary without flipping
  claim content. Four regression tests cover composition, commitment capping,
  contradiction handling, and identity on compliant clear claims. This is
  implementation plumbing until formal VinDr establishes both calibrated
  channels and OE gains at matched coverage.
- Collision checking now includes TCLA (2026), which uses labeled few-shot
  class-wise prototypes and closed-form residual logit adaptation for medical
  CLIP models, and CoEV (2026), which verifies already-generated positive
  claims by external region localization and masking before LLM rewriting.
  Consequently class-centering and generic claim verification are explicit
  baselines. The admissible novelty delta remains independent reader support,
  signed within-claim response, the early-to-language commitment transition,
  and evidence-bounded realization without omission exchange.
- A narrower calibration collision check further demotes RCCP's projection
  mathematics from a novelty claim. Diagnostic Uncertainty Calibration (2020)
  already calibrates inter-rater disagreement distributions; 2026 medical-VQA
  work already combines hallucination signals with calibration and separately
  uses a visual-presence × text-integrity factorial loss for verbalized
  confidence. Generic constrained/projection decoding is also crowded in
  current VLM work. RCCP is therefore only the smallest polarity-preserving
  realization operator. The paper-level delta must come from independently
  observed reader support, the causal support-to-commitment transition, and OE
  conclusions at fixed claim coverage—not from KL projection or calibration.
- The formal reader-support experiment now has a training-free identification
  layer before any learned probe. A matched 2×2 commitment tetrad holds finding,
  acquisition stratum, split, and majority polarity fixed: two unanimous and
  two disagreement images compare 0/3↔1/3 or 3/3↔2/3, while within-bin pairs
  estimate nuisance drift. The analyzer selects an early layer on dev only and
  tests early-minus-final majority-directed support AUROC on locked test. A
  synthetic positive/negative-branch regression test recovers a planted
  1.0→0.5 AUROC erasure. This strengthens identification but is not empirical
  VinDr evidence; cross-patient tetrads remain conditional rather than causal.
  Formal aggregation is now per finding and per polarity: each branch requires
  at least 10 test tetrads, both branches must pass, and a pooled macro cannot
  hide failure in the majority of qualified findings.
- The annotation path now intersects VinDr CSV columns with the exact frozen
  ontology before eligibility counting. `No finding`, `Other diseases`, COPD,
  lung tumor, or any other non-ontology column is excluded and recorded with
  the ontology hash in `summary.json`; CE and OE can no longer silently use
  different claim universes.
- The mechanism-to-method gap is now explicit. A commitment-only hedge cannot
  reduce content hallucination under claim contract v8. The only conditional
  content method is Evidence-Conserving Claim Exchange: at fixed positive-claim
  count, replace a weak early-support draft finding one-for-one with a stronger
  omitted ontology finding, then calibrate certainty separately. Unit tests
  verify exact claim-budget preservation, score-margin enforcement, and
  immutability of context claims. This remains plumbing, not a method result,
  until DCR+tetrad pass and OE fabrication falls without more omission.
- A refreshed formal idea-evaluator audit gives the surviving *question and
  decisive experiment* an **Accept with Revisions** verdict, while leaving the
  original CBD method rejected. Scores are Higher 4, Faster 5, Stronger 8,
  Cheaper 8, and Broader 7. The largest risk is collision: causal image-use
  audits cover generic sensitivity and CheXthought covers multi-reader
  disagreement. The paper is novel only if it establishes the three-part
  conjunction above and changes conclusions beyond polarity-only controls.

- A frozen Huatuo Findings-to-Impression substitution probe failed its
  semantic-specific gate. Across 24 MIMIC reports, matched Findings reduced the
  correct-versus-shuffled-image NLL advantage for the Impression from 0.0456
  to 0.0078, but attenuation beyond a length-matched mismatched-Findings prefix
  was 0.0183 with bootstrap CI [-0.0017, 0.0428]. The effect cannot be separated
  from position or target predictability and is pruned rather than scaled up.
- The latest collision audit found that CheXthought already predicts
  human--human disagreement and improves uncertainty communication from
  multi-reader data. A separate 2026 causal audit shows that medical VLMs can
  score well while ignoring the image. Reader disagreement is therefore not a
  novelty claim by itself, and layerwise commitment results are inadmissible
  unless the existing same-state/opposite-state Clinical Selectivity triplets
  first establish correctly directed image use. The repository already has
  this stricter matched, image-disjoint audit; no weaker duplicate pairing was
  added.
- A second frozen screening probe tested whether negative pleural-effusion
  claims depend on anatomical coverage. The lower-lung mask did not pass the
  positive-image manipulation check (target-minus-control `Yes-Maybe`
  attenuation 0.281, CI [-0.0078, 0.6016]), so its negative arm is correctly
  marked uninterpretable. No local lesion boxes are available, and no masks
  were tuned after observing the result.
- “The visual null is not null” was also rejected as a standalone headline.
  A same-image per-image-mean versus locked-global-mean audit had layerwise
  evidence-rank Spearman 0.960--0.984 and only one final CBD-state disagreement
  among 16 claims, while prior work already systematically varies and combines
  VCD contrastive samples. Null choice remains a control; clinically directed
  response beyond same-state drift remains the actual research delta.

- A 2026-07-31 collision and screening audit rejected "visual witness" as a
  novelty claim: CoEV covers assertion-to-region evidence verification and a
  2026 formal-verification paper already checks whether Impression diagnoses
  are entailed by generated Findings. Real RadGraph-XL outputs contained 82
  Huatuo versus 34 reference `suggestive_of` edges, but positive-source target
  uncertainty was essentially unchanged (about 74% versus 75%). Multi-target
  source branching was only 21.5% versus 17.2%, reversed after word-length
  normalization, and inspected branches were usually legitimate differentials.
  Witness bounding, edge-uncertainty collapse, and evidence double-spending are
  therefore controls or pruned leads, not the main method.
- A lesion-box-controlled SLAKE probe tested a sharper proof asymmetry:
  positive findings can be certified by a local witness, whereas definite
  negative findings should require coverage of all relevant anatomy.  The
  frozen 64-case result is architecture-specific rather than general.  Hulu
  passed the positive manipulation (supported-minus-undetermined attenuation
  0.582, image-bootstrap 95% CI [0.195, 0.984]); among 40 half-field views
  whose complete image was correctly negative under the coverage-explicit
  prompt, it nevertheless retained a definite `No` in 62.5%.  Huatuo's
  scaled positive manipulation was 0.262, CI [-0.016, 0.586], reversing the
  apparently favorable n=16 screen, and LLaVA-Med's was -0.016, CI
  [-0.047, 0.015].  Neither supplied an interpretable negative arm.  Thus
  coverage-blind negation is a real Hulu failure mode and a useful evaluation
  principle, but **coverage-certified decoding is rejected as a universal
  mitigation story**.  Artifacts are under
  `corrected_runs/quantifier_coverage/`.
- Collision checking further limits this branch: Budgeted Conformal Evidence
  Acquisition already formalizes answer/abstain/acquire-extra-evidence with
  adaptive crops, and current low-quality CXR work explicitly targets missing
  visual evidence.  Any future coverage claim must explain the cross-model
  boundary and outperform those controls; quantifier rhetoric alone is not a
  contribution.
- The OE contract now explicitly forbids connector-blind atomization. `A or
  B` must remain an alternative set; an extractor that loses OR/AND/negation
  scope cannot define commitment truth. This is an evaluation safeguard, not
  a claimed mitigation result.
- A formal idea-evaluator audit classifies the old mean-token-null scalar CBD
  as Reject and Pivot because the unchanged Huatuo/Hulu baselines already beat
  its core intervention. The separate VinDr reader-disagreement hypothesis is
  untested and remains the only live scientific gate.
- The current project-owned test suite passes 101/101 in the Huatuo environment;
  the VinDr download script also passes shell syntax validation. The official
  annotation URL requires HTTP Basic authentication (unauthenticated request:
  401), so the formal CSV still awaits one interactive local password entry.

- Claim contract v7 now preserves OE `prediction_polarity` independently from
  `prediction_uncertainty`. Hedged-positive claims remain positive content in
  fabrication, grounding, and matched-coverage metrics; emitted uncertain rows
  without polarity are rejected. A regression test proves that a
  polarity-preserving hedge cannot claim content-hallucination reduction. This
  closes a real evaluation loophole but becomes a paper contribution only if
  it changes method rankings or conclusions on formal OE/report data.
- A direct collision audit rejects three tempting headlines: generic internal
  uncertainty readout (Miao and Ungar, 2026), generic visual/reasoning
  confidence decomposition (VL-Calibration, ACL 2026), and dynamic VLM
  activation steering (DMAS, ICLR 2026). ConRad separately occupies calibrated
  sentence/report confidence in radiology. These are now baselines or scope
  boundaries, not claims of novelty.
- The orthogonal Claim-Plane actuator is non-empty and replicated across two
  architectures on grade-C clear cases. Huatuo n=32 preserved 87.5% baseline
  accuracy and zero omission while reducing targeted commitment by -0.627
  (95% cluster bootstrap [-0.658, -0.596]) versus random-control difference
  -0.645 [-0.684, -0.606], with mean absolute polarity change 0.041 and zero
  sign flips. Hulu n=16 preserved 100% accuracy and zero omission, reduced
  commitment by -0.887 [-0.973, -0.805] and beat random by -0.867
  [-0.988, -0.762], with mean absolute polarity change 0.059 and zero sign
  flips. This validates actuation plumbing, not efficacy or reader ambiguity.

- A grade-C paired diagnostic compared same-state patient/image swaps with
  opposite-state clinical swaps for pleural effusion and pulmonary edema.  A
  v3 manifest now fully decodes images before admission and supplies 32
  complete triplets/96 LLaVA inputs; the earlier Huatuo/Hulu runs retain two
  explicit truncated-file errors (30 complete triplets/94 inputs).
- Huatuo's final clinical-selectivity gap was -0.346 with bootstrap 95% CI
  [-0.604, -0.079]: nuisance polarity drift exceeded clinically directed
  change.  Hulu's gap was +0.402 [-0.133, +0.904] with 0.867 opposite-state
  pair accuracy, giving a preliminary cross-model non-work/work boundary.
- LLaVA-Med's loader is now admissible: 391/391 checkpoint CLIP tensors equal
  the separately loaded tower after dtype conversion, all non-vision keys are
  covered, and real image swaps change both projected tokens and final logits.
  Despite this real image sensitivity, LLaVA predicted `supported` for all 96
  inputs.  Its final CSG was -0.081 [-0.103, -0.059], significant on both
  dev and test and negative for both findings.  This directly demonstrates
  that image sensitivity is not clinical grounding.
- Directionality changes the diagnostic conclusion.  Conditional on an
  opposite-state absolute response larger than same-state drift, the response
  still moved in the wrong clinical direction in 35.3% of Huatuo, 4.8% of
  Hulu, and 63.6% of LLaVA-Med triplets.  This is the surviving mechanism
  signal, but remains grade C.
- CSG is not a universal error score.  On the locked grade-C split its error
  AUROC versus ordinary output margin changed 0.571→0.776 (Huatuo),
  0.621→0.909 (Hulu; only three errors), and 0.492→0.453 (LLaVA).  The Hulu
  interval is nominally positive but too sparse; formal VinDr must compare
  CSG to both margin and unsigned selectivity with clustered uncertainty.
- A locked-dev four-weight selectivity calibrator did not consistently beat
  simple controls.  On Huatuo it matched the supervised layer mixer in
  accuracy but had lower AUROC; on Hulu it improved AUROC from 0.906 to 0.913
  and removed omission but did not beat calibrated-final fabrication.  Method
  superiority is therefore unearned.  On LLaVA the selectivity mixer was worse
  than calibrated-final and supervised-layer controls (0.438 accuracy, 0.426
  AUROC), strengthening the decision not to package it as a method.
- A second matched-compute response-aligner probe compared task-only,
  invariance-only, unsigned-response, unsigned-selectivity, directional, and
  signed-relative-margin objectives.  The signed method collapsed toward
  `undetermined` and achieved held-out three-state accuracy 0.000/0.130/0.000
  on Huatuo/Hulu/LLaVA.  It is rejected; zero fabrication from universal
  uncertainty is explicitly counted as failure.
- The surviving formal hypothesis is now support-to-language uncertainty
  erasure, not generic “see versus know” attribution or inverted grounding;
  both have direct 2026 collisions.  A reader-agreement gate has been
  implemented that can only change definite versus `undetermined` wording and
  cannot add, delete, or flip claims. Each layer has a matched polarity-only
  control; the non-final layer is selected on dev only, then test gains use an
  image-cluster bootstrap. The gate requires conditional AUROC/Brier gain and
  at least +0.05 early-versus-final AUROC, so a monotone recalibration or
  threshold change cannot pass. It correctly refuses the current binary smoke
  because no 1/3 or 2/3 reference exists.
- The OE realization primitive is now polarity- and coverage-preserving by
  construction: it can only hedge or unhedge existing image-grounded claims
  from the agreement score. It cannot add, delete, flip, relocalize, or silently
  drop claims with missing scores. The old ontology-adding scalar CBD helper is
  retained only for negative-result reproduction.

- On 32 real balanced CXR claims, Huatuo baseline accuracy was 87.5%; the old
  CBD rule fell to 56.3%, increased positive fabrication from 20.0% to 45.5%,
  and introduced 25.0% positive omission.  A locked dev-global-null test
  reproduced the failure: 87.5% to 43.8%, fabrication 20.0% to 50.0%, and
  omission 0% to 37.5%.
- Hulu-Med-4B independently showed 100% baseline versus 75% old-CBD accuracy,
  with new fabrication and omission.  In both models, final-layer null
  commitment bias was lower rather than higher than the preregistered
  pre-collapse layer.
- Claim-plane geometry is empirically non-degenerate: final-layer commitment
  predicted from absolute polarity had R-squared 0.007 on Huatuo n=32, 0.046
  on the locked Huatuo global-null test, and 0.495 on Hulu; the more diagnostic
  Hulu layer 27 value was 0.026.  The old one-dimensional constraint left
  large residuals in both architectures.
- These labels are report-derived grade C.  Low final margin currently detects
  the two locked Huatuo errors better than low visual commitment, so no new
  decoder claim is allowed before VinDr reader-vote and OE validation.
- The cached LLaVA-Med report mitigation run is invalid: its 694 outputs
  collapsed to the single token `The` and cannot support an OE conclusion.

- Thirty-nine targeted tests for frozen semantics, manifest splitting, causal
  controls, leakage-free layer selection, clustered conditional-feature tests,
  matched coverage, reference provenance, and RadGraph claim conversion pass;
  all 54 repository-owned tests pass. Unbounded repository
  collection is not a valid boundary because vendored projects contain their
  own incompatible tests and dependencies.
- HuatuoGPT-Vision-7B completed one real MIMIC image through real and
  mean-token-null visual embeddings and decoder layers 7/14/21/28. This is a
  plumbing smoke only; `corrected_runs/missing_third_state_hook_smoke_v2/` is
  not mechanism evidence.
- The Huatuo probe now performs a layer-21 exact Claim-Plane commitment
  intervention: the null $\nabla C$ direction is projected orthogonal to
  $\nabla P$, preserving polarity to first order. The equal-step random control
  is orthogonal to both target and polarity; exact norm matching and temperature
  controls remain. A regression test also verifies that the full gradient
  readout explicitly exits outer `inference_mode`; the earlier implementation's
  `enable_grad` alone was insufficient. A new real-image Huatuo v5 plumbing
  smoke completed with zero errors: exact norm error 0, targeted-polarity cosine
  $4.19\times10^{-9}$, random-target cosine $5.59\times10^{-9}$, and random-
  polarity cosine $1.12\times10^{-8}$. Its reference remains grade C and one
  image, so all scientific mechanism gates correctly remain false.
- OE matched-coverage evaluation now keeps the complete claim ontology in every
  method denominator and separately matches positive-or-hedged abnormality
  claims. A real-report protocol smoke correctly invalidated a terse
  uniform-negative output rather than treating it as zero hallucination.
- Formal OE evaluation now separates reference observability from a method's
  prediction and rejects automatic labelers/LLM judges as truth. A provenance-
  complete VinDr-vote schema smoke passed under claim contract v2.
- RadGraph 0.1.18 with modern-radgraph-xl completed a real-report CPU smoke
  after pinning Transformers 4.51.2. It extracted six expected finding claims
  with correct polarity. Three unlinked anatomy entities were exposed in the
  audit instead of being silently attached; this confirms that automatic
  location labels remain inadmissible without structured/expert reference.
- Hulu-Med-4B's previously truncated processor/tokenizer files were completed
  from its public model repository. An isolated official-style environment at
  `/home/dbw/.venvs/hulumed` now runs the shared real/null layerwise analyzer,
  null-gradient activation intervention, orthogonal random control, norm
  matching, and temperature control on one real MIMIC image. The official
  16,384-token visual default OOMed; the auditable smoke fixed a 1,024-token
  cap and produced 1,015 visual tokens. Exact norm restoration and the random-
  target orthogonality audit passed. This remains plumbing evidence only.
- The probes now distinguish per-image visual-detail ablation from a genuinely
  image-independent null. A dev-only, equal-image-weighted global projected
  mean can be calibrated once, hashed, and locked for test; plumbing-only or
  sidecar/hash-mismatched nulls are rejected. A one-image Hulu round trip
  verified this mechanism, but cannot provide scientific evidence.
- LLaVA-Med's earlier unused-vision-key warning has been resolved by an exact
  serialization audit: the official loader delays its frozen CLIP module and
  reloads the same weights separately.  The checkpoint/loader pairing and
  cross-image sensitivity now pass; scientific results remain grade C.
- The VinDr download pipeline now produces the formal CSG triplets after the
  selected DICOMs arrive.  It requires known view position, preserves the
  image-level dev/test split, matches acquisition view exactly, controls
  resolution deterministically, and forbids image reuse across findings.
- The unauthenticated PhysioNet file endpoint returns HTTP 403, as expected for
  credentialed access.

## Blocker

The official VinDr annotation CSV and selected DICOMs require an interactive
PhysioNet password. The username is `dfdu233`; the password must not be sent in
chat or stored. Run `bash scripts/download_vindr_subset.sh annotations` in an
interactive terminal; it stops after annotations and manifest construction. Audit
`manifests/summary.json`, then separately run the `images` and `triplets`
phases. This prevents an unaudited image download.

## Immediate next action

After the download script completes, inspect vote-bin counts before changing
any threshold. Audit the automatically built, view-matched, image-disjoint
triplets, then run model hooks on dev and test. Dev alone selects the candidate
early layer; the locked test evaluates conditional commitment information,
early-versus-final loss, and polarity-preserving causal intervention. Open the
locked test once; do not run OE mitigation or train a VLM adapter unless these
mechanism gates pass in at least two models.
## 2026-08-02 19:16 UTC — fail-closed baseline evidence ladder

- Added `anchor/medeval/audit_method_evidence_ladder.py` and generated
  `corrected_runs/unified_eval/provenance/method_evidence_ladder_v1.json`.
- The audit verifies current artifact and qualification hashes and recognizes
  only exact maintained evidence scopes. Suggestive filenames and historical
  diagnostics cannot promote a baseline.
- Current result: 18 methods; T0 pass 10 / not-admissible 8; only `greedy` and
  `shared_medical_rag` have qualified T3 execution evidence; no mitigation has
  passed the full efficacy gate. Shared RAG fails its preregistered causal
  grounding cutoff (no supported dataset).
- Added canonical greedy/beam/sample controls to the native Hulu/LLaVA and
  Huatuo OE runners. Beam uncertainty now follows returned beam ancestry via
  transition scores instead of indexing the wrong live beam. No GPU baseline
  was launched while the formal VinDr Hulu screen and confirmation queue are
  active.
- Verification: 8 focused tests passed; source compilation passed.

## 2026-08-02 20:00 UTC — Clinical Presupposition generation queued

- Added a generation-only Huatuo probe for Clinical Presupposition
  Amplification without changing the common evaluator.  The three conditions
  are frozen as distinct pragmatic tasks/answer spaces (`neutral`,
  `existential`, `negative_obligation`), not represented as paraphrases; all
  share greedy decoding, a 256-new-token cap, and one concise-sentence response
  form.
- Sampling is outcome-blind: the runner reconstructs all 44,008 claims for
  5,501 exact R8/R9/R10-panel images from the complete VinDr label CSV, applies
  the pre-existing global image-hash split, then selects 200 of 1,072 pilot
  images by image-ID hash without reading vote outcomes.  Each selected image
  binds the same frozen eight-finding multi-reader claim universe.
- Generation is crash-safe at image-condition granularity.  Every shard binds
  runner/model/source/ontology/renderer hashes; strict resume rejects changed
  fingerprints.  The recorded generation IDs are the actual
  `outputs.sequences[0]` IDs from Huatuo's generation-only `inputs_embeds`
  path, not decoded-text retokenization.  Visible-answer token count is kept as
  a separate diagnostic.
- A one-image/three-condition canary must first show exact decoded-text equality
  between direct `model.generate` and standard `bot.inference` under the same
  greedy seed.  Only then does the 200-image/600-generation run proceed.  Raw
  generation records mark clinical claim evaluation as
  `pending_shared_audit`; no regex or LLM defines clinical truth.  Refusal
  phrase matching is surface-only and explicitly non-clinical.
- Detached job `clinical-presupposition-huatuo-generation-exact-v1` shares
  `gpu0-vindr-v2.lock`, survives terminal loss under PPID1, and is registered in
  both active and required recovery manifests with `retry_failed_jobs=false`.
  Output: `corrected_runs/vindr_v2/clinical_presupposition_huatuo_generation_v1/`;
  canary: `corrected_runs/vindr_v2/clinical_presupposition_huatuo_canary_v1/`.
  Five focused tests pass.  Formal mechanism screening remains blocked on the
  shared evaluator's human/multi-reader-audited claim counts and strict
  matched-answer-length gate.

### Completion audit — 2026-08-02 20:20 UTC

- The detached job finished with exit code 0.  All 600 expected image-condition
  shards are present for 200 unique images and three conditions; the error
  directory is empty.  An independent reconstruction verified shard names and
  validators, exact pair coverage, three conditions per image, frozen prompts,
  within-image claim-universe identity, actual sequence-ID counts, cap flags,
  pending-audit provenance, deterministic aggregate equality, and the summary
  contract.  Config, runner, model artifact, labels, ontology, renderer,
  selected-manifest, and aggregate hashes all match.  Direct-versus-standard
  Huatuo conformance passed.
- Exact-sequence/visible-answer mean token counts were 68.495/66.495 for
  neutral, 63.900/61.905 for existential, and 59.360/57.360 for
  negative-obligation.  There was one retained existential cap hit, no empty
  answers, and zero conservative surface-refusal matches.
- The preregistered same-image length guard is the current fail-closed blocker:
  only 10 existential-versus-neutral and 18 negative-obligation-versus-neutral
  pairs pass the visible-answer `<=12 tokens` and `<=10%` rule, below the
  required 50 per contrast.  The generated data remain valid descriptive
  candidates, but cannot authorize the bidirectional mechanism screen.  Do not
  loosen the rule, truncate outputs post hoc, or treat this as a clinical
  result; all claim truth remains `pending_shared_audit`.

## 2026-08-02 23:30 UTC — ASCC controlled reader-interaction gate

- The four-pair CIPCA lexical clue was KILLed as confirmatory evidence before
  any new GPU execution.  It had sign-test `p=0.125`, one repeated template,
  post-treatment pair selection, differential-diagnosis deletion, and parser
  false positives for `no signs of` and `rule out`.  The latter two are now
  regression-tested; free OE is external validity only.
- The surviving mechanism is Ambiguity-Selective Commitment Collapse: with the
  image, observation, diagnosis, response vocabulary, and claim coverage fixed,
  abnormality-focused framing may selectively erase the 1/3 and 2/3 reader-
  disagreement state rather than add a uniform prompt prior.
- A discovery-disjoint controlled census is frozen at
  `corrected_runs/ascc/confirmatory_substrate_v1/`: 768 image-claim rows across
  Lung Opacity→Pneumonia, Infiltration→Pneumonia, and Nodule/Mass→Lung tumor;
  all four reader-vote bins; two matched prompt pairs; 3,072 registered native
  forward jobs.  Local 0↔1 and 2↔3 comparisons are exactly matched on parent
  votes and DICOM aspect.  Fingerprint:
  `039fe7486d583dab03a051ec8bd8de49956fe547398bda9337f70496d97f63e4`.
- The readout is a frozen two-plane decomposition over single contextual tokens:
  `K=0.5*(z_present+z_unlikely)-z_possible` and
  `pi=z_present-z_unlikely`.  The primary DID is
  `0.5*((deltaK_1-deltaK_0)+(deltaK_2-deltaK_3))`; a generic main effect is
  failure, and the polarity interaction must pass the frozen equivalence bound.
- Thirteen focused tests pass.  The Huatuo primary edge is running as detached
  PPID1 job `huatuo-ascc-primary-v1` for 1,552 crash-safe atomic forwards.
  Hidden-state intervention and replication jobs remain gated on the formal
  behavioral analyzer; the shared evaluation system was not modified.

### Blind invalidation and ASCC-v2 — 2026-08-02 23:49 UTC

- ASCC-v1 completed 1,552 engineering forwards, but two outcome-blind red teams
  found fatal construct defects before formal analysis.  No score outcome was
  opened.  The run is sealed as
  `construct_invalidated_before_outcome_inspection` and cannot authorize
  patching or a paper claim.
- ASCC-v2 uses symmetric `absent/uncertain/present`, proper
  `logsumexp(definite)-uncertain` commitment, and a true
  `describe/list × findings/abnormalities` factorial.
- Its exact-panel census retains all 1,105 eligible image-edge rows and all four
  reader bins.  The primary opacity edge has 509 rows and 2,036 forwards.
  Independent images use frozen-stratum bootstrap, not arbitrary pairing.
  Fingerprint:
  `a9191fa3fb6fe5866b754a62419a1a44e032232234c4c52c33dd2d9ada4cecde`.
- Gates include neutral third-state admission, both local support boundaries,
  both speech acts, positive ambiguous-bin shifts, local polarity equivalence
  and a five-fold clear-bin affine/temperature residual.  Passing is only a
  primary-edge screen.
- Sixteen focused tests pass.  Detached PPID1 job
  `huatuo-ascc-factorial-primary-v2` is running; common evaluation is untouched.
## 2026-08-03 — PPI v3.1 CPU passes, natural bridge fails; GPU NO-GO

- Permissions are verified as `danger-full-access` with approval policy
  `never`; model downloads run as detached jobs and survive editor disconnects.
- PPI v3 is **not** authorized for GPU. Simple empty-cue-to-label imprinting
  collides with RaVL, VLM backdoors and controlled medical shortcut work.
- The only retained high-ceiling variant is two-plane provenance binding:
  adaptation may bind a shared source coordinate separately to claim mean and
  evidence precision, causing errors near an independently measured evidence
  boundary. Protocol:
  `docs/PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_1.md`.
- The controlled operator is feasible: 5,500 VinDr images, eight claims, all
  70 balanced sign vectors admissible, two orthogonal fingerprints, exact
  `+/-` complements, and matched `zero` arms. Covariance-aware simulations
  distinguish evidence gating from unconditional trigger and margin artifacts;
  the 80%-power interaction floor is about 0.08 q-logit. Focused tests pass.
- Qwen processor Gate A passes only for a fixed, patch-aligned, unmasked
  visual-token frame with a neutral framed parent/control. Ordinary letterbox
  is fully attended and cannot be called masked padding or metadata.
- The repaired 25-claim PubMedVision source audit leaves only consolidation
  (30/5 train/dev positives) and pleural effusion (34/13). Because 2 claims are
  below the frozen 8-claim natural bridge, PPI is now **GPU-NO-GO** regardless
  of its clean synthetic assignment/power result.
- Any future reopening additionally requires a new independent natural source
  substrate plus a shared causal
  availability--binding--commitment pathway in controlled and natural models,
  a second exact-parent family, and OE gains at fixed claim coverage.
## 2026-08-03 04:30 UTC — restart recovery and autonomous continuation verified

- The persistent `/home/dbw` mount is present and writable; models, repository,
  restricted datasets, and the trace-certified runtime remain at their frozen
  absolute paths. About 230GB is free and GPU 0 is healthy/idle.
- The PPID-1 research watchdog and four fail-closed transition monitors are
  alive after editor/container recovery: physician OE, CECD clinical admission,
  CECD dual-semantics transition, and PCEM ECHO access. Each monitor advances
  the next preregistered stage immediately after stable valid input appears;
  none synthesizes clinician labels, credentials, or authorizations.
- No GPU job was blindly restarted. The current gates wait for real independent
  review returns or an explicitly mounted protected table; rerunning completed
  generation would create duplicate evidence without resolving those gates.
- Frozen `docs/INTERNAL_BASELINE_CONTROL_CONTRACT_20260803.md`. It defines
  development-only tuning, claim-level self-consistency, matched-coverage risk,
  and abstention accounting. Self-consistency and calibrated abstention remain
  intentionally T2-missing until a disjoint development substrate exists.
## 2026-08-10 18:12 UTC — intervention-code correction and RAG state-transport Gate 0

- The earlier patient-aligned intervention-code interpretation is superseded.
  Exact-question RAG-response shuffling is not worse than the paired code:
  delta -0.354pp BAcc on 3,391 complete cases (cluster 95% CI -0.961 to
  +0.263pp), and -0.035pp on the 1,384-row strictly exchangeable subset
  (-1.045 to +0.977pp).  The gain is ordinary question-template
  prior/stacking, not patient-specific evidence.
- Fisher Minimum Intervention Basis is rejected after conditional controls;
  DICOM render-response fingerprinting is rejected after a -1.25pp held-out
  BAcc change with CI crossing zero.  Neither is a paper method.
- A frozen CPU-only Gate 0 supports a narrower observation: on CXR-VisHal,
  plain-to-RAG answer changes align with the other-patient report polarity by
  +13.12pp for Huatuo and +8.26pp for Hulu, both with cluster CIs excluding
  zero.  Knowledge-MIMIC replicates only negative/no-state transplantation,
  not a symmetric general polarity-copy law.  This is not causal evidence.
- A 32-claim, eight-concept canary now contains exact frozen raw-RAG,
  no-context, and state-neutral contexts.  Selection and transformation never
  read target labels; 96/96 reports transformed, 463/463 recognized concepts
  retained, and zero forbidden patient-state patterns remain.  Prompt-length
  and disease-concept priming are preregistered confounds and require matched
  ablations.
- The active candidate is Cross-Patient Evidence Transportability / Patient-
  State Noninterference.  It is killed unless matched positive/negative donor
  swaps show a signed claim effect across two models and the firewall removes
  it without erasing transferable knowledge, increasing FN, shortening output,
  or lowering fixed claim coverage.
- All ten baseline tmux sessions remain live.  The current Hulu MIMIC-CXR beam
  report job is healthy and retains GPU priority; no exploratory GPU job has
  interrupted it.

### Gate 1 input completion — 2026-08-10 18:25 UTC

- Built 108 target-blind matched donor pairs (34 VinDr, 74 CXR) and 432
  present/absent/neutral/random-deletion prompts without GPU execution.
- Query and both donors are patient-disjoint; present/absent report length gap
  is at most 10% (median 4.04%); every pair has all four arms and zero semantic
  state mismatch under the frozen lexicon audit.
- The neutral and random-deletion arms remove identical word counts, while only
  neutral removes the target-state claim.  This is the critical control against
  ordinary shortening or context deletion.
- Unsupported findings were not forced into the matrix.  Emphysema, pleural
  thickening, and other lesion have zero strict pairs; two one-pair, very-low-
  similarity findings are documented as weak and excluded from the fast core.
- No model conclusion exists until the target-blind generator runs.  A balanced
  32-pair fast pilot and a no-ground-truth generation runner are being prepared
  behind the baseline GPU lock.

### Fast causal pilot and exact-token control — 2026-08-10 18:35 UTC

- Frozen a CXR-only 32-pair pilot with four core findings and four arms (128
  prompts).  Selection is SHA256-only and target/model-output blind.  All 32
  pairs pass the post-selection fail-closed quality gate: cosine >=0.10 and
  present/absent length gap <=10%; all source images decode as RGB.
- Built Huatuo- and Hulu-specific state-neutral length controls using their
  actual local tokenizers.  Both match raw-RAG full-prompt token counts exactly
  on 32/32 samples, with no trailing-space padding, state terms, target fields,
  or outside-context drift.  The prior whitespace/trailing-space control is
  deprecated and will not be used for conclusions.
- Cached raw-RAG and no-context answers were extracted 32/32 in exact QID order
  for both models.  Only new neutral arms require generation.  The five-arm
  analysis is frozen and fail-closed before those outputs exist.

### TSD collision and semantic invalidation — 2026-08-10 18:50 UTC

- Transportability Symmetrization Decoding is killed as a paper algorithm.
  Counterfactual Averaging for Fair Predictions (CAFP, 2026) has the same
  binary counterfactual two-query averaging formula; Frame Averaging,
  Reynolds averaging, and logarithmic opinion pools also cover its stated
  mathematical novelty.  TSD may remain only as a mechanism baseline.
- The natural matched-donor pilot is sealed before GPU execution.  Independent
  sentence review found coordination-negation scope errors in the frozen donor
  polarity extractor: some alleged present reports actually say no effusion or
  no pneumothorax.  Running the 160-row pilot would therefore produce precise
  answers to a semantically invalid manipulation.
- Both waiting research tmux sessions were removed before they acquired the
  canonical GPU lock.  No exploratory output was generated and the baseline
  Hulu MIMIC-CXR beam job remained at full utilization.
- A strict same-report twin audit currently admits 15/32 candidates and rejects
  17/32 for multiple target assertions, temporal/laterality/severity language,
  uncertainty, or missing non-target controls.  Coverage will not be expanded
  by weakening these rules.  Direct polarity is being re-audited independently.
- The retained research question is source ownership at the circuit level:
  whether other-patient state creates a polarity-odd information-flow component
  distinct from current-image evidence.  Any intervention must exceed ordinary
  output averaging and the EMNLP 2025 image-guided head-suppression baseline.

### Source-ownership candidate boundary — 2026-08-10 19:05 UTC

- Collision audit kills whole-head suppression/projection: Modular Attribution,
  SPIN, V-ITI, ITI, CAA, LEACE, Owl and TAF already cover its components.
- The conditionally retained novelty is Clinical Source-Ownership Binding
  Failure: polarity survives while the fact that it belongs to another patient
  decays, allowing a true donor fact to bind to the current image claim.
- The identifying experiment is CURRENT/OTHER x present/absent, not polarity
  twins alone.  It must separate general polarity, source identity and their
  binding interaction, then patch ownership in both directions.
- Architecture audit finds direct source-token-to-answer-query attention-edge
  masking feasible in both Huatuo eager Qwen2 and Hulu SDPA Qwen3, with no-op
  hooks bitwise exact in tiny homologous models.  This authorizes a direct-edge
  causal probe only: source information can travel indirectly through question
  tokens, so complete noninterference is not yet implementable or claimed.
- Full natural-report twin audit fails the breadth gate: among core findings,
  strict usable counts are cardiomegaly 11/16, effusion 2/16, opacity 0/16 and
  pneumothorax 0/8.  No broad natural twin manifest was generated.  Controlled
  cross-hospital source-factorial data are being built with VinDr targets and
  MIMIC train-only donors instead of relaxing semantic rules.

### Controlled source factorial ready and queued — 2026-08-10 19:18 UTC

- The hash-blind discovery substrate is complete: 64 unique VinDr DICOMs, four
  findings and eight exact-token-matched arms per image (512 rows).  Its 64-image
  confirmation split remains sealed.  Generation manifests contain no votes,
  labels, answers or targets.
- A separate reader-vote-stratified interaction substrate is complete and image-
  disjoint: 253 VinDr DICOMs and 2,024 rows.  Fifteen of sixteen finding x vote
  cells contain 16 images; pneumothorax 2/3 contains all 13 available images and
  is declared short rather than filled from another bin.
- A DICOM-capable Huatuo/Hulu trinary-margin runner passes 9/9 self-tests and a
  real-image renderer check in both environments.  The target-blind discovery
  analysis and its thresholds were frozen before model outputs existed.
- `source_ownership_discovery_v1` now waits on the same canonical GPU lock as the
  formal baseline queues.  It cannot interrupt the active Hulu report job; at
  queue time that baseline had completed 515/694 answers.
- Novelty was narrowed after retrieving TrustNLP 2026 *Ghost Context*, which
  already covers wrong-context misattributed grounding, source-blind metric
  failure, mask-and-rerun attribution and remediation.  The candidate survives
  only if it establishes a new multimodal patient-owner x clinical-predicate
  binding/decay circuit with bidirectional causal evidence and fixed-coverage
  mitigation; otherwise it is a cosmetic medical extension and is killed.

### Goal execution checkpoint — 2026-08-28 05:xx UTC

- Fresh coverage audit: main 336 cells = 82 completed, 15 pending, 2 running/partial, 237 evidence-backed N/A; auxiliary 40 = 16 completed, 8 pending, 1 running/partial, 15 N/A. No generated-unscored cells remain.
- Persistent Baseline monitors remain alive. The prior follow-up chain completed native/CE/shared-RAG/report/cross-model/trained stages and stopped only at the VHR gate; no Baseline process was stopped for Codex usage.
- Repaired LLaVA/Transformers 4.37 compatibility: verified local CLIP source hash, restored legacy mask helpers, and fixed cached multimodal decoding to slice full embeddings to one new token. Single-sample multimodal smoke passed (RC=0, 2 generated tokens).
- First repaired VHR T1/T2 run completed native/off/VHR generation (32/32 each) and found a bookkeeping-only custom-vs-HF mismatch (custom decoded IDs omit EOS; raw IDs match prefixes), so it correctly did not launch the full queue. Gate script now compares raw IDs when available and a second gate is queued behind the active LLaVA Baseline lock; the dependent full-queue wrapper remains fail-closed.
- CEB B0 remains a reproducible NO-GO for the naïve label-free selected-support score (pair AUROC 0.542–0.646, below the preregistered 0.70 gate); no online mitigation is enabled.

### Goal continuation checkpoint — 2026-08-28 05:57 UTC

- Fresh audit now reports main 82 completed, 8 pending, 3 running/partial, 243 N/A; auxiliary remains 16 completed, 8 pending, 1 running/partial, 15 N/A. The reduction reflects completed/N/A qualification updates, not silent deletion.
- The persistent LLaVA methods queue is actively generating `VCD × visual_mimic_oe` (454/490 at the last audit; output files continue to grow). The shared-RAG recovery wrapper waits for this queue's RC before starting.
- VHR gate2 is alive behind the canonical GPU lock and has not consumed GPU time; its first gate's fail-closed RC=10 remains preserved as evidence until the raw-ID comparison rerun completes.

### Goal continuation checkpoint — 2026-08-28 05:58 UTC

- The active LLaVA `visual_mimic_oe` recovery is healthy: `VCD` has advanced to 459+ rows while `DoLa`, `opera`, `PAI`, `avisc`, and `VISTA` each have 490 rows in the resumable output tree. The worker remains GPU-active; no stale-output deletion or forced restart was performed.
- A fresh coverage audit records 82 completed, 8 pending, 3 running/partial main cells and 16 completed, 8 pending, 1 running/partial auxiliary cells. The three running/partial entries are explicitly retained as partial rather than counted complete.
- Because the first full-queue wrapper correctly exited on gate RC=10, a second dependency wrapper now waits for the gate2 RC and re-evaluates `passed=true` before launching VHR full; it is detached and GPU-idle while waiting.
- The LLaVA VCD recovery is now revalidating all chunks (chunk_0002 active after chunk_0000/0001 rewrite). Historical and current completed chunks consistently carry `failed / empty_predictions`; this is retained as a reproducible qualification failure until the queue finalizes, rather than silently converting the cell to complete or stopping the Baseline worker mid-run.
- At the 06:09 UTC checkpoint, chunk_0002 is the active VCD worker and the GPU lock is held only by this Baseline process; gate2 and the dependent VHR wrapper remain idle/waiting. No duplicate worker was started.

### Goal continuation checkpoint — 2026-08-28 06:20 UTC

- Baseline persistence re-verified: the tmux native queue plus RAG/non-report/report scoring monitors are alive; the LLaVA recovery wrapper remains attached to the same canonical GPU lock.
- `VCD × visual_mimic_oe` is making forward progress in the fresh rerun (chunk_0002 closed at 64/64; chunk_0003 is actively writing with a GPU worker at ~16 GiB). Existing completed chunks remain subject to the empty-prediction quality gate and are not counted complete without final meta/metrics.
- VHR gate2 is still waiting on the Baseline GPU lock, and neither the gate2 RC nor the dependent full-queue RC exists yet. No Baseline process was stopped or duplicated.
- Coverage audit remains 82 completed, 8 pending, 3 running/partial main cells; 16 completed, 8 pending, 1 running/partial auxiliary cells. CEB stays CPU-only after the preregistered B0 NO-GO.

### Goal continuation checkpoint — 2026-08-28 06:23 UTC

- Re-audit confirms all persistent Baseline processes remain alive, including the native tmux queue and three scoring monitors.
- The LLaVA recovery advanced `VCD × visual_mimic_oe` chunk_0003 to 36/64; the GPU worker remains active and output mtime advances. No restart or duplicate worker was introduced.
- Recovery RC files are intentionally absent while the queue is still running; shared-RAG and VHR gate2 wrappers remain correctly blocked on their documented dependencies.
- Coverage is unchanged at main 82/8/3/243 and auxiliary 16/8/1/15 (completed/pending/partial/N/A). Pending cells remain explained by active generation or qualification gates, not abandoned work.

### Goal continuation checkpoint — 2026-08-28 06:27 UTC

- To close the native partial risk, a resumable `baseline_native_recovery_20260828b` tmux queue was started. It verified existing Huatuo report artifacts and is now waiting on `gpu0-vindr-v2.lock` before any generation; no duplicate GPU worker is running.
- The current LLaVA VCD worker remains the sole GPU generation process. Native recovery will resume missing/partial Huatuo cells after the lock is released, then run qualification and lexical scoring.
- This recovery is detached and independent of the Codex session, preserving Baseline continuity if the interactive quota is exhausted.

### Goal continuation checkpoint — 2026-08-28 06:29 UTC

- Native recovery is confirmed alive and has already reached its CPU qualification/scoring pass for existing Huatuo report outputs; it is blocked only at the shared GPU lock for remaining generation.
- LLaVA VCD chunk_0003 advanced to 62/64 with the same active worker; both recovery RC files remain absent because their queues are still running.
- The audit remains conservative (82/8/3/243 main; 16/8/1/15 auxiliary), retaining partial cells until final qualification and score artifacts exist.

### Goal continuation checkpoint — 2026-08-28 06:34 UTC

- LLaVA VCD chunk_0003 reached 64/64 and the resumable worker moved to chunk_0004; all six method trees now contain 490 answer rows, but VCD chunks still carry the reproducible `empty_predictions` quality failure until final qualification resolves them.
- Native recovery is alive behind the shared lock; it has not launched a competing GPU job. The fresh audit moved one cell to evidence-backed N/A (main 82 completed, 8 pending, 2 running/partial, 244 N/A).
- shared-RAG and VHR gate2 remain dependency-waiting; no RC file is synthesized while their upstream queue/gate is active.

### Goal continuation checkpoint — 2026-08-28 06:38 UTC

- VCD chunk_0003 is finalized at 64/64; chunk_0004 has begun and currently contains 3 fresh rows, so the audit conservatively reports it as partial again until the chunk closes.
- The native recovery queue remains alive behind `gpu0-vindr-v2.lock`; it is not consuming GPU time while LLaVA VCD is active.
- No upstream RC has been emitted yet. The shared-RAG and VHR dependency wrappers remain persistent and will react automatically when their prerequisites finish.

### Goal continuation checkpoint — 2026-08-28 08:12 UTC

- VCD chunk_0004 reached 49/64 and remains actively generated; all chunk metadata currently retain the explicit `empty_predictions` failure state, so no false completion is recorded.
- Native recovery queue remains alive behind the shared GPU lock; it will resume Huatuo generation only after LLaVA releases the device.
- shared-RAG and VHR gate2 wrappers remain persistent waiters, with no upstream RC yet.

### Goal continuation checkpoint — 2026-08-28 07:20 UTC

- VCD chunk_0004 reached 38/64 and continues writing under the same GPU worker; no stall or duplicate process detected.
- Native recovery remains alive behind the shared lock, while the shared-RAG and VHR wrappers continue waiting on explicit upstream RC/gate evidence.
- Coverage remains conservatively audited; no partial or pending cell is promoted solely from row counts.

### Goal continuation checkpoint — 2026-08-28 07:31 UTC

- VCD chunk_0004 is still advancing (41/64); the active model process is healthy and remains the only GPU generation worker.
- Native recovery queue remains live and waiting on the same lock; no competing generation was launched.
- No RC or scoring transition has occurred yet; the current audit remains conservative until final chunk quality artifacts are written.

### Goal continuation checkpoint — 2026-08-28 06:45 UTC

- LLaVA VCD chunk_0004 continues forward progress (12/64 at the latest check); its model worker is active, so this is not a stalled queue.
- Native recovery remains serialized behind the shared GPU lock; the lock wait is expected while VCD owns the device.
- No RC/score transition is claimed early; audit remains conservative until each chunk has final metadata, qualification and scoring artifacts.

### Goal continuation checkpoint — 2026-08-28 06:51 UTC

- VCD chunk_0004 continues steadily (16/64) under the same GPU worker; the queue has not stalled or restarted.
- Native recovery remains alive in lock wait, preserving serialization with LLaVA. All upstream RC files are still absent because their prerequisite queues are active.
- No coverage change is claimed this checkpoint; partial/pending states remain evidence-backed and monitored.

### Goal continuation checkpoint — 2026-08-28 07:42 UTC

- VCD chunk_0004 advanced to 43/64; the same GPU worker remains active and healthy.
- Native recovery is still alive in shared-lock wait, preserving Baseline serialization. shared-RAG and VHR gate2 remain persistent dependency waiters.
- No RC or audit promotion is issued before the current generation and quality gates finish.

### Goal continuation checkpoint — 2026-08-28 08:25 UTC

- VCD chunk_0004 advanced to 50/64 with an active GPU worker; the process remains healthy.
- Native recovery remains serialized behind the shared lock, and shared-RAG/VHR gate2 wrappers remain persistent dependency waiters.
- No result was promoted based on partial row counts; quality and scoring evidence remain required.

### Goal continuation checkpoint — 2026-08-28 08:40 UTC

- VCD chunk_0004 reached 52/64 and remains healthy under the same GPU worker.
- Native recovery is still alive in shared-lock wait; shared-RAG and VHR gate2 remain persistent waiters.
- No RC or quality promotion has been issued before complete metadata and scoring evidence.

### Goal continuation checkpoint — 2026-08-28 09:40 UTC

- LLaVA VCD chunk_0004 is finalized at 64/64; the worker has entered chunk_0005. Its answer file still shows the historical full 64 rows while the active process revalidates them, so the audit temporarily omits VCD from `running_or_partial`; this is not treated as final quality completion because chunk metadata remain `failed/empty_predictions`.
- Native recovery remains alive and waiting on the shared GPU lock. The latest audit reports main 82 completed, 8 pending, 2 running/partial, 244 N/A; the two partials are the Huatuo greedy VQA-RAD and Huatuo shared-RAG visual-mimic cells.
- shared-RAG and VHR gate2 wrappers remain persistent dependency waiters, and all RC files are still absent.

### Goal continuation checkpoint — 2026-08-28 10:05 UTC

- VCD chunk_0005 is actively regenerating; the fresh answer file is at 5/64 while its model worker remains GPU-active. This is expected resumable progress, not a stall.
- Native recovery remains alive behind the GPU lock. The conservative audit still reports main 82 completed, 8 pending, 2 running/partial, 244 N/A; no quality promotion is inferred from historical full files.

### Goal continuation checkpoint — 2026-08-28 10:25 UTC

- VCD chunk_0005 is actively regenerating and has written 11/64 fresh rows; the worker remains GPU-active with normal progress.
- Native recovery continues to wait on the shared lock, while shared-RAG and VHR gate2 remain persistent dependency waiters.
- No Baseline result is promoted until the corresponding metadata, qualification and scoring artifacts are complete.

### Goal continuation checkpoint — 2026-08-28 10:55 UTC

- VCD chunk_0005 is actively regenerating and has reached 18/64 fresh rows; the GPU process remains healthy.
- Native recovery remains alive behind the shared lock; shared-RAG and VHR gate2 have not started prematurely.
- No RC, quality or coverage promotion is issued before complete artifacts are available.

### Goal continuation checkpoint — 2026-08-28 11:25 UTC

- VCD chunk_0005 advanced to 21/64 fresh rows; the GPU worker remains active with normal progress.
- Native recovery remains alive behind the shared lock, and shared-RAG/VHR gate2 remain dependency-waiting.
- No result promotion or RC was issued before complete quality/scoring artifacts.

### Goal continuation checkpoint — 2026-08-28 11:10 UTC

- VCD chunk_0005 is actively regenerating and has reached 16/64 fresh rows; the GPU process is healthy.
- Native recovery remains alive behind the shared lock, with no competing GPU worker. shared-RAG and VHR gate2 are still dependency-waiting.
- The audit remains conservative at main 82/8/3/243 and auxiliary 16/8/1/15; no row-count-only promotion is made.

### Goal continuation checkpoint — 2026-08-28 10:45 UTC

- VCD chunk_0005 is progressing (14/64 fresh rows) with an active GPU worker; no stall or duplicate execution detected.
- Native recovery remains alive behind the same lock, and shared-RAG/VHR gate2 remain persistent dependency waiters.
- Coverage and quality states remain conservative until the current chunk and its qualification/scoring artifacts close.

### Goal continuation checkpoint — 2026-08-28 07:55 UTC

- VCD chunk_0004 reached 47/64 and remains GPU-active; generation rate is stable and the process is not hung.
- Native recovery continues waiting on the shared lock, with no GPU contention or duplicate execution.
- shared-RAG and VHR gate2 remain persistent waiters; coverage is not promoted until final quality/scoring artifacts exist.

### Goal continuation checkpoint — 2026-08-28 07:11 UTC

- LLaVA VCD chunk_0004 is still actively regenerating (26/64 at the latest check); its process remains GPU-active and no worker duplication occurred.
- The native recovery queue remains alive in the expected shared-lock wait, so Huatuo partials are queued for serialized continuation rather than abandoned.
- shared-RAG and VHR gate2 wrappers remain persistent and dependency-safe; no upstream RC has been fabricated.

### Goal continuation checkpoint — 2026-08-28 07:02 UTC

- LLaVA VCD chunk_0004 advanced to 22/64 under the same active GPU worker; no restart, duplication, or stale-output deletion occurred.
- Native recovery remains alive in expected lock wait, while shared-RAG and VHR gate2 wrappers remain persistent and dependency-safe.
- Coverage audit is unchanged (main 82/8/3/243; auxiliary 16/8/1/15), so no cell is promoted before final qualification/scoring evidence.

### Goal continuation checkpoint — 2026-08-28 06:42 UTC

- VCD chunk_0004 is actively regenerating (5/64 at the latest check) after chunk_0003 closed at 64/64; the worker remains GPU-active with advancing output.
- The restarted native queue is confirmed alive and blocked in `flock` behind the active LLaVA generation, so Huatuo recovery is serialized rather than competing.
- The latest authoritative audit is conservatively unchanged at 82 completed, 8 pending, 3 partial, 243 N/A (main) and 16 completed, 8 pending, 1 partial, 15 N/A (auxiliary).
## 2026-08-28 — DG 可视化模块与风格迁移对齐审计

- 新增 `anchor/corrected_sgta/visualize_dg_alignment_v1.py`：CPU-only 后处理 dashboard，读取 style phenomenon、FEDD-G raw generations 或 evidence-DG raw JSONL，输出 claim delta vs image-change 散点图、variant flip-rate、逐样本 PSNR/edge/prediction 表和原图缩略图；不训练、不改 Baseline、不建立图像池。
- 已在 `corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/dg_visual_audit/` 完成 128-case smoke；LF variant median PSNR≈19、edge≈0.996，gamma variants median PSNR≈30–33、edge≈0.998，flip rate≈1.56–3.13%。
- 新增 `docs/DG_ALIGNMENT_VISUAL_AUDIT.md`。DG 假设保留，但当前对齐失败定位为：参数语义跨实现不一致（low-frequency window vs full-spectrum alpha）、默认 CT source bank 与 CXR modality 不匹配、PSNR/edge 不能证明临床证据保持、flip 主要可能由低 margin boundary 驱动，以及视觉域对齐未传递到 claim binding。
- 下一步采用四格反事实 A/B/C/D（原始/DG × evidence 保留/削弱），计算 `DG_interaction=(B-A)-(D-C)`；Baseline GPU 释放前仅做 CPU/post-hoc 复核。
## 2026-08-28 — DG paired validation queued behind Baseline

- 已启动持久 tmux `dg_paired_validation_v1`，脚本 `scripts/run_dg_paired_validation_after_baseline_v1.sh` 使用共享 `gpu0-vindr-v2.lock`，不会抢占当前 Baseline GPU。
- 验证直接复用已完成的 Huatuo `visual_mimic_oe/greedy` 490 条 native answers，先取前 64 条做 paired canary；只对同一 image/question 生成 `FEDD-G l=0.01, source_ratio=0.8` 视图，保存原图对应变换图、token ids、NLL 和 native 文本。
- 生成后自动运行 OE qualification；输出目录为 `corrected_runs/dg_paired_validation_v1/huatuo_visual_mimic_n64/`。只有 64-case 质量通过后，才考虑扩展到其它已完成数据集或第二种变换。
## 2026-08-28 — DG paired canary status and visualization refresh

- Paired DG canary remains safely queued behind Baseline lock; no DG model worker has started yet. Baseline currently continues LLaVA AVISC generation.
- Existing 128-case Huatuo style artifact was re-rendered with the new dashboard at `corrected_runs/huatuo_rule_mimic_feddg/stage_n128_reselect_v3_switch/dg_visual_audit/dashboard.html`.
- Existing result remains negative for the current selector: native accuracy `98/128=0.7656`, selected FEDDG style `95/128=0.7422` (`-2.34pp`); style-phenomenon transforms show flip rates `1.56–3.13%`, with flipped samples having median native margin about `0.02–0.05` versus `0.59–0.61` for non-flips.
- The new paired canary will add direct native-vs-transformed OE scoring and saved transformed thumbnails; no result is promoted until generation qualification and paired evaluation finish.
## 2026-08-28 — Paper result figures and DG analysis document generated

- 新增 `anchor/corrected_sgta/make_paper_results_v1.py`，从统一 coverage audit 和 score artifacts 生成论文用 `figure_baseline_coverage.png`、`figure_baseline_primary_score.png` 及 `paper_results.json`；status/score 图均显式保留 N/A、PEND、PART，不将未测试项填零。
- 结果目录：`corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/`；当前快照主矩阵 `82 completed / 8 pending / 2 partial / 244 N/A`，auxiliary `16 / 8 / 1 / 15`。
- 新增 `docs/PAPER_RESULTS_AND_DG_ANALYSIS_20260828.md`，汇总已尝试方法、DG 当前负结果的模块级原因和修正版五段流程。核心 DG 假设保留，当前主要问题定位为参数语义冲突、CT-named source center→CXR mismatch、PSNR/edge 不等于临床证据保持、view selector winner's curse、无 native anchor 的整句重生成和 operating-point 评测漂移。
## 2026-08-28 — LaTeX/PDF 结果交付与 DG CPU failure-mode audit

- Baseline 结果已由 `make_paper_results_v1.py` 生成 LaTeX/PNG，并用 `pdflatex` 成功编译 6 页论文附录 PDF：`corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/paper_results.pdf`；coverage/score 矩阵均将未测试格子显式标为 N/A，未将 pending 或不可评分项填零。
- 新增并运行 `analyze_dg_failure_modes_v1.py`：对现有 128-case style phenomenon 和 FEDDG raw generations 完成 CPU 分析。selected FEDDG accuracy `0.7656→0.7422`，style flips `1.56–3.13%`，flipped native margin `0.02–0.05` vs stable `0.59–0.61`。
- DG 假设保持；当前问题进一步定位到参数语义不一致、CT-named source center→CXR modality/provenance 未对齐、结构 gate 不能证明 evidence 保持、candidate selector/winner's curse、无 native anchor 的整句重生成及 operating-point 评测混淆。
- 已完成 LaTeX 文档 `docs/PAPER_RESULTS_AND_DG_ANALYSIS_20260828.md`，下一步在 Baseline 不受影响前提下验证 CXR-matched center 与四格 `DG_interaction`。
