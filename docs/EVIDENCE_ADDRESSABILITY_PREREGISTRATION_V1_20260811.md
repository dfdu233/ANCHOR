# Evidence Addressability：增量信息致死实验预注册 v1

日期：2026-08-11

## 研究边界

本实验只回答：在已知最终 claim margin 和 finding identity 后，冻结医学 VLM 的视觉表征是否仍包含与固定三位读者投票对应的病例级增量信息。这里训练的是冻结backbone上的监督线性探针，不称training-free方法。

它不把线性可解码性称为病灶定位、因果证据、可控制性或幻觉缓解；也不把标准投影公式作为理论贡献。若第一门通过，后续定位、因果特异性和真实纠错仍需分别验证。

## 数据与拆分

- 数据：VinBigData/VinDr，本地 `/workspace/vinbigdata`。
- reader panel：R8/R9/R10。
- 初始候选含8项；正式冻结的7项为`aortic_enlargement`、`cardiomegaly`、`lung_opacity`、`nodule_mass`、`pleural_effusion`、`pleural_thickening`、`pulmonary_fibrosis`。
- development-v2：原1,920条confirmation已经在Stage 1打开，现明确降级为development；只使用其中7个finding，每个finding × vote bin各60例，共1,680 claims。
- endpoint-held-out confirmation-v2：从官方完整reader CSV中选择旧100-per-cell direct-CE manifest从未包含的图像；固定R8/R9/R10 panel；7个finding × 4 vote bins × 19例，共532 claims/532张唯一图像。稀有cell优先选择，且每张图只进入一个claim，避免置换时破坏跨claim图像块。
- `other_lesion`因旧manifest外固定panel的3/3图像仅余8张，在看到任何模型结果前按功效约束删除。七个finding为`aortic_enlargement`、`cardiomegaly`、`lung_opacity`、`nodule_mass`、`pleural_effusion`、`pleural_thickening`、`pulmonary_fibrosis`。
- development-v2 与 endpoint-held-out confirmation-v2 image-disjoint；由于公开manifest没有可信patient ID，不称patient-disjoint。
- 该holdout在构造后、任何本端点模型输出读取前冻结；它读取reader labels以完成固定格分层，因此称`model-output-unqueried`，不称严格事前outcome-blind。
- 暴露审计显示532/532图像ID均曾出现在无关历史实验文本中（ASCC 134、specificity-ratchet 36、PPI 532）；因此它只对当前incremental-decoding端点前瞻留出，**不称仓库级image-unseen**。任何强结论仍需外部独立数据确认。
- 预注册文档比holdout构造晚约6秒、但早于模型输出；诚实称“holdout构造后、模型输出前冻结”，不称严格的事前预注册。
- 标签保留完整 reader support `v/3`，不把1/3、2/3强制二值化。
- 风险是人为finding×vote-bin等权的macro-balanced风险，不声称代表VinDr自然患病率。

## Stage 0：数据可行性（已完成）

`/workspace/vinbigdata/train.csv` 含15,000张训练图像、67,914条标注；其中5,501张具有完整R8/R9/R10 panel。八个目标finding中，至少一位读者画框的阳性图像为773–2,953张，三位读者均画框为113–1,714张。因此bbox不是数据阻塞；旧pilot的16例只是既往实验规模。

## Stage 1：缓存的decoder visual-token预筛

目的：不占GPU、不影响baseline，先检查已有post-decoder-block visual-token均值是否在final margin之外携带对齐到病例的增量信息。

### 基础模型

对每个claim使用：

- finding-specific intercept；
- finding-specific calibrated final margin。

final margin定义为最终层 `supported logit - refuted logit`。

### 增强模型

在基础模型上加入某层visual-token均值的低维PCA表示，并允许每个finding具有独立视觉系数。PCA维数、正则强度和候选层只在development的image-grouped五折交叉验证中选择；旧confirmation已经打开并明确降级为development，不作为正式确认集。

### 主指标

- per-reader vote-fraction log loss：三次Bernoulli计数log-loss除以3且去掉组合常数；越低越好。它是投票比例的proper score，但不假设读者iid，也不等同真实病理概率。
- reader-support Brier：预测概率与`v/3`的平方误差；越低越好。
- clear-case macro AUROC：只在0/3与3/3上作为辅助指标。

### 病例对齐负对照

冻结development拟合的模型，先在development拟合`visual block ~ finding + final margin + simple nuisance`，再在endpoint-held-out confirmation内按development冻结的finding×margin strata对整图残差作block permutation。该对照保留模型化的visual–margin–nuisance关系且不按reader vote分层；使用plus-one随机化p值，要求`p <= 0.05`。它明确称为conditional residual-permutation control；p值依赖Ridge条件均值足够且层内残差可交换，不冒充分布无假设的exact CRT。

### Stage 1解释边界

Stage 1只使用decoder内部visual-token状态：

- PASS：允许短暂占用GPU，正式采集raw vision-tower与post-projector表征。
- FAIL：不直接关闭raw视觉接口；只说明已有decoder visual-token缓存没有增量信息，随后仍需一个最小raw/projector确认，因为Huatuo pooled视觉塔已有正控制信号。

## Stage 2：正式raw vision/projector增量信息门

在Huatuo和Hulu完全相同的claim keys上采集：

1. vision tower pooled token；
2. vision tower patch tokens的均值/标准差；
3. post-projector token均值/标准差；
4. 原有decoder visual-token状态与final margin。

所有超参数仅用development-v2；endpoint-held-out confirmation-v2只由独立confirm程序打开一次。首次读取其reader labels前，程序以原子`O_EXCL`写固定位置的开封凭据，锁定分析及import源码、软件版本、holdout/exposure哈希、输出路径、5,000次bootstrap与1,000次置换；只有完全相同契约的崩溃恢复被允许，换路径或参数重开会失败。

### Stage 2冻结的探针族（在采集/打开confirmation前登记）

为避免把某一种高维参数化的失败误写成“视觉塔没有信息”，development只允许在以下两个预先声明的轻量探针中选择，选择标准为image-grouped五折per-reader vote-fraction log loss：

1. `centroid-scalar`：对每个finding，仅用development的0/3与3/3图像计算标准化均值差方向；每张图只增加一个沿该方向的标量分数，再与final margin联合校准。
2. `PCA-interaction`：与Stage 1相同，将低维PCA视觉特征按finding交互后加入final margin；PCA维数与L2强度仍只在development选择。

pre-projector与post-projector是两个预先声明的表示平面；每个平面预先允许`token mean`和`token mean + token std`两种摘要，也只由development选择。confirmation不重新选平面、摘要、方向、PCA维数、正则、finding或阈值。`centroid-scalar`用于覆盖“病例信号集中在一个finding方向、PCA方差排序未保留它”的可能性；它不是确认集上的补救分析。

主比较是嵌套模型：

- 基础：`finding + final margin + simple nuisance`；
- 增强：`finding + final margin + simple nuisance + visual`。

正式PASS必须同时满足：

1. 两个模型的confirmation NLL相对基础模型改善至少5%，在28个固定finding×vote格内分别重采样的paired bootstrap 95% CI下界大于0；
2. 两个模型的Brier相对改善至少5%，95% CI下界大于0；
3. 每个模型至少5/7 findings的NLL方向为正，且公开全部finding结果；这只是supermajority一致性护栏，不作显著性检验（随机正负号下达到5/7的概率仍为29/128）；
4. 条件残差block permutation的plus-one `p <= 0.05`；
5. 两模型的claim keys、reader votes、split与数据哈希完全一致，并执行逻辑AND。

simple nuisance固定为view position、patient position、photometric interpretation、原始行列数、宽高、64×64灰度均值/标准差/5%与95%分位数。增强模型必须在这些变量已经进入基础模型后仍改善NLL/Brier；这里只能写“超出列出的简单统计”，不能写“排除所有域或设备混杂”。各模型的正则强度均由development grouped-CV独立选择。

任一条件失败，关闭当前“raw视觉全局mean/std可被轻量监督探针增量读出”的路线，不事后更换阈值、finding或split；不把冻结backbone上的监督探针误称training-free。

## 后续门（仅Stage 2 PASS后）

1. 定位：高贡献token相对等面积随机框显著富集于reader bbox。
2. 因果特异性：删除/恢复这些token对目标finding的影响显著超过七个off-claims、随机token与等范数方向。
3. 真实纠错：原生FP下降，matched-correct harm不超过1pp；只改善FN时只能称遗漏恢复。

只有四门跨模型成立并自然导出有效方法后，才允许将Evidence Addressability作为ICLR方法主线。几何投影只作为框架表述，不列为理论贡献。
