# L0 historical falsification matrix for new candidates

目的：复用已经完成的真实模型实验，避免把已关闭机制换名后重新消耗GPU。这里的
`close`只针对表中精确主张，不外推到更窄且尚未测的接口。

| Candidate claim screened | Existing real experiment | Key result | L0 decision |
|---|---|---|---|
| 全局/中间视觉表征中有稳定额外病例信息 | Evidence Addressability，Huatuo/Hulu各532图 | NLL相对改善1.25%/1.45%、Brier 3.42%/3.60%，bootstrap CI均跨0 | close global-summary route；局部patch未关闭 |
| 中间层普遍比最终层保留更多正确证据 | formal admission，7 findings | non-final macro AUROC相对final为-0.109 | close universal early-layer erasure |
| 病灶框擦除能稳定降低对应阳性claim | VinDr focal erasure，128 claims | lesion drop均值0.029，CI[-0.049,0.109]；Nodule/Mass方向错误 | close ordinary bbox-erasure causal law |
| 对侧/镜像区域可充当干净患者内null | 同一focal-erasure实验 | mirror erasure均值-0.100，反而提高claim；relocation overshoot +0.291 | close naive symmetry/mirror null |
| ROI响应比背景响应能可靠定位临床证据 | static VinDr ROI control，79 records | paired AUC 0.545，CI[0.462,0.628] | close mean-dose ROI response as localization proof |
| reader空间一致度自然驱动模型commitment | reader topology，80 claims | 控制vote/面积/finding后Spearman 0.075，p=0.511 | close simple reader-topology mechanism |
| 图像风格漂移是错误主因 | Huatuo RULE 128 | error AUROC: margin 0.798 vs mean style drift 0.425；翻转6次全在低margin组 | close style/DG primary mechanism |
| 多种临床DICOM render形成可利用响应指纹 | 80-case five-render crossfit | canonical BAcc 0.725 vs fingerprint 0.7125，CI[-7.50,+5.01]pp | close cosmetic render ensemble |
| 双中心化能抽取claim-specific evidence | claim common-mode canary，128 | AUROC 0.736→0.655，delta -0.081，CI跨0 | close NCD/ISD/CMEI family as mitigation |
| 患者匹配RAG提供病例特异证据 | exact-question placebo | patient matched BAcc 0.9099，placebo 0.9134；delta -0.35pp | close patient-specific text-code claim |

## Consequence for the daylong search

以上给出超过6个零成本机制淘汰。尚可进入真实新实验的接口被压缩为：

1. 不使用bbox均值响应的**局部稀疏/空间联合统计**；
2. 相对于同一观测后处理的**真实新增观测**；
3. 对完整baseline强度/风险轨迹的**sensitivity–criterion分解**；
4. 不把模型response当证据的外部像素似然或反例图像接口，但必须通过强碰撞门。
