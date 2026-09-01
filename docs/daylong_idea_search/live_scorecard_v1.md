# 全天探索实时记分板 v1

更新：2026-08-12 17:47 UTC。`GO`只表示通过当前层级，不等于论文成立。

| 候选/实验 | 数据与规模 | 冻结判据 | 结果 | 决定 |
|---|---|---|---|---|
| Convex layer mixture | VinDr clear claims；Huatuo/Hulu | final以上macro AUROC +0.02，两模型 | +0.0174 / +0.0115；Hulu CI跨0；固定阈值阳性率大升、BAcc下降 | NO-GO |
| Criterion-shift mirage | LLaVA CXR-VisHal 5,587题，5方法 | 同一输出在评判规则间排名是否反号 | 10个方法对中5对反号；VCD-VISTA从-4.58pp变+4.98pp | GO，评测原则 |
| Sparse lesion boundary | VinDr 3/3 boxed positives；开发480/模型，确认133/模型 | 病种内面积-margin相关≥0.20且CI>0，两模型复现 | 开发0.232/0.475；确认0.323/0.415；四次p≤0.0006 | GO，自然现象 |
| Lesion area预测miss | 同上开发420、确认133/模型 | finding以上AUROC +0.05且AUROC/NLL CI>0 | Huatuo +0.083通过；Hulu +0.015、AUROC CI跨0 | Joint NO-GO |
| Sparse spatial scan | 7 findings clear dev/test，Huatuo n=840/266 | 超`final+mean+max+top5`至少0.02且CI>0 | AUROC 0.7376→0.7416，+0.0040，CI[-0.0194,0.0273]；仅4/7正向 | NO-GO；不扩Hulu |
| Raw search inflation | 七项全部0/3；Huatuo dev147、test62 | claim搜索1→7时raw p95增长CI>0 | p95 +1.1368，CI[0.5165,1.3876] | GO，仅内部现象 |
| Evidence-conserving pool | Huatuo clear n=840/266；16/64/576分区 | 守恒且超强base +0.02、CI>0 | null均值0.998/0.961/0.956稳定；AUROC仅+0.0016，NLL恶化CI下界0.0026 | NO-GO算法 |
| Selection–Reuse Inflation | 七项全部0/3；Huatuo test62，744次单步评分 | 同claim扩大region数后selected-random margin与FP均显著增长 | 最大搜索selected-random FP +17.74pp CI[3.23,33.87]；但region 16→361 gap增长CI[-.052,.206] | 总门NO-GO；crop先验效应保留新问题 |
| Lesion delete–relocate | Nodule/Mass n128 | 删除下降、搬运恢复，CI>0且各≥60%同向 | 删除效应-0.025[-.128,.069]；搬运恢复+.159[.067,.250]但较原图过冲+.185；joint仅34.4% | NO-GO |
| Real second observation | IU-Xray 64-study Huatuo pilot | 第二真实view超同图/错患者，accuracy或Brier过门 | Acc 90.63→92.19%，+1.56pp CI[-4.69,7.81]；Brier相对+5.92% CI[-2.04,12.60] | NO-GO；不扩Hulu |
| Anatomy-conditional null | VinDr CPU；confirmation32、dev16 recipients | 上下文匹配、编辑不可识别、解剖条件可验证 | 像素距离降约一半，但关键DICOM元数据0可用，dev CI上界0.676>.65 | NO-GO |
| Lesion-area confound audit | 3/3 boxed positives；两模型dev420/test133 | 面积效应控制位置、对比、纹理、碎片与读者一致性后仍存在 | partial rank Huatuo .239 CI[.036,.419]；Hulu .420[.218,.559] | GO，现象更可信但非因果 |
| Blackwell/criterion audit | LLaVA CXR二元3669题/469图，六方法 | 连续临床通道可识别且跨代价比较 | 全部未保存连续score；strict/official 7/15排名反转，invalid与额外得分rho=.943 | NO-GO主线；保留评测约束 |
| Observation-policy pragmatics | VinDr 62个全局0/3阴性 + 62个病种匹配3/3 bbox阳性；Huatuo；5 render×3同像素prompt=1,860评分 | crop特异Gamma>0.25且CI>0；FP降>=10pp；full变化<=3pp；阳性recall损失<=1pp | Gamma=-.117 CI[-.188,-.042]；crop FP降61.3pp但阳性recall也降59.7pp；native外部context模糊使FP增62.9pp，scale仅+3.2pp | **NO-GO**；是全局criterion shift，不扩Hulu；转向negative-context机制 |
| Context completion | 同一124图；true-context vs phase-sham；Huatuo CPU审计 | delta在full+crop+finding强base上AUROC>=+.01、NLL均CI>0 | delta单独AUROC .657；强base .855→.843，增量-.0117 CI[-.0330,.0044]，NLL/Brier变差 | **NO-GO**；响应冗余，不花fresh holdout |
| Witness/certificate logic | OR型局灶claim；partial observation理论审计 | 推出非拒答、非普通fusion且FP降/FN不增方法 | completion pair上任意二元规则FPR+FNR=1；只能补观测、加假设或Unknown | **NO-GO算法**；保留不可辨识边界 |
| Blackwell computation paradox | 同病例full vs context-removed；62全阴性 | crop敏感性来自可验证计算瓶颈并推出新算法 | AUROC .7946→.7980，+0.0035 CI[-.0623,.0730]，但FP+62.9pp；自然修复退化global-local fusion | **NO-GO**；标准bounded rationality解释 |

## 当前论文级判断

- 已有一个更可信自然现象：小病灶在两个医学VLM中获得更弱的正确支持，且控制九个可测位置、像素与标注代理后仍存在；但这仍是关联而非因果。
- 已有一个可信社区警告：同一生成结果可因解析/评分口径不同得出相反方法排序。
- 尚无ICLR Oral级缓解方法。稀疏scan和证据守恒pool均未带来增量判别；BCEA又已占据自适应选区后的完整流水线校准。剩余潜力只取决于“内部raw搜索膨胀是否传到最终FP”的端到端门。
- 端到端搜索税总门已失败：最大搜索有selection effect，但region-count递增效应不显著。该相变主线关闭；更强的新现象是任意局部crop本身把全阴性FP从8.1%推至约63%，正审计其是否属于observation-policy bias，而不是用旧名续命。
- SECOND目前无有效医学效果数字：官方递归接口不兼容且关闭态一致性失败，应报告N/A，
  不能把近似端口的结果写成SECOND负效果。
- “Observation policy is a prompt”已被同像素反事实关闭：来源说明没有提取临床证据，只把正负样本一起推向No。新的可信自然现象是：即便ROI保持native位置和尺度，只模糊ROI外完整解剖，阴性FP也从8.1%升到71.0%；外部正常结构是分布式反证。它尚不是算法，正审计context-completion是否有病例级增量。
