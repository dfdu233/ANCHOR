# Blackwell / ROC Clinical Dominance Audit v1

## 审计问题

同一模型经不同 hallucination mitigation 方法生成回答后，准确率变化究竟表示
“新增了可区分临床真假的信息”，还是只改变了更愿意回答 Yes、No 或无法解析的
工作点？本审计复用 LLaVA-Med-v1.5 在 CXR-VisHal 上六种方法的完整输出，不重新
生成、不占 GPU。

经典背景是：如果一个分数真正包含更多判别信息，那么改变判定阈值时，它应在一组
假阳性率下持续换得更高真阳性率，即形成更好的 ROC 包络。Blackwell 比较、
Neyman–Pearson ROC 排序和 Le Cam deficiency 都是经典统计理论，这里只把它们用作
审计工具，不主张数学创新。

## 首要结果：连续信息增益不可识别（NO-GO）

六套 `5,587` 题生成都只保存：文本、生成 token ID、长度和停止原因；没有保存
logit、概率、置信度或其他预先定义的连续临床分数。token ID 和回答长度不能事后
冒充置信度。因此：

- 可以审计最终输出形成的 `Yes / No / Invalid` 三值粗粒度通道；
- 不能判断方法是否在模型内部增加了连续临床证据；
- 也不能凭准确率提升宣称 Blackwell dominance。

这项方法信息增益主张严格判为 `NO_GO`。

## 三值输出通道的可识别下界

二元子集共有 `3,669` 题、`469` 个图像簇。将 `Yes / No / Invalid` 看作有限观测，
按每种观测的阳性/阴性似然比排序，得到它能达到的最优随机化 ROC 包络：

| 方法 | 三值通道 AUC | 95% image-bootstrap CI | Yes 输出率 | Invalid 率 |
|---|---:|---:|---:|---:|
| AvisC | 0.5969 | [0.5841, 0.6098] | 79.80% | 9.65% |
| OPERA | 0.5830 | [0.5725, 0.5938] | 87.52% | 9.29% |
| VCD | 0.5706 | [0.5564, 0.5846] | 72.66% | 20.90% |
| PAI | 0.5141 | [0.5079, 0.5206] | 97.14% | 2.04% |
| DoLa | 0.5108 | [0.5052, 0.5170] | 97.49% | 1.77% |
| VISTA | 0.5078 | [0.5027, 0.5134] | 98.09% | 1.14% |

这里的 AUC 不是模型 logit AUC，而只是最终三值回答通道的可判别信息下界。它提示
DoLa、PAI、VISTA 的输出几乎退化为统一 Yes，临床真假区分接近随机；AvisC、OPERA、
VCD 的最终回答类别保留更多区分度。AvisC 与 OPERA 的 ROC 曲线交叉，说明不存在
单一方法在所有临床代价下都更好。

## 相同工作点与经验缺口

脚本对每对方法报告：

1. 在对方实际假阳性率处，两条三值 ROC 包络各自可达到的真阳性率；
2. 101 个假阳性率网格上的最大垂直 ROC 短缺；
3. 图像簇配对 bootstrap 的 AUC 差和短缺区间；
4. 不同漏诊/误报代价下的最优条件风险曲线。

这些量是“相同工作点 placebo”的有限输出近似。输出为二元/三值时，ROC 只有很少
的顶点，所以它不能替代保存连续分数的正式审计；文中的垂直短缺也只是经验近似，
不称为正式 Le Cam deficiency。

## 为什么 strict / official 会反转

完整 `5,587` 题上，strict 排名为
`AvisC > OPERA > VISTA > PAI > DoLa > VCD`，official proxy 排名为
`OPERA > VCD > AvisC > PAI > DoLa > VISTA`，共有 `7/15` 对方法反转。

反转可由 official 对无法严格解析答案的重新映射精确解释：方法的 Invalid 率越高，
通常获得越大的 official 加分；六方法的 Invalid 率与 official-minus-strict 加分的
Spearman 相关为 `0.943`。例如 VCD 的 strict Invalid 率为 `48.06%`（这里包含全部
二元与选择题的严格无效），official 相对 strict 增加 `11.95pp`；VISTA 的相应加分
仅 `2.39pp`。所以 official 排名提升不能单独解释为临床证据增加。

## 可信边界与下一次生成要求

本审计可信地支持：现有输出足以证明方法排名强烈依赖判定/解析工作点，但不足以证明
任一方法增加内部判别信息。下一轮若要回答该问题，必须在生成时同步保存每题的
`Yes / No / Uncertain` 首步 logit 或预注册 claim margin，并在相同图像上运行方法开关
与 placebo；之后才能进行连续 ROC、风险曲线和更接近 Blackwell/deficiency 的比较。

结果：`corrected_runs/daylong_idea_search_v1/blackwell_clinical_dominance_v1.json`

