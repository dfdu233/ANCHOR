# 新增复核结果（非最终排行榜）

以下均为 HuatuoGPT-Vision-7B。保留原始快照，不覆盖历史指标；这里只列已完成的新增复核。

## PMC-VQA：同一解析修复后的全量结果

| 方法 | 样本数 | Accuracy ×100 |
|---|---:|---:|
| Greedy | 33,430 | 52.3542 |
| DoLa | 33,430 | 53.8109 |
| ICD | 33,430 | 42.3332 |
| VCD | 33,430 | 43.7212 |
| AvisC | 33,430 | 44.8280 |
| AGLA（复现版） | 33,430 | 49.5154 |

本次修复保留列表选项中会被标签前缀正则误删的非空内容。例如 `A,` 本身可能就是选项内容。逐行隔离比较显示：除 DoLa 不变外，其余五种方法均仅有 `pmcvqa-v2-test-141162` 从正确改为错误；不是重新推理，也没有改动原答案。独立复核文件为各结果目录的 `evaluation_local_review_20260919.json`，DoLa 为 `evaluation_choice_content_20260919_review.json`。本表不替代各方法适配与质量审计。

## PathVQA：MERIT 全量及单条截断补跑

| 方法 | 样本数 | CE Accuracy | OE Token Recall | 综合指标 |
|---|---:|---:|---:|---:|
| Greedy | 6,719 | 66.0321 | 11.2406 | 38.6567 |
| MERIT native-expanded compact-rows | 6,719 | 62.7900 | 8.9611 | 35.8956 |

综合指标按样本加权：3,362 个 CE 样本贡献严格正确率，3,357 个 OE 样本贡献参考答案词级召回率。不是纯 Accuracy。

配对比较：CE 改善324、退步433、不变2605；OE 改善139、退步258、不变2960。下降不能归因于唯一一条截断。词级Recall也可能奖励较长但离题答案中的偶然词重合，不能将全部分差解释为医学能力变化。

原始全量结果无空答、无检测到的复读，有1条1024-token截断。仅对这条按结构异常增加预算至2048，复用原路由与专家缓存；1455 token以原生EOS结束。前1024 token、专家证据及调用数均与原输出一致。其他6718条不变。保留完整输出（包括模型复制的专家JSON），没有裁剪，也没有按正确答案选择补跑。补跑后综合指标不变。

补跑后版本通过空答、复读、实际逐行生成预算检查；**这不代表专家JSON复制问题已解决，也不代表答案均正确**。原始版本及其失败质量记录仍保留。

| 审计对象 | SHA-256 |
|---|---|
| PathVQA manifest | `9d559391099de51105aa6a17f9e54c74442f2d031317dd38091a9b1867ed7fd1` |
| Greedy answers | `0efab48afd391dfd469d7b4e51b033ce3af4826cc9571efaa86984b7f65092bc` |
| MERIT 原始 answers | `bf6516e1cf32ed686c72bfe388b82f630fdb5dc1edd17d9453e310500ff31641` |
| MERIT 补跑后独立 review | `d9a2cff7b33dea2a4c6b757413c04efe5c172049da7358b4ac4139cb08998f43` |

结果位置：`merit-feddg-expert-coverage/runs/huatuo-merit-remaining-full-v2/pathvqa/official_review/` 与 `cap_retry_2048/review/`。这里只发布指标、解释和哈希，不发布原始预测、图像、模型或凭证。
