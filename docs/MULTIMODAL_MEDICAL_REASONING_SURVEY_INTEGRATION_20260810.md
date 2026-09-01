# ZJUMAI Multimodal-Medical-Reasoning 审计后的流程增量

## 判决

该仓库是文献目录，不是实验框架；其中存在多个错链，因此只作检索入口，任何论文、
数据和代码状态必须回到原始来源核验。现有七数据集 baseline 矩阵保持冻结，不在长队列
中途扩表。

## 优先接入

1. **MediConfusion（P0）**：同一问题、相反图像标签的176对样本，直接检测模型是否
   忽视图像；先建立 pair manifest 与 set-accuracy/confusion 指标，待当前矩阵释放 GPU
   后只跑352样本。
2. **MedVH（P0）**：错误图像、无正确选项、错误临床前提、错误答案诱导和报告生成；
   用 paired rescue/harm 评估 Missing Third State，不与自然分布 accuracy 混平均。
3. **Med-R1（P1）**：固定 `VQA_X-Ray` checkpoint，先做32/128例 template、parser、
   base-off qualification；只作为 reasoning-trained medical VLM 边界控制。
4. **PadChest-GR + CURE（P1）**：仅当核心机制需要空间证据时申请数据；优先复用
   CURE 的 loader、grounding metric 与正常/异常区域审计，不立即扩完整训练。

## 暂缓

- MedXpertQA：适合 knowledge/reasoning appendix，但其答案不全是 image-grounded。
- ReXVQA：跨医院价值高，但数据访问和653k QA成本过大，核心机制成立后再抽冻结子集。
- Neural-MedBench：主评分依赖 LLM judge 和医生验证，当前无人审条件下不进入主结果。
- GMAI-VL-R1：当前官方仓库只有 README，无可验证代码、权重和许可证，标记 N/A。

## 对统一流程的约束

- 数据必须声明 `image_grounded / knowledge / unobservable` claim scope。
- 成对 benchmark 必须保持 pair，不拆成独立 accuracy；报告 rescue 与 harm。
- 无医生时，OE/report 自动分数只能称 benchmark proxy。
- 所有新模型仍过 T0→T1→T2→Full；不以综述表中的数字或链接替代官方一致性验证。

## 当前接入状态

- 官方仓库已固定到 `/home/dbw/MediConfusion` commit
  `a3045b03a0a2b5e9f051842e3fd088ed0861f2b6`；176 pair metadata 完整。
- 官方 `scripts/download.py` 在2026-08-10无法取得首个 NCBI OA tarball：论文元数据给出的
  FTP/HTTPS 路径均返回 missing/404。未伪造图片、未使用无许可镜像；adapter 可先基于
  metadata 开发，图像需从 ROCO/NCBI 可用源恢复后才运行。
