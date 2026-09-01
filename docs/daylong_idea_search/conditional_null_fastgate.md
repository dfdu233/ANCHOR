# Anatomy-Conditional Randomization：零 GPU 致死门

日期：2026-08-12  
结论：**当前 VinDr 构造 NO-GO；不进入 GPU 队列，不影响 baseline。**

## 1. 为什么重新检查 visual null

历史实验中，模型对几种“无证据图”反应很大，但反应没有临床方向性：

| null / counterfactual | 真实结果 | 它实际改变了什么 |
|---|---:|---|
| global-null visual vector | 16 张 dev 图的 visual-token mean，之后对所有图复用 | 抹掉病例、空间和 patch covariance；不是一张可能出现的胸片 |
| VCD Gaussian noise | 正式 baseline 中常改变输出，但医学 CE 可明显下降 | 频谱、纹理和自然胸片流形同时改变 |
| reader-box blur | 128 claims 的 `original - erased = +0.029 [-0.049,+0.109]` | 病灶、局部纹理和 blur 斑块一起改变 |
| horizontal-mirror control | `original - mirror-erased = -0.100 [-0.153,-0.049]` | 左右胸腔并不可交换，且心影/胃泡/marker 非对称 |
| delete-then-relocate | 删除后分数反升；relocation 比原图高 `+0.291` | 位置、拼接与病灶同时改变，明显 overshoot |

所以自然问题不是“怎样做更强的噪声”，而是：**一个 null 是否仍属于给定解剖和采集条件下的可信胸片分布？**

Conditional Randomization Test（CRT）的背景很简单。假设要检验区域 `S` 是否含某 claim 的证据；保留图像其余部分 `X_-S`，从 claim 缺失时的条件分布抽出多个替代区域 `X_S^(b)`。只有在 null 下原区域和替代区域条件可交换时，原图黑盒分数在这些候选中的 rank 才有有效含义。rank 公式和其有效性是经典 CRT / model-X 统计学，不是本项目的新定理。

## 2. 本轮非生成式构造

代码：`anchor/corrected_sgta/build_anatomy_conditional_null_v1.py`

冻结设计：

1. finding 固定为 `Nodule/Mass`，读者固定为 VinDr `R8/R9/R10`；
2. confirmation 取 16 个 `0/3` 和 16 个 `3/3` 图，阳性图必须有 reader bbox；
3. 对每个阳性 bbox 记录标准化坐标；阴性图一对一继承该位置作为 pseudo-ROI；
4. donor 只来自同 split、同 finding 的 `0/3` 图；
5. 用**排除 ROI 后**的 24×24 全胸 context descriptor 找四个最近 donor；
6. 从 donor 的相同标准化位置取 patch，按 recipient 周围组织做边缘均值/方差匹配，再 cosine feather；
7. self patch 也通过同一个 harmonize + feather 操作，避免把“是否调用编辑器”直接等同于 candidate identity；
8. 全程 CPU，无 diffusion、无生成模型、无 VLM forward。

它比 mirror / wrong-patient 随机 patch 更接近目标，因为 donor 是独立读者确认的 claim-negative 图，并且位置和外部胸廓上下文都匹配；但它仍只是近似 imputation。

## 3. 两次 CPU 实验

### 3.1 Confirmation：32 recipients / 288 candidates

产物：`corrected_runs/daylong_idea_search_v1/anatomy_conditional_null_fastgate_v1/result.json`

| 检查 | 结果 | 判定 |
|---|---:|---|
| 最近邻 context distance / 随机 donor | `0.254 / 0.549 = 46.3%` | 匹配确实有效 |
| 0/3 图上 matched replacement vs self 的低级编辑可识别性 | direction-free AUROC `0.505`, image-bootstrap 95% CI `[0.502,0.625]` | 没有发现明显接缝 shortcut |
| ROI 外最大像素变化 | `0` | 操作严格局部 |
| ROI 内平均绝对变化，3/3 / 0/3 | `0.0946 / 0.0887` | 操作生效，但不证明病灶被删除 |
| ViewPosition / orientation / laterality metadata | `0/76` loaded images | **结构门失败** |

### 3.2 Development：16 recipients / 144 candidates

产物：`corrected_runs/daylong_idea_search_v1/anatomy_conditional_null_fastgate_dev_v1/result.json`

| 检查 | 结果 | 判定 |
|---|---:|---|
| 最近邻 context distance / 随机 donor | `0.170 / 0.334 = 51.0%` | 匹配再次有效 |
| 低级编辑可识别性 | `0.523 [0.504,0.676]` | CI 上界超过冻结 `0.65`，未通过 |
| anatomy metadata | `0/28` | **再次失败** |

全量 15,000 DICOM 的既有只读审计进一步确认：`ViewPosition`、`PatientPosition`、`PatientOrientation`、`ImageLaterality`、`Laterality` 和 `ImageOrientationPatient` 均无可用值；见 `corrected_runs/style_domain_revisit_v1/substrate_audit.json`。因此失败不是 76 张抽样偶然。

## 4. 严格解释

本轮得到两个不同层级的结论：

1. **工程层面有弱正信号。** 同位置、context-nearest negative donor 能产生肉眼不显著、简单低级特征难以识别的局部替换，明显好于随机 donor。
2. **统计识别层面仍失败。** VinDr 缺少投照位、患者方向与侧别，且没有独立 anatomy segmentation；因此“像素 context 很近”不能证明“条件在相同解剖状态”。一个未检出的 AP/PA、旋转、体型或肋骨相位差就能破坏 exchangeability。development 的 detectability CI 也未过门。

尤其不能把“分类器没有识别出接缝”写成“已经证明可交换”。这是典型的 absence of evidence；本实验只是尽力证伪 exchangeability，不可能用有限 diagnostics 证明它。

另一个必须修正的设计细节：每图只有 2 或 4 个 donor 时，单图 rank p-value 最小分别为 `1/3` 或 `1/5`，不可能达到 `.05`。小 pilot 只能检验**组级 rank 分布**；正式逐图 `.05` 检验至少需要 19 个 replacement，并且 donor selection 必须完全冻结。

## 5. 直接文献碰撞

| 邻近工作 | 已占据内容 | 本候选剩余空间 |
|---|---|---|
| Candès et al., [Panning for Gold: Model-X Knockoffs](https://arxiv.org/abs/1610.02351), JRSS-B 2018 | conditional randomization / exchangeable rank 的统计基础 | **不能**把 CRT rank 或 plus-one p-value 当数学创新 |
| Pan & Bareinboim, [Counterfactual Image Editing with Disentangled Causal Latent Space](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e1419b1cbca8ada96a87c96567e954-Abstract-Conference.html), NeurIPS 2025 | 因果一致的图像编辑和 disentangled latent counterfactual | 仅做更漂亮的医学 inpainting 不新 |
| Li et al., [Contrastive Learning with Counterfactual Explanations for Radiology Report Generation](https://arxiv.org/abs/2407.14474), 2024 | 相似正负病例与 counterfactual image patch 用于报告训练 | matched positive/negative donor 本身已高度邻近 |
| Cohen et al., [RoentMod](https://www.nature.com/articles/s41746-026-02497-6), npj Digital Medicine 2026 | 经放射科医生复核的合成 CXR counterfactual，用于 shortcut 发现和缓解 | 无医生复核的 patch replacement 证据更弱 |
| [VCD](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_Decoding_CVPR_2024_paper.html), CVPR 2024；[SECOND](https://proceedings.mlr.press/v267/park25c.html), ICML 2025 | evidence-degraded views 与细粒度 mask 的 contrastive decoding | “换一种 null 再做 contrast”不足以构成新方法 |

因此本方向即使数据门通过，也只能把新颖性放在一个跨方法规律上：**hallucination mitigation 的结论是否由 visual-null conditional validity 决定**。单个 anatomy-matched patch 算法不够 ICLR oral。

## 6. 冻结 Go / No-Go 与 GPU 入口

当前决定是 `NO-GO`，`schedule_gpu_32=false`。不允许因为构造看起来漂亮就抢占 baseline GPU。

只有同时出现以下新条件，才允许重开：

- 获得带可靠 AP/PA、patient orientation/laterality 的 DICOM，或冻结的独立肺野/解剖 registration；
- 在至少两个 image-disjoint split 上，matched-vs-self edit detector 的 direction-free AUROC 95% CI 上界 `<0.65`；
- matched context distance 明显优于随机 donor，并报告 overlap，而不是只给均值；
- 至少 19 个 frozen donors/recipient，或明确把 endpoint 限定为组级 rank；
- vote-0 原图 rank 与均匀分布相容，且不存在统一 Yes/No shift；
- vote-3 原图相对 matched null 的极端程度显著高于 vote-0；
- 与 VCD noise、mean-fill、mirror、随机 donor 同时比较，长度、parse 和 working point 全部报告。

如果未来只是换用 diffusion 让接缝更自然，却仍没有 anatomy/projection 条件与独立 clinical manipulation truth，仍然判失败。当前 32 图候选和完整 provenance 已保留，后续可复核，但不进入 VLM 评分。

