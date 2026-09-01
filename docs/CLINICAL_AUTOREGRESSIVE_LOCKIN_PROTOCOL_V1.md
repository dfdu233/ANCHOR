# Clinical Autoregressive Lock-in：最小机制验证协议

状态：**F6 CONSTRUCT REJECTED；v4 永久禁止 GPU/论文报告。**  
范围：HuatuoGPT-Vision-7B、VinDr-CXR、`pleural_effusion` 与
`lung_opacity`；不能外推到通用医学 VLM 幻觉。

## 0. F6 否决（后续条款仅作失败协议取证）

v4 把同一个 `embedded_claim` 机械拼到每个 prefix 后。CPU exact-string
审计暴露出：

- `The chest X-ray pleural effusion`；
- `This chest X-ray shows no common abnormalities opacity`；
- `... effusion or pneumothorax. opacity`。

因此 early/late likelihood 同时改变语法、话语边界、命题可预测性与长度，
不能解释为 autoregressive lock-in；smooth length null 无法修复。v4 artifact
已写入 `REJECTED_F6.json`，`gpu_authorized=false`。当前 runtime 在读取模型、
manifest 或写 shard 前无条件退出；任何 adapter 修补或 job 恢复都被禁止。

下面第 1--5 节是被否决设计的完整取证，**不是 executable protocol**。

### 唯一允许讨论的 v5

v5 不再比较手工 prefix 后的固定 continuation，而只允许：

1. 每个 finding 冻结一条 pilot 中已经完整出现的自然 full sequence，并由
   独立 clinical-language reviewer 检查自然性、命题泄漏与位置语义；
2. prefix 只能由 tokenizer 对这一条 full sequence 的真实 token boundaries
   逐步截取得到；manifest 不得保存任何手写 prefix；
3. 在同一自然序列 teacher forcing 下，记录每个真实 token boundary 的
   original / same-support / opposite-support / text-only hidden；
4. 用 Gate 1 在 `probe_fit` 学得的 claim-specific prompt-end direction (d_l)
   测量
   \[
   R_{t,l}=\operatorname{sign}(v)\langle d_l,h_l(x,t)-h_l(x_{opp},t)\rangle
   -|\langle d_l,h_l(x,t)-h_l(x_{same},t)\rangle|;
   \]
5. 明确承认不同 token position 的 context 不同；它只能检验“自然生成轨迹上
   的视觉残差何时消失”，不能再声称同命题 fixed-continuation likelihood
   的变化；
6. random direction、same-support、text-only、token-position、activation norm
   与第二条自然 control sequence 必须在人工 construct review 中预注册；
7. dev 只能冻结 changepoint；fresh confirmation 的 selective patching 才能
   提供 causal lock-in 证据。

CPU validator
`anchor/corrected_sgta/validate_clinical_lockin_stimulus_contract_v1.py`
只核对 exact strings、hash、pilot provenance 与人工签署，不自动判断语法。
当前没有已签署 v5 construct，且 tokenwise runtime 尚未实现，所以状态是
**F6 kill / blocked on scientific construct, not engineering**。即使未来人工
review 通过，validator 也只返回 `construct_admitted_cpu_only`，绝不自动授权
GPU。

## 1. 唯一问题

在每个 claim 内完全相同的 frozen prompt 下，Huatuo 是否在生成首个回答
内容 token **之前**保留 reader-grounded claim polarity，但当已生成的临床
prefix 逐步进入 pilot 中观察到的 claim-specific template 时，图像对下一
个 embedded claim 的因果影响在可复现的 prefix token / decoder layer 上
坍缩？

它不是三个相邻问题：

- exact report 重复已经属于 Template Collapse；
- prompt head ablation 已被 Prompt-Induced Hallucination 占据；
- 用 reference-image logits 做 subtraction 与 Pensieve 直接碰撞。

因此，本 probe 不把报告重复当贡献，不搜索 hallucination heads，也不做
跨图像 contrastive decoding。只有“首回答 token 前 polarity 可解码 +
prefix-conditioned image influence 转折 + fresh-split selective patching”三者
同时成立，才允许使用 **Clinical Autoregressive Lock-in** 这个机制名。

## 2. Pilot 与 dev 的严格边界

600-output pilot 只冻结两个表面候选：

| VinDr claim | pilot embedded surface | claim 前 prefix ladder |
|---|---|---|
| pleural effusion | `The chest X-ray shows a right-sided pleural effusion` | empty → `The chest X-ray` → `... shows` → `... shows a` → `... shows a right-sided` |
| lung opacity | `... However, there is a subtle opacity` | empty → `This chest X-ray` → `... shows no common abnormalities` → 完整首句 → `... However, there is a subtle` |

prompt 也按 pilot attractor 冻结：pleural effusion 使用 `existential`，lung
opacity 使用 `negative_obligation`。这里的“同 prompt”指同一 claim 的
`0/3↔3/3` 原图与 swaps 完全同 prompt，不要求两个不同 speech act 的 claim
跨任务共用一个 prompt。

构念审计确认 opacity 是 **positive embedded claim**，不是前句否定的延伸：
前句在句号结束，随后明确出现 `However, there is a subtle opacity`。pilot
exact surface 分层如下（只是发现集，不是 dev 结果）：

| claim / condition | frozen surface matches | reader bins | exact Top-1 full response count |
|---|---:|---|---:|
| pleural effusion / existential | 124/200 | 0/3:102, 1/3:7, 2/3:7, 3/3:8 | 58 |
| lung opacity / negative obligation | 176/200 | 0/3:134, 1/3:26, 2/3:10, 3/3:6 | 72 |

Top-1 opacity response begins `This chest X-ray shows no common abnormalities
such as consolidation, effusion or pneumothorax. However, there is a subtle
opacity in the right lower lung field...`; its local opacity polarity is
positive existential. 完整 text、SHA-256、condition、prefix 与 vote bins 均写入
canonical metadata，不依赖正则临床 judge。

pilot 不提供临床真值、layer、阈值或 changepoint。正式 manifest 只来自 seed
42 已存在的 global `dev` hash split；confirmation 完全未触碰。canonical
artifact 是：

`corrected_runs/vindr_v2/clinical_autoregressive_lockin_dev_v4/`

- 48 anchor rows；
- 24 个独立 block；
- 96 个 DICOM，任何 DICOM 不跨 block；
- 每个 finding 12 对 `0/3 ↔ 3/3` anchors；
- 每个 anchor 另有一个未作 anchor 的 same-support swap；
- 每个 finding 的前 6 blocks 为 `probe_fit`，后 6 blocks 为 `probe_eval`，
  在读取任何 dev activation 前冻结；
- 每个 claim 的 continuation identity 固定；prefix length 仅由 exact
  contextual tokenizer 记录，绝不跨 speech act 宽化 caliper。

早期 drafts `clinical_autoregressive_lockin_dev_v1` 没有 prompt-end fit/eval
split，`dev_v2` 没有完整 builder source / exact-command provenance，`dev_v3`
错误地把 opacity 放入弱 existential surface 而非 pilot 主导的 positive
`However, there is a subtle opacity` surface；三者协议 ID 均与 runtime 不匹配，
**不可运行、不可报告**。

## 3. Gate 1：真正的 pre-response polarity admission

Gate 1 绝不使用 teacher-forced `present/absent` likelihood。adapter 必须在
该 finding 的 frozen pilot prompt 序列化完成、首个 assistant response content token
尚未被消费时，返回每层 prompt-boundary hidden state：

```python
PromptEndTrace(
    layer_prompt_end_hidden=[layer, hidden_dim],
    prompt_end_position_contract=...,
    first_response_token_consumed=False,
    multimodal_expansion_certified=True,
)
```

每个 finding、每个冻结 decoder quartile 独立进行：

1. 只用 `probe_fit` blocks 的 original 与 unique same-support DICOM；
2. 对 hidden 做 per-vector L2 normalization；
3. 方向固定为 positive centroid minus negative centroid，不调正则或维度；
4. 只在 `probe_eval` blocks 报 AUROC；
5. 以 block 为单位交换 fit labels，冻结 256 个 shuffle controls；
6. eval 上必须同时满足 opposite-support signed change 大于 same-support drift；
7. 另跑 exact serialized text-only prompt-end。它不参与拟合；因为删除 image
   可能改变 position/template，runtime 明文记录差异而不硬要求 token identity。
   同一 claim-specific text-only prompt 的 hidden 必须确定性不随 row 漂移。

Gate 1 通过需至少两个非最终 decoder quartiles 同时满足：macro AUROC
`>=0.70`、每 finding AUROC `>=0.60`、block-bootstrap 95% CI lower `>0.5`、
held-out signed AUROC 乘 fit centroid-direction magnitude 的统计量高于
block-label-shuffle 95th percentile（避免小 block 下把偶然同向的 unit
direction normalization 放大为 AUROC=1）、opposite-minus-same causal
excess CI lower `>0`，且 text-only prompt-end invariant。

`Regarding {claim} ... finding is present/absent` 仍被记录，但正式名称是
`non_attractor_preclaim_template_control`。它只是同 claim 的 teacher-forced
template control，**不能替代 Gate 1，也不能被写成 hidden polarity decoding**。

## 4. Gate 2：prefix × layer lock-in

在解释 hidden transition 前，fresh dev generation 必须先确认被解释的输出
endpoint：actual greedy-256（非重 tokenize）中，每个 finding 至少 25% 的独立
`0/3↔3/3` blocks 在两张图上都生成 pilot-frozen embedded claim surface。
这是 exact surface admission，不是临床正确性指标；full-text collision 只报告
不设门。任一 finding 低于 25%，说明 pilot attractor 未在 dev 复现，直接 kill。

对每个 frozen prefix step，teacher-force pilot-frozen embedded claim，分别在
original、same-support swap、opposite-support swap 与 text-only 下取 exact
contextual gold-token log probabilities。adapter 必须证明：

- prompt、assistant prefix 与 continuation 使用 Huatuo 的真实 chat template；
- prefix/continuation offsets 来自完整 contextual serialization；
- multimodal expansion 后 gold token 顺序不变；
- 每层 logit 使用声明的 final-norm + LM-head 规则；
- final layer 与标准 teacher-forcing logits 完全一致。

主要量为：

\[
E_{s,l}=\operatorname{sign}(v)\,[\ell(x)-\ell(x_{opp})]
- |\ell(x)-\ell(x_{same})|.
\]

主要 early step 固定为每个 finding 的 step 2（PE=`The chest X-ray shows`；
opacity=`This chest X-ray shows no common abnormalities`），late 固定为 step 4
（各自 claim-specific modifier 完成）。两条轨迹只按预注册 phase 在 finding
内计算，不比较 raw prefix/position。不从 dev 选 step 或 layer。Gate 2
至少需要两个已通过 Gate 1 的非最终 quartile 满足：

- 两个 finding 的 early `E` 都为正；
- macro `(E_early-E_late)` block-bootstrap CI lower `>0`；
- relative collapse `>=50%`；
- 两个 finding 的 one-break SSE changepoint 都落在预注册 modifier phase
  step 3 或 4，且方向为下降；
- text-only embedded-claim likelihood 从 step 2 到 4 显著增加；
- 用 common steps 0--2 拟合的平滑 prefix-length trend 不能解释 late drop；
- continuation token identity 在 finding 内固定；用 steps 0--2 的 exact
  contextual prefix-token length 拟合 smooth length null，并要求 modifier late
  residual 为负；跨 claim length 只作诊断，不扩 caliper、不作为门；
- manifest-hash 冻结、配对时不读取 support 的 different-block random
  pairing 不能用于选择 layer/step；primary opposite-support intervention 的
  early effect 与 early-to-late decline 都必须大于该 random control；
- 同一 claim 的 non-attractor teacher-forced template 在该层仍须保留正的
  image causal excess（bootstrap CI lower `>0`），否则只是一般序列长度/层
  退化，不能叫 template-specific lock-in。

任何 Gate 1 失败都直接判 perception/prompt-boundary limited；不得用
teacher-forced template score“补救”。任何 Gate 2 失败都 kill lock-in；不得
修改 50%、step 或 quartile 阈值。

runtime 的 `COMPLETE.json` 只可声明 `analysis_input_complete`，并强制
`scientific_gate_authorized=false`。只有冻结 analyzer 可把两个 gate 变为
scientific GO；行数完整本身从不授权机制结论。

## 5. 下一阶段 selective patching（仅双 Gate GO 后）

dev 只冻结 changepoint token group 与 layer；所有 causal patch 只在从未读取
的 global confirmation split 运行。每个 block 用相同 prompt、相同 prefix、
同 claim 的 `0/3↔3/3` source/destination：

1. 在 changepoint 前一层取得 source-minus-same-support visual residual；
2. 只 patch destination 的 frozen assistant-prefix boundary state，并恢复原
   activation norm；
3. 正反向 patch 都必须使 target claim margin 沿 reader polarity 选择性移动；
4. same-support patch、random layer、random norm-matched direction、text-only、
   non-target claim、temperature/length 是必要 controls；
5. actual greedy generation 必须改变目标 embedded claim，而回答长度、claim
   数、拒答率、非目标 template 不能系统变化。

若 patch 只提高/降低共同模板概率，或等价于 reference-image score
subtraction，则判失败并停止，不能包装成 mitigation。跨第二个医学 VLM
复现前，最多称为 Huatuo/VinDr model-boundary mechanism。

## 6. 入口与验证

- manifest builder：
  `anchor/corrected_sgta/build_clinical_autoregressive_lockin_manifest_v1.py`
- model-independent runtime / adapter contract：
  `anchor/corrected_sgta/clinical_autoregressive_lockin_probe_v1.py`
- frozen analyzer：
  `anchor/corrected_sgta/analyze_clinical_autoregressive_lockin_v1.py`
- CPU fake tests：
  `tests/test_clinical_autoregressive_lockin_probe_v1.py`

本阶段明确禁止启动 GPU。production Huatuo adapter 完成 exact serialization
conformance 后，才可申请 dev runtime；在此之前不能产生方向性结果。
