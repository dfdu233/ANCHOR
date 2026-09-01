# C68 — Write-Protected Visual Memory：架构审计与公式级碰撞

> 日期：2026-08-13  
> 范围：HuatuoGPT-Vision-7B、Hulu-Med-4B、LLaVA-Med-v1.5；CPU/源码审计，不占 baseline GPU。  
> 最终裁决：**`FATAL NO-GO AS A NEW MITIGATION PRIMITIVE`**。

## 1. 结论先行

`write-protected visual memory` 的核心直觉是：先把病例视觉状态写入一块只读 memory，后续语言只能读、不能把自己生成的内容写回，从而避免“语言污染视觉证据”。

这个直觉对当前三种医学 VLM **没有产生新操作**。原因不是实验信号弱，而是架构上该性质原本就成立：

1. 三个模型都把视觉 feature 作为若干 decoder 输入位置的 embedding；
2. decoder 使用 causal mask，后面的生成 token 不能影响前面的视觉位置；
3. 启用 KV cache 时，每层视觉 K/V 在 prefill 中计算一次，后续只追加新 token 的 K/V，不覆盖旧位置；
4. 未启用 cache 时，视觉前缀虽被重复计算，但 deterministic eval 下得到相同状态，仍不受未来 token 影响。

因此，“锁住原视觉 K/V、禁止后续文本写入”的实现与 native decoding **函数相同**，不能改变一个 logit，更不能降低幻觉。若为了改变答案而引入新的 K/V、跨层重注入或额外 cross-attention，它就不再是 write protection，而分别退化为 visual feature editing、re-injection、attention routing 或 trained fusion；这些方向已有直接邻近工作。

## 2. 必要背景：视觉 token 在 decoder-only VLM 中是什么

设输入序列的位置依次为 system/text、视觉 token、问题和已生成回答。第 `l` 层第 `i` 个位置的状态可写成

\[
h_i^{(l)}=F_l\!\left(h_1^{(l-1)},\ldots,h_i^{(l-1)}\right).
\]

这条式子表达的只是 causal attention：位置 `i` 最多读取自己和左边的位置，不能读取右边未来位置。若视觉位置为 `v`，任一更晚生成位置为 `t>v`，则

\[
\frac{\partial h_v^{(l)}}{\partial e_t}=0,\qquad \forall l.
\]

含义是：后续文本改变不了视觉位置在任何 decoder 层的 hidden state。这是一个结构性零导数，不是需要训练出来的经验现象。

### 2.1 “视觉 hidden 每层会不会更新？”

会，但只在初始 prefill 中沿**网络深度**更新：

\[
h_v^{(0)}\rightarrow h_v^{(1)}\rightarrow\cdots\rightarrow h_v^{(L)}.
\]

每层视觉状态可以读取位置不晚于 `v` 的 token，所以不同层一般不相同；如果 image placeholder 前有 system/question token，它也可以读取这些更早文本。但它不能读取 image 后面的生成 token。

启用 KV cache 后，这些层级状态导出的

\[
K_v^{(l)}=W_K^{(l)}h_v^{(l-1)},\qquad
V_v^{(l)}=W_V^{(l)}h_v^{(l-1)}
\]

在 prefill 后不会随自回归时间再次更新。生成第 `t` 个 token 时，cache 操作是

\[
K^{(l)}\leftarrow [K^{(l)};K_t^{(l)}],\qquad
V^{(l)}\leftarrow [V^{(l)};V_t^{(l)}],
\]

即追加，不是覆盖。

### 2.2 真正可能衰减的是什么

视觉 memory 没被改写，不代表视觉作用不会变弱。随回答增长，变化的是：

- 当前文本 query 对固定视觉 K/V 的读取权重；
- RoPE/相对位置造成的长距离读取代价；
- 当前生成位置的 residual stream 中，语言上下文对视觉读取结果的压制；
- 不同层对视觉 value 的变换与最终 readout。

所以正确机制名称应是 **visual read-path attenuation / competition**，而不是 visual memory corruption。

## 3. 本地三模型源码证据

| 模型 | 视觉接入 | decoder | cache | 审计结论 |
|---|---|---|---|---|
| HuatuoGPT-Vision-7B | `prepare_inputs_labels_for_multimodal_new` 生成 `inputs_embeds` 后调用标准 Qwen2 causal LM | 28-layer Qwen2 | checkpoint config 为 `use_cache=false` | 每步可重算前缀，但 causal noninterference 保证视觉位置不读未来回答 |
| Hulu-Med-4B | vision encoder/projector 输出直接替换 `image_token_index` 的 embedding | 36-layer Qwen3 | `use_cache=true`，`add_cross_attention=false` | prefill 后视觉 K/V 固定；DynamicCache 仅 `torch.cat` 追加 |
| LLaVA-Med-v1.5 | LLaVA multimodal preparation 后把视觉 embedding 送入 Mistral | 32-layer Mistral | `use_cache=true` | 与 Hulu 相同的 causal-prefix + append-only cache 语义 |

关键源码：

- Hulu 视觉 embedding 替换：[modeling_hulumed_qwen3.py](/home/dbw/models/Hulu-Med-4B/modeling_hulumed_qwen3.py#L373-L389)
- Hulu 调用标准 causal decoder：[modeling_hulumed_qwen3.py](/home/dbw/models/Hulu-Med-4B/modeling_hulumed_qwen3.py#L430-L463)
- Huatuo multimodal embeddings 进入 Qwen2：[llava_qwen2.py](/home/dbw/HuatuoGPT-Vision/llava/model/language_model/llava_qwen2.py#L80-L115)
- Qwen3 创建 causal mask：[modeling_qwen3.py](/home/dbw/.venvs/hulumed/lib/python3.10/site-packages/transformers/models/qwen3/modeling_qwen3.py#L497-L552)
- DynamicCache 仅追加 K/V：[cache_utils.py](/home/dbw/.venvs/hulumed/lib/python3.10/site-packages/transformers/cache_utils.py#L404-L445)

## 4. CPU 数值复核

可复现实验：

```bash
/home/dbw/.venvs/hulumed/bin/python \
  anchor/corrected_sgta/audit_visual_prefix_write_protection_v1.py
```

产物：`corrected_runs/daylong_idea_search_v1/visual_prefix_write_protection_v1.json`。

微型三层 Qwen3 上固定 5-token prefix，再追加 3-token suffix：

| 检查 | 结果 |
|---|---:|
| 各层 prefix hidden 最大差异 | `0, 3.7e-9, 7.5e-9, 4.8e-7` |
| `1e-5` 容差内 invariance | PASS |
| cache 从 5 增长到 | 8 |
| 所有层原 prefix K 最大差异 | `0` |
| 所有层原 prefix V 最大差异 | `0` |

hidden 的亚微小差异来自一次处理 5 个和 8 个位置时浮点 kernel 归约路径不同；cache 原切片逐元素完全不变。

## 5. 一个直接的“无效性定理”

设 native 模型每层已经保存视觉前缀的 `(K_V^{(l)},V_V^{(l)})`。若所谓 write-protected 方法只规定

\[
\widetilde K_V^{(l)}=K_V^{(l)},\qquad
\widetilde V_V^{(l)}=V_V^{(l)},
\]

并禁止未来 token 覆盖它们，但其余权重、mask、位置编码与采样均不变，那么对任一解码步 `t`：

\[
\widetilde p(y_t\mid x,y_{<t})=p(y_t\mid x,y_{<t}).
\]

证明只需按生成步归纳：初始 cache 相同；native cache 本就不覆盖旧视觉切片；同一 query 读取同一 K/V 得到同一 hidden、logit 与 token，因此下一步 cache 仍相同。

这不是论文理论贡献，而是一个致命的等价性检查：**纯 write protection 必然 token-exact 等于 greedy。**

## 6. 2024–2026 公式/机制级碰撞

| 工作 | 它实际修什么 | 与 C68 的关系 |
|---|---|---|
| [M3ID / Multi-Modal Hallucination Control by Visual Information Grounding, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf) | 随生成长度视觉条件对 next-token 的影响下降，以视觉/语言条件分布差异修正 logits | 已把“固定视觉前缀后来读不出来”定位为 read influence dilution，而非 prefix 被改写 |
| [Multimodal Hallucination Snowballing / Residual Visual Decoding, ACL 2024](https://arxiv.org/abs/2407.00569) | 先前生成的幻觉会误导后续答案；用 residual visual distribution 给当前输出直接视觉通路 | 若 C68 改成“绕过语言历史重新读取视觉”，就进入 RVD 的问题与解码接口 |
| [VISTA, ICML 2025](https://arxiv.org/abs/2502.03628) | 视觉信息在生成/深度中逐渐失势；activation steering + early-layer logits 强化视觉信息 | 若 C68 改成跨层重注入或强化视觉 state，直接落入 VISTA 类 visual steering |
| [SPIN, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.631/) | 对每个 query 保留高 image-attention heads，抑制低 image-attention heads | 证明可以改变的是 text query 的读取路径；但这已是 attention/head suppression，不是 memory protection |
| [Interpreting and Editing VLM Representations, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9f14fb9acd243c13c95d4a490d1684ce-Abstract-Conference.html) | 投影内部 image representation，并正交编辑 image features 消除 hallucinated concepts | 若 C68 直接修改视觉 latent，visual representation editing 已有强直接邻居 |
| [ReVisiT, ACL 2026 Oral](https://arxiv.org/abs/2506.09522) | 缓存多层视觉 token 的 vocabulary projection；每步选相关视觉 token，再以 log-space PoE 修正输出 | “保存只读视觉 token 并在每步读取以指导 decoding”已被非常直接地实现；区别只剩 memory 参数化 |
| [DiVE, ACL 2026](https://aclanthology.org/2026.acl-long.1742/) | 动态选 visual-rich layer，分离 intra-layer visual evidence，构造语言先验参考并 contrastive decode | 若 C68 从只读 memory 提取视觉方向再校正 logits，就落入 intra-layer evidence decoupling |
| [Image Tokens Matter, NeurIPS 2025](https://arxiv.org/abs/2505.21547) | 针对 discrete image token 共现先验修改 latent image embeddings | 若 C68 声称视觉 token 自身发生错误 association 并做 latent edit，这个假设/接口也有直接邻居 |

补充：本轮按精确缩写检索，`VTCD` 对应的是 [Understanding Video Transformers via Universal Concept Discovery, CVPR 2024 Highlight](https://arxiv.org/abs/2401.10831)，不是一个 VLM hallucination decoding 方法。不能把未找到的“VTCD hallucination”当作已核验文献。

## 7. 严格 verdict

### 7.1 作为 training-free 方法

**关闭。** 纯 write protection 与 native causal decoding 相同；无需也不应启动 GPU efficacy pilot。

### 7.2 作为 `<1%` 参数训练架构

若另建 specialist fast-weight memory，内容不再等于 native visual K/V，确实可能改变输出；但此时新变量来自 translator / memory codebook / cross-attention read interface，而非“防止文本写入”。需要和同预算 cross-attention、feature concat、Perceiver memory 比较。当前又没有“native visual memory 被写坏”的现象，因此 WPCM 的原机制叙事不成立。

### 7.3 是否还有可保留的开放问题

只剩一个不同的问题：

> 固定视觉 K/V 明明没有被污染，为什么当前生成 query 后来不再有效读取它？

但这个问题已处于 M3ID、VISTA、SPIN、ReVisiT、DiVE 的密集碰撞区；且本项目已排除 attention mask/head suppression、层融合、普通视觉重注入。除非发现一个新的、病例特异且可被因果验证的 read-path law，否则不能把它改名继续。

## 8. 对主搜索的影响

1. C65 的 WPCM 不应进入 GPU 队列；“language cannot write visual memory”不是新增 invariant。
2. 后续候选必须先区分 **memory state** 与 **read operator**；把 read failure 叙述成 state corruption 一律公式级拒绝。
3. baseline 未暂停；本审计完全在 CPU 完成。

