# CECD 端到端执行 DAG：outcome-blind / no-GPU 审计

**日期：** 2026-08-03  
**范围：** clinical admission v4、three-stage v3、dual-semantics formal CE、
VinDr 14 类 listing admission/runtime、HALP、System/PIH、reader-threshold
aliasing。  
**审计边界：** 未打开任何人类回传、admission decision、模型输出、Stage-1
analysis 或评测结果；未初始化 CUDA、未运行模型。

## 结论先行

当前没有图论意义上的循环依赖。Listing 的两个父条件：

1. 独立 listing render/prompt 人类 admission；
2. 上游 binary CE three-stage GO；

是一个 **convergent AND**，没有从 listing 回指 CE 的边。然而有七个会让真人回传后
永久停住、错误并发或接受伪造 provenance 的 fatal handoff blocker：

| ID | 触发时刻 | 结论 |
|---|---|---|
| `DUAL_PREFLIGHT_PRODUCER_GAP` | Stage-1 GO 后 | transition monitor 等待当前不存在的 dual preflight；仓库没有 canonical builder |
| `DUAL_RUNNER_TRIGGER_GAP` | authorization 后 | transition 只写/revalidate authorization，不 prepare/launch formal CE runner |
| `GPU0_LOCK_SPLIT_BRAIN` | dual GPU launch 前 | three-stage/listing 使用 `gpu0-vindr-v2.lock`，dual 默认另一把 GPU0 lock |
| `LISTING_ADJUDICATION_ADMISSION_GAP` | listing returns valid 后 | monitor 在结构验证处终止；没有 human adjudication/admission producer |
| `LISTING_RECEIPT_PROVENANCE_UNCLOSED` | admission 前 | runtime receipt 只检查布尔字段，未要求 return、attestation、adjudication、analyzer source 的哈希记录 |
| `LISTING_UPSTREAM_CE_HASH_UNVERIFIED` | listing GPU launch 前 | runtime 只要求 upstream hash 是非零 64 位 hex，没有与 canonical three-stage artifact 比对 |
| `LISTING_RUNNER_TRIGGER_GAP` | receipt 后 | 没有 Huatuo/Hulu pilot→dev→confirmation 的 canonical detached scheduler |

这七项全是科学/执行合同问题，不是 Linux、Codex 或 shell 权限问题。
`general_gpu_authorized=false` 的含义是“不授权任意额外 GPU 实验”；它不禁止已经被
命名且 hash-bound 的 controlled comparison。

## 状态机

```text
four clinical returns + verified clinical pack
                    |
                    v
       [clinical admission v4]
          | FAIL             | PASS
          v                  v
   terminal NO-GO     [CECD three-stage v3]
                         | FAIL       | BOTH-MODEL GO
                         v            v
                  terminal NO-GO   [dual preflight builder]  <-- MISSING
                                         |
                                         v
                              [transition authorization]
                                         |
                                         v
                              [formal CE detached trigger]   <-- MISSING

four independent listing returns
              |
              v
[structural validator / monitor]
              |
              v
[human adjudication + provenance-closed receipt]             <-- MISSING
              ^
              |
   CE three-stage BOTH-MODEL GO hash
              |
              v
 [Huatuo/Hulu pilot -> dev -> confirmation scheduler]        <-- MISSING
```

Listing 的上游依赖不是循环。Operational dead end 来自 AND 合流节点及其 producer
缺失。

## 已闭合的自动链

### Clinical admission → three-stage v3

`cecd-clinical-admission-monitor-v4` 当前 detached supervisor/child 存活。它执行：

1. 等待 4 个独立角色的 8 个精确文件名；
2. 要求两次轮询 size/SHA-256 不变；
3. 在打开 sealed mapping 前冻结 source pack 与完整 human bundle；
4. 运行 v3 return validator 与 admission analyzer；
5. FAIL 终止且不重试；PASS 才启动 `cecd-three-stage-v3`。

Three-stage 的执行顺序固定：

```text
pilot_screen:        pilot / 10 per finding-vote bin / 160 claims per model
dev_fit:             dev / 20 per bin / 320 claims per model
confirmation_locked: confirmation / 60 per bin / 960 claims per model
```

精确 selection SHA-256：

```text
pilot_screen        276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a
dev_fit             2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420
confirmation_locked 39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9
```

Verifier 再绑定 admission hash、两个模型的 config/weights/conformance、每个 stage
19-cell 完整 orbit、factorial row hashes、dev fit hash、confirmation hash、analyzer
source hash，以及三 stage whole-image 零重叠。这里没有发现循环或 hash 失效。

### 当前静态 substrate hashes

```text
clinical pack manifest
62e5ebc13f572ebea0e7f082b8fb8a66f51b5685b1a2b92335de04096c82f4ec

clinical sealed mapping
f7f7a86596b91038aecf059c4254255fe143042432288492ba76f12fbeada93d

clinical delivery index
aa4c729e781ffae7cc03d6f72a4ac83d53eece619575032df477f46bdc8e2190

listing admission pack manifest
54cb1d96dc5bd66d5ad59ea1cd8bcdbb7dc4acd2102dfc90dd092c51e661f109

listing experiment manifest
79e7469be61ef17ad9ac7764652e077434cad39fc9a806d7118ce659ec97be06

listing reference JSONL
0c229f63b2c3427bdc49eebb5072abacf650471bc1ba98b1b3552e603f950213
```

## Dual-semantics 交接接口

### Builder 必须满足的接口

未来 canonical builder 必须一次性生成：

```text
configs/cecd_dual_semantics_preflight_v1.json
configs/cecd_dual_semantics_preflight_v1.inputs.json
```

验收条件：

1. 先调用 v3 verifier 重建 genuine two-model GO；不得只相信一个布尔字段；
2. preflight 精确绑定 `confirmation_locked.json`、`input_gate.json`、clinical
   `analysis.json` 的 hashes；
3. Huatuo/Hulu `model_fingerprints.model_id` 与 passing model IDs 完全相等；
4. dev calibration、locked evaluation、ordered record keys、claim contract 都是
   regular-file + SHA-256 closure，至少 30 个 image clusters；
5. sidecar 的 `preflight_sha256` 匹配 preflight，并绑定 model dirs、Huatuo source
   root 和四个 input paths；
6. 两文件写一次或逐字相等，method output root 必须尚未产生 outputs；
7. 不读取 method outputs，不生成 clinical label，不修改 confirmation threshold；
8. builder 完成后，现有 transition monitor 才可以写 authorization。

### Runner trigger 必须满足的接口

Authorization 产生后，应由一个 canonical detached job 幂等执行当前真正实现的范围：

```bash
CECD_DUAL_WORKER=anchor/corrected_sgta/cecd_dual_semantics_worker_v1.py \
CECD_DUAL_GPU_LOCK=corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock \
CECD_DUAL_EXECUTE_CE_ONLY=1 \
bash scripts/run_cecd_dual_semantics_controlled_v1.sh
```

不能使用 `CECD_DUAL_EXECUTE=1`：当前 CECD hidden intervention、两种 Treble 和 OE
仍是明确 `method_not_implemented`；full runner 会正确 fail-stop。Formal CE-only
允许的只是 7 个 centered-logit factorial controls。

Trigger 验收条件：

- 唯一 detached job name 与 write-once run contract；
- 同一 canonical GPU0 lock；
- 已完成 model×method shard 校验后直接 resume/skip；
- previous `failed_stop` 只能显式 `--resume-after-failure`；
- 完成只写 `ce_stage_manifest.json`，不得伪装成 full `run_manifest.json`；
- 不自动解释 metrics 或授权 paper claim。

## Listing 交接接口

### Return monitor 的合法终点

现有 listing monitor 只验证 4 个 reviewer/attestation 对的结构，并在
`four_independent_returns_structurally_valid` 终止。这是正确的科学边界。安全自动化
只能额外制作一个 hash inventory / `ready_for_human_adjudication` bundle；不能合并
临床判断、自动 adjudicate、补标签或造 receipt。

### Admission receipt 的最低闭合条件

未来 receipt producer 必须由独立 human adjudication 触发，并要求：

- 4 个 completed returns 与 4 个 attestations 的 absolute path/SHA-256/bytes；
- structural validation artifact 与 validator source hash；
- frozen adjudication protocol、两位 primary decisions、分歧 adjudicator record；
- pack manifest、experiment manifest、reference 与 frozen guard-failure-set hashes；
- canonical upstream `cecd_three_stage_v3/input_gate.json` 和
  `confirmation_locked.json` 的 path/SHA-256/bytes；
- 由 v3 verifier 重建 `both_models_pass=true`，不能接受任意非零 hash；
- analyzer/receipt-builder source hash；
- `model_outputs_read_for_admission=false`；
- write-once receipt 及单独 pin artifact。

当前 listing runtime 对最后两类 upstream/provenance binding 不充分；在修复前不能
运行科学 GPU job，即使手上有一个字段全为 `true` 的 JSON。

### 两模型 scheduler 的最低闭合条件

Receipt genuine GO 后，按以下顺序各跑 Huatuo/Hulu，全部使用
`gpu0-vindr-v2.lock`：

```text
pilot -> dev -> confirmation_locked
```

每个 `run_runtime` 已支持 atomic cell shards 与相同 config 的自动 resume。Scheduler
还必须：

1. 每个 split/model 使用唯一 write-once output root；
2. busy GPU lock 是 recoverable wait/retry，不创建第二把锁；
3. corrupt existing shard 是 fail-stop，不静默覆盖；
4. pilot 不得作科学 GO/NO-GO；
5. dev-only 冻结方法/阈值；confirmation 只 apply，不重新选择；
6. confirmation 的原始生成可以提前完成，但结果必须 sealed/unread，直到 dev choice
   write-once；
7. fixed-K 与 matched-coverage 两套 evaluation 必须共同完成；
8. parser failures、hedges、refusals、claim budget 与 omission 不得被过滤。

## 失败恢复

| 节点 | 已实现恢复 | 必须人工审计/显式动作 |
|---|---|---|
| clinical returns | 两次稳定轮询；输入错误继续等待 | admission FAIL 为科学终止，不重写回传 |
| three-stage model run | 每 stage `config.json` 存在即 `--resume`，atomic cells 可跳过 | canonical detached state 为 `failed` 时，审计 log 后以同 name/state/command 重启；随后重启已终止的 transition monitor |
| dual formal CE | raw four-cell shared cache、model×method atomic shards、完整 shard skip | `failed_stop` 后仅 `--resume-after-failure`；run-contract/hash drift 必须换全新 protocol，不可覆盖 |
| listing generation | exact shard validation；完成 resume 不加载模型/不取 GPU lock | corrupt shard/config collision fail-stop；只能审计后修复来源或换新 output root |
| human admission | write-once/fail-closed | human/adjudication FAIL 不重跑模型来“救” admission |

## 控制分支 readiness

- **HALP:** Huatuo/Hulu CPU source audit 与 fake-hook conformance 已闭合；仍为
  conceptual compatibility port。`probe_training_authorized=false`，没有 model-specific
  dev probe fit/confirmation apply pipeline。
- **System/PIH:** generic tensor components 已 hash-bound；Huatuo/Hulu runtime eager
  integration、native-vs-eager canary、pre-`o_proj` integration、dev selected/random
  heads 均缺失。当前正确状态是 `control_execution_ready=false`。
- **Reader-threshold aliasing:** protocol/analyzer source 已绑定，但 dev-fit input、
  confirmation input、listing matched-count/length input 均为 null。它只能是 alternative
  explanation control，不能修改 CECD primary gate，也不能授权 mitigation。

这些控制没有反向依赖 CECD 来“证明自己”，因此没有循环；它们是 CECD GO 后的
adversarial compatibility closure，未就绪时必须阻止更宽 paper claim，但不应被显示成
系统命令权限请求。

## CPU verifier

新增：

```text
anchor/corrected_sgta/audit_cecd_execution_dag_v1.py
tests/test_audit_cecd_execution_dag_v1.py
corrected_runs/vindr_v2/cecd_execution_dag_audit_v1/audit.json
```

运行：

```bash
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.audit_cecd_execution_dag_v1 \
  --output corrected_runs/vindr_v2/cecd_execution_dag_audit_v1/audit.json

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.audit_cecd_execution_dag_v1 --strict
```

普通模式始终产生 machine-readable audit；`--strict` 在任何 blocker 存在时 exit 2。
Verifier 在代码层禁止打开 return/outcome roots，只读取静态 source/config/manifest 和
detached process metadata。

当前测试：

```text
4 passed
```

当前三个 canonical monitors 均 `alive=true` 且 command/name 匹配。当前 verdict 是
`blocked_handoffs_outcome_blind`；这说明 pipeline 工程未闭合，不表示科学假设失败，
也不表示缺少操作系统权限。

本文件与 audit artifact 是 **pre-repair snapshot**。当前 machine-readable audit 有
7 个 fatal handoff blockers，fingerprint 为：

```text
7a4d16cd181367fda3e86be6733d2a768f7db70c28114027ce311fdf9a97ff2e
```

Artifact file SHA-256：

```text
31c7f7e486d58941d5288f23786e8131fc7643d32dea6ce1bc5b82bb5785cf7b
```

Dual 与 listing 的 repair 分支正在由独立执行单元处理；修复后必须重跑同一个
verifier，不能手工删除 blocker 或覆盖本 snapshot 来宣称通过。
