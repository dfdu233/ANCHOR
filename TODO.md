# Active Research Priorities

- [active/P0-20260831] 以 qualification-v3 重绑全部已有结果；OE 长度/截断只作诊断，明显循环才失败。
- [active/P0-20260831] Visual-MIMIC 全部方法补齐报告式 BLEU-1/2/3/4、ROUGE-1/2/L、METEOR 表；VQA-RAD 保留短答案指标并增加 answer-token recall。
- [blocked-external/P0-20260831] LLaVA 八个缺失单元已可恢复，但宿主不可见 GPU 上下文占用约34GB；显存释放后立即重启 `baseline_pending_recovery_v3`。
- [completed/P0-20260901] OmniMedVQA 已解压并冻结八模态88,995题/64题smoke；严格 evaluator 的 perfect/empty、逐模态与缺图门已通过。
- [active/P0-reference-parity] 最终表完整覆盖附件论文全部模型（HuatuoGPT-Vision-7B、LLaVA-1.6-7B）、方法（Baseline、VCD、ICD、AGLA、VAF、AvisC、CVE）和数据集（PMC-VQA、PathVQA、SLAKE、VQA-RAD、MMMU-med、OmniMedVQA、MIMIC-CXR generation）；OmniMedVQA 另报八模态和 macro average。缺失项只允许审计后标 N/A，不允许省略。
- [completed/P0-reference-data] PMC-VQA v2 test、PathVQA test、MMMU medical五学科与LLaVA-1.6权重均已落盘、哈希冻结；列表选项/空列表回归门和严格perfect/empty门通过。
- [completed/P0-reference-methods] VCD/ICD/CVE/DoLa已进入五模型统一运行器；AGLA 已完成 BLIP-ITM GradCAM clean-room 论文公式路径，AvisC、ClearSight/VAF 已加入明确标注的跨架构 proxy 与公式单测。它们与 paper-native 许可实现分列，逐模型 smoke 通过后才进入结果表。
- [active/P0-reference-matrix] `run_cross_model_complete_matrix_v1.sh` 等待 Baseline 退出后执行五模型×八推理臂 smoke；通过后继续冻结数据集的 full manifest，失败单元写入可审计 N/A。
- [active/P0-beyond-reference] 在附件全覆盖线之上加入本项目额外模型（至少 Hulu-Med、LLaVA-Med、Qwen 系列）、额外推理方法及现有 CE/OE/report 数据集；最终模型数、方法数、数据集/任务覆盖均严格多于附件论文，并用 coverage audit 验收。

- [in progress] 扩展并碰撞检查至少12个机制不同的候选。
- [completed] 完成rare/weak、active sensing及search-tax三条独立碰撞审计；经典局部聚合与普通VOI降级。
- [completed] 将search-tax严格限制为全局reader-null，并把patch gate升级为超过mean/max/top-K强对照。
- [completed] CPU审计VinDr局部/稀疏病灶现象：两模型×开发/确认均通过病种内面积—margin门。
- [completed] 审计可在2小时内完成的缓存复用致死实验入口并冻结运行顺序。
- [completed] 冻结首批L1/L2协议：criterion shift、layer mixture、lesion transplant、sparse patch scan、real second view。
- [in progress] 等baseline释放GPU后依序运行lesion transplant、Huatuo patch scan、条件Hulu patch scan、真实第二视图。
- [completed] lesion transplant n=128真实门：双向守恒律失败并按预注册关闭。
- [pending] 对通过门槛的候选进行第二模型、更多finding与matched-risk验证。
- [pending] 输出候选总表、失败记录、Top-3和有证据约束的论文主线。
# 2026-08-15 Goal execution priorities

- [active/P0] 保持 `baseline_native_recovery_20260815` 与 follow-up chain 持久运行；Codex 额度不足时不得停止 Baseline。
- [active/P0] 以 coverage audit 为唯一完成真相，收口 87 pending、3 partial/unscored 及所有训练型 gate/N/A 证据。
- [completed/B0] CEB 简单无标签视觉支持 CPU 致死门：未过 pair-AUROC 0.70，暂不在线解码；artifact=`corrected_runs/visual_edge_constraint_v1/huatuo_pairs34_suffix_direct/label_free_analysis.json`。
- [pending/B1] 仅在 Baseline 不受影响时，对已有 open-ended draft 做 Full/−V/−H teacher-forcing；若需要 GPU，必须等待 Baseline 释放并经 Goal turn 明确授权。
- [pending/B2] 只有 B1 通过且无标签 source-separation 分数超过熵基线，才实现 CEB 在线 canary；否则形成 NO-GO/PIVOT。
- [active/P0-20260828] VHR LLaVA gate repair: single-sample cached multimodal smoke passed; first 32-sample gate exposed EOS bookkeeping mismatch (not model divergence), raw-ID comparison patch is queued for a second gate; dependent full queue remains fail-closed.
- [active/P0-20260828] Keep Baseline monitor sessions alive and rerun coverage audit after every queue stage; no unexplained pending/partial may remain at Goal completion.
- [active/P0-20260828] LLaVA VCD visual-mimic chunks are being revalidated; completed chunks currently fail the behavioral quality gate for `empty_predictions`, so retain them as failed/partial until the queue writes final state and a qualification-backed N/A decision is recorded.
- [completed/DG-visual-audit] Added CPU-only `visualize_dg_alignment_v1.py` and generated the 128-case style dashboard; use it to separate image evidence destruction, boundary sensitivity, and decoder/domain effects before any DG mitigation claim.
- [active/DG-alignment] Preserve the DG hypothesis but audit four cells (original/DG × evidence-preserved/evidence-weakened). Do not widen style banks or run GPU until `DG_interaction` is reproducible across models and control claims.
- [active/DG-paired-canary] `dg_paired_validation_v1` 已挂到共享 GPU lock；优先复用已完成 Huatuo visual_mimic native 结果做 64-case FEDDG paired comparison，qualification 不通过则停止扩展。
