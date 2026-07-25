# SGTA-v4 / SGTA-ConfGen-v2 运行说明

## 0. 当前硬门控

```bash
cd /root/autodl-tmp/Hulu-Med/MedUniEval
corrected_sgta/run_wave0.sh
```

当前本机 Qwen2-VL-7B snapshot 缺 5 个权重 shard、processor/tokenizer 配置；Report 端缺 RaTEScore、RadGraph、BERTScore 以及 CheXbert/RadGraph checkpoint。`readiness.json` 未通过前，Knowledge/Report OE 只能作探索，不能作为论文证据。

人工或本地 judge 完成盲评 JSONL 后：

```bash
python -m corrected_sgta.analyze_judge_agreement \
  --left annotator_a.jsonl --right annotator_b.jsonl \
  --output corrected_runs/sgta_v4_wave0_v54/knowledge_judge_agreement.json
```

要求 quadratic Cohen κ≥0.60 且 Spearman≥0.70。

## 1. CE pilot 与 validation

```bash
corrected_sgta/run_wave1.sh pilot
corrected_sgta/run_wave2.sh pilot

# 只有查看 pilot 缓存、decode 和结构字段均正确后才执行：
corrected_sgta/run_wave1.sh validation
corrected_sgta/run_wave2.sh validation
cat corrected_runs/sgta_v4_wave2_validation_v54/gates.json
```

`run_wave1.sh` 对 Hulu/LLaVA × CXR/MM 缓存 matched-center 频域网格。LLaVA 将同题 style views 批量前向，Hulu 因模型约束逐 style 前向。脚本可恢复；`--max-samples` 是固定 qid-hash 目标总数，不会在重启时继续膨胀。



### 1.1 Pixel-frequency stress diagnostic

Fresh pilot 已显示当前 pixel-frequency SGTA 无 surface/sequence oracle headroom。若需要复现诊断，可运行：

```bash
corrected_sgta/run_wave1_stress.sh pilot
cat corrected_runs/sgta_v4_wave1_stress_pilot_v54/gates.json
```

该脚本使用更强 `L={0.05,0.10,0.20}` 与 `gamma={0.6,0.8,1.2,1.4}`，只用于诊断，不解锁 validation/full 或论文 claim。

## 2. OE pilot

默认严格读取 Wave-2 gate，未通过会拒绝生成。`FORCE=1` 仅供显式诊断，不得用作论文协议。

```bash
corrected_sgta/run_wave3.sh pilot-generate
# 完成盲评/临床指标后：
JUDGMENT_DIR=/path/to/judgments \
KNOWLEDGE_AGREEMENT_DIR=/path/to/per_model_agreements \
REPORT_METRIC_JSON=/path/to/report_metric_validation.json \
corrected_sgta/run_wave3.sh pilot-analyze
```

每个 judgment 文件名为 `${model}_${task}.jsonl`。Knowledge 至少包含 `item_id,hallucination_score,annotator_id,rubric_version,annotation_bundle_id,cache_fingerprint`；Report 至少包含 `item_id,clinical_entity_precision,clinical_fact_recall,critical_contradiction,metric_manifest_sha256`。所有 clinical 流必须严格为每题 8 个候选。Report metric manifest 先运行：

```bash
python -m corrected_sgta.validate_report_metric_manifest \
  --manifest report_metric_manifest.json \
  --output report_metric_validation.json
```

## 3. OE validation 与 full gate

```bash
corrected_sgta/run_wave3.sh validation-generate
# 同上配置 judgment/evidence gate 后：
corrected_sgta/run_wave3.sh validation-analyze
cat corrected_runs/sgta_confgen_v2_validation_v54/gates.json
corrected_sgta/run_wave4.sh
```

validation 默认 Knowledge 640、Report 490，因此 proper calibration 分别达到至少 128/147。Wave 4 只打印冻结后的全量命令；它不会绕过 gate 自动改论文。最终需要 3 seeds、paired bootstrap、McNemar、效应量、Holm 校正和独立完整性审计。

## 兼容入口

```bash
python sgta_confgen.py infer-ce --help
python sgta_confgen.py analyze-oe --help
```

旧 `sgta_confgen.py oe|ce_gen|oe_eval` 已禁用，原文件保存在 `sgta_confgen_legacy_v53.py`；旧 `medheval_sgta_v4.py` 保存在 `medheval_sgta_v4_legacy_v53.py`。


## 1.2 Feature-space SGTA + SCA-T CE path

After pixel-frequency SGTA failed to create stable surface/sequence oracle headroom, the current CE path is feature-space SGTA with semantic prototype TIM-KL. This does not regenerate VLM answers; it consumes v5.4 Wave-1 caches containing `style_features`, `style_logits`, sequence NLL, and decoded labels.

Run one split:

```bash
corrected_sgta/run_feature_sgta.sh validation 42
```

Run the three validation splits used in the current report:

```bash
for seed in 42 43 44; do
  corrected_sgta/run_feature_sgta.sh validation "$seed"
done
```

Useful overrides:

```bash
SELECTOR=tim-kl-only ITERATIONS=100 corrected_sgta/run_feature_sgta.sh validation 42
SELECTOR=alpha0-tim-kl corrected_sgta/run_feature_sgta.sh validation 42  # conservative SCA-T-only ablation
SELECTOR=calibration-all corrected_sgta/run_feature_sgta.sh validation 42  # diagnostic only; may select weak initial branch
```

Current formal selector is `tim-kl-only`: candidates are `baseline_surface_logits` plus feature/prototype `TIM-KL` methods. Ties prefer baseline/alpha=0 before alpha>0. Report alpha=0 and alpha>0 separately; do not attribute the full TIM-KL gain to FedDG style centers.
