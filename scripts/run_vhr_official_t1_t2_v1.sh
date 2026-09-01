#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
env=/home/dbw/.venvs/vhr-official-4.45
model=/home/dbw/models/llava-hf-llava-1.5-7b-hf-16952161b5e90aea6e332e36a6fe99024096dd0a
root=corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15/vhr_gates
provenance="$root/gate_provenance_v2.json"
manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
log=corrected_runs/detached_jobs/logs/vhr-official-t1-t2-v1.log
while [[ ! -f "$model/download_audit.json" || ! -x "$env/bin/python" ]]; do sleep 30; done
"$env/bin/python" -c 'import torch,transformers;assert transformers.__version__=="4.45.0" and torch.__version__.startswith("2.1.2")' || exit 2
provenance_fp=$(PYTHONPATH=. "$env/bin/python" -m anchor.medeval.build_baseline_gate_provenance_v1 \
  --output "$provenance" --model llava15-hf --method VHR \
  --checkpoint "$model" --config configs/unified_eval/baseline_matrix_v1.json \
  --manifest "$manifest" \
  --source anchor/medeval/build_baseline_gate_provenance_v1.py \
  --source anchor/corrected_sgta/run_vhr_official_baseline_v1.py \
  --source anchor/corrected_sgta/run_trained_llava_baseline_v1.py \
  --source third_party/baselines/VHR/generation.py \
  --source third_party/baselines/VHR/vhr.py \
  --source third_party/baselines/VHR/main.py \
  --generation-json '{"limit":32,"max_new_tokens":256,"seed":42,"arms":["native","off","vhr","custom_base"],"vhr_aug_ratio":2.0,"vhr_last_layers":14,"vhr_layer1":true,"vhr_filter":true}')
gate_run="$root/gate_runs/$provenance_fp"
mkdir -p "$gate_run"
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock; flock 8
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.run_trained_llava_baseline_v1 \
  --variant base --manifest "$manifest" --image-root "$images" --output-dir "$gate_run/custom_base" \
  --limit 32 --max-new-tokens 256 --seed 42 >>"$log" 2>&1 || exit $?
for arm in native off vhr; do
  method=greedy; extra=()
  [[ "$arm" == native ]] && extra+=(--native-control)
  [[ "$arm" == vhr ]] && method=vhr
  PYTHONPATH=third_party/baselines/VHR:. "$env/bin/python" -m anchor.corrected_sgta.run_vhr_official_baseline_v1 \
    --model-path "$model" --manifest "$manifest" --image-root "$images" \
    --output-dir "$gate_run/$arm" --method "$method" --limit 32 --max-new-tokens 256 --seed 42 \
    "${extra[@]}" >>"$log" 2>&1 || exit $?
done
/opt/miniconda3/bin/python - "$gate_run" "$provenance" "$provenance_fp" "$root/t1_t2_audit.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); provenance=Path(sys.argv[2]); binding=json.loads(provenance.read_text())
def load(arm):return [json.loads(x) for x in open(root/arm/'answers.jsonl') if x.strip()]
native,off,vhr,custom=map(load,['native','off','vhr','custom_base'])
def ids(x):
    # The custom compatibility runner stores both the decoded suffix
    # (without EOS) and the raw generated sequence.  Official VHR outputs
    # retain EOS, so compare the raw sequence when available to avoid a
    # bookkeeping-only false T1 failure.
    return [r['metadata'].get('raw_generated_token_ids', r['metadata']['generated_token_ids']) for r in x]
exact=sum(a==b for a,b in zip(ids(native),ids(off)))
cross=sum(a==b for a,b in zip(ids(native),ids(custom)))
changed=sum(a!=b for a,b in zip(ids(off),ids(vhr)))
provenance_passed=binding.get('fingerprint')==sys.argv[3]
result={'protocol':'vhr-official-t1-t2-v2','gate_provenance':str(provenance),'gate_provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest(),'gate_provenance_fingerprint':binding.get('fingerprint'),'provenance_passed':provenance_passed,'n':32,'t1_native_vs_off_token_exact':exact,'t1_passed':exact==32,'common_base_custom_vs_hf_token_exact':cross,'common_base_passed':cross==32,'t2_vhr_changed_sequences':changed,'t2_passed':changed>=1,'passed':provenance_passed and exact==32 and cross==32 and changed>=1,'arms':{a:hashlib.sha256((root/a/'answers.jsonl').read_bytes()).hexdigest() for a in ['native','off','vhr','custom_base']}}
Path(sys.argv[4]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
PY
flock -u 8
