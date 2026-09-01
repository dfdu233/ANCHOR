#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p corrected_runs/detached_jobs/locks corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola
root=corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola
provenance="$root/gate_provenance_v2.json"
provenance_fp=$(PYTHONPATH=. /home/dbw/.venvs/qwen25vl-v2/bin/python -m anchor.medeval.build_baseline_gate_provenance_v1 \
  --output "$provenance" --model qwen --method DoLa \
  --checkpoint /home/dbw/models/Qwen2.5-VL-7B-Instruct \
  --config configs/unified_eval/baseline_matrix_v1.json \
  --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
  --source anchor/medeval/build_baseline_gate_provenance_v1.py \
  --source anchor/corrected_sgta/models_oe.py \
  --source anchor/corrected_sgta/cross_model_dola.py \
  --source anchor/corrected_sgta/run_cross_model_dola_gate_v1.py \
  --source anchor/corrected_sgta/run_cross_model_method_full_v1.py \
  --generation-json '{"limit":32,"max_new_tokens":256,"seed":42,"decode":"greedy","candidate_policy":"DoLa"}')
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
out="$root/gate_runs/$provenance_fp/t2_n32.jsonl"
mkdir -p "$(dirname "$out")"
PYTHONPATH=. /home/dbw/.venvs/qwen25vl-v2/bin/python \
  -m anchor.corrected_sgta.run_cross_model_dola_gate_v1 \
  --model qwen \
  --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
  --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images \
  --output "$out" --limit 32 --max-new-tokens 256 --seed 42 \
  > corrected_runs/detached_jobs/logs/qwen-dola-t2-v1.log 2>&1
PYTHONPATH=. /opt/miniconda3/bin/python - "$out" "$provenance" "$provenance_fp" "$root/summary.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
provenance=Path(sys.argv[2]); binding=json.loads(provenance.read_text()); provenance_passed=binding.get('fingerprint')==sys.argv[3]
summary={"protocol":"qwen-dola-t1-t2-v2","gate_provenance":str(provenance),"gate_provenance_sha256":hashlib.sha256(provenance.read_bytes()).hexdigest(),"gate_provenance_fingerprint":binding.get('fingerprint'),"provenance_passed":provenance_passed,"n":len(rows),"off_token_exact":sum(x["off_token_exact"] for x in rows),"dola_changed":sum(x["dola_changed"] for x in rows),"passed":provenance_passed and len(rows)==32 and all(x["off_token_exact"] for x in rows) and any(x["dola_changed"] for x in rows)}
Path(sys.argv[4]).write_text(json.dumps(summary,indent=2)+'\n')
raise SystemExit(0 if summary['passed'] else 2)
PY
flock -u 8
