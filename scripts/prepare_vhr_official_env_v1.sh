#!/usr/bin/env bash
set -euo pipefail
env=/home/dbw/.venvs/vhr-official-4.45
log=/home/dbw/ANCHOR/corrected_runs/detached_jobs/logs/vhr-official-env.log
wheel_dir=/home/dbw/model_cache/python_wheels/vhr-official
torch_wheel="$wheel_dir/torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl"
torch_url='https://files.pythonhosted.org/packages/03/f1/13137340776dd5d5bcfd2574c9c6dfcc7618285035cd77240496e5c1a79b/torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl'
torch_sha256=3a871edd6c02dae77ad810335c0833391c1a4ce49af21ea8cf0f6a5d2096eea8
if [[ ! -x "$env/bin/python" ]]; then
  /home/dbw/.venvs/hulumed/bin/python -m venv --system-site-packages "$env"
fi
mkdir -p "$wheel_dir" "$(dirname "$log")"

# The 670 MB official PyPI wheel repeatedly timed out through pip.  Download it
# resumably and verify the exact artifact before installation; this changes no
# dependency version and prevents a truncated wheel from entering the gate.
until wget --no-check-certificate -c --timeout=300 --tries=0 -O "$torch_wheel" \
  "$torch_url" >>"$log" 2>&1; do
  sleep 15
done
printf '%s  %s\n' "$torch_sha256" "$torch_wheel" | sha256sum -c - >>"$log" 2>&1

env -u ALL_PROXY -u all_proxy -u HTTPS_PROXY -u https_proxy \
  -u HTTP_PROXY -u http_proxy \
  NO_PROXY=pypi.org,files.pythonhosted.org \
  "$env/bin/pip" install --index-url https://pypi.org/simple \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  --timeout 300 --retries 20 \
  "$torch_wheel" transformers==4.45.0 accelerate==0.26.0 numpy==1.26.3 pandas==2.2.2 \
  pillow==10.2.0 protobuf==3.20.0 sentencepiece==0.1.99 >>"$log" 2>&1
"$env/bin/python" - <<'PY' >>"$log" 2>&1
import accelerate,json,numpy,PIL,torch,transformers
from pathlib import Path
print({'transformers':transformers.__version__,'torch':torch.__version__,'accelerate':accelerate.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__})
assert transformers.__version__ == '4.45.0' and torch.__version__.startswith('2.1.2')
Path('/home/dbw/.venvs/vhr-official-4.45/env_audit.json').write_text(json.dumps({'passed':True,'transformers':transformers.__version__,'torch':torch.__version__,'accelerate':accelerate.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__},indent=2)+'\n')
PY
