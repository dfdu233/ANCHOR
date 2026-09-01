#!/usr/bin/env bash
set -uo pipefail
root=/home/dbw/model_cache/report_metrics
log=/home/dbw/ANCHOR/corrected_runs/detached_jobs/logs/report-metric-checkpoint-download.log
mkdir -p "$root" "$(dirname "$log")"

fetch() {
  local repo=$1 file=$2 output=$3
  mkdir -p "$(dirname "$output")"
  until env -u ALL_PROXY -u all_proxy aria2c --check-certificate=false --continue=true --max-tries=0 \
    --retry-wait=5 --timeout=300 --max-connection-per-server=16 --split=16 \
    --min-split-size=1M --file-allocation=none --auto-file-renaming=false \
    --allow-overwrite=true --all-proxy=http://172.17.0.1:7890 \
    --dir="$(dirname "$output")" --out="$(basename "$output")" \
    "https://huggingface.co/$repo/resolve/main/$file?download=true" >>"$log" 2>&1; do sleep 15; done
}

for file in added_tokens.json config.json model.safetensors special_tokens_map.json spm.model tokenizer.json tokenizer_config.json; do
  fetch Angelakeke/RaTE-NER-Deberta "$file" "$root/rate-ner-deberta/$file"
done
for file in config.json model.safetensors sentence_bert_config.json special_tokens_map.json tokenizer_config.json vocab.txt; do
  fetch FremyCompany/BioLORD-2023-C "$file" "$root/biolord-2023-c/$file"
done
fetch StanfordAIMI/RRG_scorers chexbert.pth "$root/xdg/chexbert/chexbert.pth"

bert="$root/hf_home/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
for file in config.json tokenizer.json tokenizer_config.json vocab.txt; do
  fetch google-bert/bert-base-uncased "$file" "$bert/$file"
done

# Reuse the already verified HuggingFace blob without duplicating 579 MB.
mkdir -p "$root/radgraph"
ln -sfn "$root/radgraph/models--StanfordAIMI--RRG_scorers/blobs/dfc715608d0f1bfc15168f80cd1953ea63bd7960a5506401848858f8dbdea5ff" \
  "$root/radgraph/modern-radgraph-xl.tar.gz"

/opt/miniconda3/bin/python - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
files={
 'modern-radgraph-xl':root/'radgraph/modern-radgraph-xl.tar.gz',
 'chexbert.pth':root/'xdg/chexbert/chexbert.pth',
 'RaTE-NER-Deberta':root/'rate-ner-deberta/model.safetensors',
 'BioLORD-2023-C':root/'biolord-2023-c/model.safetensors',
}
expected={
 'modern-radgraph-xl':'dfc715608d0f1bfc15168f80cd1953ea63bd7960a5506401848858f8dbdea5ff',
 'chexbert.pth':'6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1',
 'RaTE-NER-Deberta':'b380aae68a26fc608cbb568994006910d0fc7156c63fd05308b3bbaf77ed53f2',
 'BioLORD-2023-C':'6164f3a7476eaaeee4b7fa2d4dea655f8ca1927853b867d59fb9d66507f5d36f',
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
actual={k:sha(v) for k,v in files.items()}
result={'checkpoints':{k:{'path':str(files[k]),'sha256':actual[k]} for k in files},'expected_sha256':expected,'passed':actual==expected}
(root/'download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={'protocol':'report-metric-checkpoints-v1','checkpoints':{k:{'sha256':actual[k]} for k in files}}
(root/'metric_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
PY
