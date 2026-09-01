#!/usr/bin/env bash
set -uo pipefail
revision=16952161b5e90aea6e332e36a6fe99024096dd0a
repo=https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/$revision
target=/home/dbw/models/llava-hf-llava-1.5-7b-hf-$revision
log=/home/dbw/ANCHOR/corrected_runs/detached_jobs/logs/vhr-official-checkpoint-download.log
mkdir -p "$target" "$(dirname "$log")"
fetch() {
  local file=$1
  until wget --no-check-certificate -c --timeout=300 --tries=0 -O "$target/$file" \
    "$repo/$file?download=true" >>"$log" 2>&1; do sleep 15; done
}
files=(config.json generation_config.json preprocessor_config.json added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json model.safetensors.index.json)
for file in "${files[@]}"; do
  fetch "$file"
done
for file in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  fetch "$file"
done
/opt/miniconda3/bin/python - "$target" <<'PY'
import hashlib,json,math,struct,sys
from pathlib import Path
root=Path(sys.argv[1]); index=json.load(open(root/'model.safetensors.index.json'))
expected=int(index['metadata']['total_size'])
files=sorted(set(index['weight_map'].values()))
dtype_bytes={'BOOL':1,'U8':1,'I8':1,'U16':2,'I16':2,'F16':2,'BF16':2,
             'U32':4,'I32':4,'F32':4,'U64':8,'I64':8,'F64':8}
weight_map=index['weight_map']; tensor_bytes=0; headers={}; shard_sha256={};
observed_keys=set()
for name in files:
    path=root/name
    with path.open('rb') as handle:
        header_len=struct.unpack('<Q',handle.read(8))[0]
        header=json.loads(handle.read(header_len))
    keys={key for key in header if key != '__metadata__'}
    observed_keys.update(keys)
    shard_tensor_bytes=sum(math.prod(header[key]['shape'])*dtype_bytes[header[key]['dtype']] for key in keys)
    tensor_bytes += shard_tensor_bytes
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16*1024*1024), b''): digest.update(chunk)
    shard_sha256[name]=digest.hexdigest()
    headers[name]={'file_bytes':path.stat().st_size,'header_bytes':header_len,
                   'tensor_bytes':shard_tensor_bytes,'tensor_count':len(keys)}
expected_keys=set(weight_map)
# Each index key must occur in exactly the shard named by weight_map.  Checking
# this directly also rejects a valid-looking but mismatched shard set.
placement_ok=True
for name in files:
    with (root/name).open('rb') as handle:
        header_len=struct.unpack('<Q',handle.read(8))[0]
        keys={key for key in json.loads(handle.read(header_len)) if key != '__metadata__'}
    if keys != {key for key, shard in weight_map.items() if shard == name}:
        placement_ok=False
result={'expected_tensor_bytes':expected,'actual_tensor_bytes':tensor_bytes,
        'actual_file_bytes':sum((root/name).stat().st_size for name in files),
        'shards':files,'shard_audit':headers,'shard_sha256':shard_sha256,
        'index_key_count':len(expected_keys),'observed_key_count':len(observed_keys),
        'placement_ok':placement_ok,
        'complete':tensor_bytes==expected and observed_keys==expected_keys and placement_ok}
(root/'download_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['complete'] else 1)
PY
