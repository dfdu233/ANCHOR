#!/usr/bin/env python3
"""Build a second untouched two-domain source confirmation split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from corrected_sgta.build_rule_source_manifest import BuildConfig,load_iu,load_slake,load_vqarad,sha256_bytes,sha256_file,source_stats,stable_digest,write_json_and_jsonl
from corrected_sgta.build_rule_source_confirm_manifest import independence_unit
VERSION="rule-source-reconfirm-manifest-v1"
DOMAINS=("rule_iuxray","vqa_rad_train")
def load(path): return json.loads(Path(path).read_text())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base-manifest',type=Path,required=True); ap.add_argument('--exclude-confirm-manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260726); ap.add_argument('--images-per-domain',type=int,default=96); a=ap.parse_args()
 out=a.output_dir/'manifest.json'
 if out.exists(): raise FileExistsError(out)
 base=load(a.base_manifest); previous=load(a.exclude_confirm_manifest); c=base['config']
 cfg=BuildConfig(Path(c['iu_json']),Path(c['iu_image_root']),Path(c['slake_root']),Path(c['vqarad_parquet']),Path(c['locked_test']),Path(c['locked_image_root']),a.output_dir,a.seed,0.2,0,1)
 excluded=[]
 for split in ('train','dev'): excluded.extend(load(base['outputs'][split]['json']))
 for spec in previous['outputs'].values(): excluded.extend(load(spec['json']))
 ex_hash={r['image_sha256'] for r in excluded}; ex_units={(r['source_domain'],independence_unit(r)) for r in excluded}
 allrows=load_iu(cfg)+load_slake(cfg)+load_vqarad(cfg)
 chosen={}
 for d in DOMAINS:
  grouped={}
  for r in allrows:
   if r['source_domain']!=d or r['image_sha256'] in ex_hash or (d,independence_unit(r)) in ex_units: continue
   grouped.setdefault(r['image_sha256'],[]).append(r)
  hashes=sorted(grouped,key=lambda h:stable_digest(a.seed,'reconfirm-image',d,h))
  if len(hashes)<a.images_per_domain: raise RuntimeError(f'{d}: only {len(hashes)} unused images')
  rows=[]
  for h in hashes[:a.images_per_domain]: rows.append(sorted(grouped[h],key=lambda r:stable_digest(a.seed,'reconfirm-qa',r['id'],r['conversations'][0]['value']))[0])
  chosen[d]=sorted(rows,key=lambda r:r['id'])
 outputs={d:write_json_and_jsonl(a.output_dir/f'reconfirm.{d}',rows) for d,rows in chosen.items()}
 protocol={"version":VERSION,"base_manifest_fingerprint":base['fingerprint'],"previous_confirm_fingerprint":previous['fingerprint'],"seed":a.seed,"images_per_domain":a.images_per_domain,"domains":list(DOMAINS),"records":[{"domain":d,"id":r['id'],"image_sha256":r['image_sha256']} for d,rows in sorted(chosen.items()) for r in rows]}
 fp=sha256_bytes(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()); payload={**protocol,"fingerprint":fp,"base_manifest_sha256":sha256_file(a.base_manifest),"exclude_confirm_sha256":sha256_file(a.exclude_confirm_manifest),"target_file_opened":False,"excluded_images":len(ex_hash),"selected":{"total":source_stats([r for rows in chosen.values() for r in rows]),"domains":{d:source_stats(rows) for d,rows in chosen.items()}},"outputs":outputs}
 a.output_dir.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({"manifest":str(out),"fingerprint":fp,"selected":payload['selected']},indent=2))
if __name__=='__main__': main()
