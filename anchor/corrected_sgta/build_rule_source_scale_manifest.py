#!/usr/bin/env python3
"""Build a larger untouched IU-Xray validation split for a frozen method."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from corrected_sgta.build_rule_source_confirm_manifest import independence_unit
from corrected_sgta.build_rule_source_manifest import BuildConfig,load_iu,sha256_bytes,sha256_file,source_stats,stable_digest,write_json_and_jsonl
VERSION="rule-source-scale-manifest-v1"; DOMAIN="rule_iuxray"
def load(path): return json.loads(Path(path).read_text())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base-manifest',type=Path,required=True); ap.add_argument('--exclude-confirm-manifest',type=Path,required=True); ap.add_argument('--exclude-reconfirm-manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260727); ap.add_argument('--images',type=int,default=384); a=ap.parse_args()
 out=a.output_dir/'manifest.json'
 if out.exists(): raise FileExistsError(out)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 base=load(a.base_manifest); confirm=load(a.exclude_confirm_manifest); reconfirm=load(a.exclude_reconfirm_manifest); c=base['config']
 cfg=BuildConfig(Path(c['iu_json']),Path(c['iu_image_root']),Path(c['slake_root']),Path(c['vqarad_parquet']),Path(c['locked_test']),Path(c['locked_image_root']),a.output_dir,a.seed,0.2,0,1)
 excluded=[]
 for split in ('train','dev'): excluded.extend(load(base['outputs'][split]['json']))
 for manifest in (confirm,reconfirm):
  for spec in manifest['outputs'].values(): excluded.extend(load(spec['json']))
 ex_hash={r['image_sha256'] for r in excluded}; ex_units={(r['source_domain'],independence_unit(r)) for r in excluded}; grouped={}
 for row in load_iu(cfg):
  if row['image_sha256'] in ex_hash or (DOMAIN,independence_unit(row)) in ex_units: continue
  grouped.setdefault(row['image_sha256'],[]).append(row)
 hashes=sorted(grouped,key=lambda h:stable_digest(a.seed,'scale-image',DOMAIN,h))
 if len(hashes)<a.images: raise RuntimeError(f'{DOMAIN}: only {len(hashes)} unused images')
 rows=[sorted(grouped[h],key=lambda r:stable_digest(a.seed,'scale-qa',r['id'],r['conversations'][0]['value']))[0] for h in hashes[:a.images]]; rows=sorted(rows,key=lambda r:r['id'])
 outputs={DOMAIN:write_json_and_jsonl(a.output_dir/'scale.rule_iuxray',rows)}
 protocol={'version':VERSION,'base_manifest_fingerprint':base['fingerprint'],'previous_confirm_fingerprint':confirm['fingerprint'],'previous_reconfirm_fingerprint':reconfirm['fingerprint'],'seed':a.seed,'images_per_domain':a.images,'domains':[DOMAIN],'sequential_alpha':0.025,'records':[{'domain':DOMAIN,'id':r['id'],'image_sha256':r['image_sha256']} for r in rows]}
 fp=sha256_bytes(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()); payload={**protocol,'fingerprint':fp,'base_manifest_sha256':sha256_file(a.base_manifest),'exclude_confirm_sha256':sha256_file(a.exclude_confirm_manifest),'exclude_reconfirm_sha256':sha256_file(a.exclude_reconfirm_manifest),'target_file_opened':False,'excluded_images':len(ex_hash),'selected':{'total':source_stats(rows),'domains':{DOMAIN:source_stats(rows)}},'outputs':outputs}
 a.output_dir.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'manifest':str(out),'fingerprint':fp,'selected':payload['selected']},indent=2))
if __name__=='__main__': main()
