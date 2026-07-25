#!/usr/bin/env python3
"""Resumably cache HuluMed Yes/No surface-energy margins on named source files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.models_surface import HuluSurfaceAdapter
from corrected_sgta.rule_source_preference import canonical_binary_answer,file_sha256,rule_mimic_prompt,stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import canonical_answer
VERSION='rule-hulu-source-margin-cache-v1'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source-json',action='append',required=True); ap.add_argument('--raw',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 paths={};
 for spec in a.source_json:
  name,value=spec.split('=',1); paths[name]=Path(value)
 inputs={d:file_sha256(p) for d,p in sorted(paths.items())}; fp=stable_json_sha256({'version':VERSION,'inputs':inputs,'code':file_sha256(Path(__file__))}); done={}
 if a.raw.exists():
  for line in a.raw.read_text().splitlines():
   r=json.loads(line)
   if r['fingerprint']!=fp: raise ValueError('raw fingerprint mismatch')
   done[(r['domain'],r['id'])]=r
 selected={d:json.loads(p.read_text()) for d,p in paths.items()}; total=sum(map(len,selected.values())); a.raw.parent.mkdir(parents=True,exist_ok=True); adapter=HuluSurfaceAdapter()
 try:
  with a.raw.open('a') as out:
   for d,rows in sorted(selected.items()):
    for row in tqdm(rows,desc=f'hulu-margin:{d}',total=len(rows)):
     key=(d,str(row['id']))
     if key in done: continue
     with Image.open(row['image']) as h: image=h.convert('RGB')
     q=str(row['conversations'][0]['value']).replace('<image>','').strip(); prompt=rule_mimic_prompt(q); gt=canonical_binary_answer(canonical_answer(row['conversations'][1]['value'])); f=adapter.forward_ce([image],prompt,['Yes','No'])[0]; scores={'Yes.':float(f.logits[0]),'No.':float(f.logits[1])}; pred=max(scores,key=scores.get); rec={'version':VERSION,'fingerprint':fp,'domain':d,'id':str(row['id']),'image':row['image'],'ground_truth':gt,'predictions':{'identity':pred},'sequence_log_probabilities':{'identity':scores},'score_interface':'maximum single-token surface energy over case/whitespace forms'}; out.write(json.dumps(rec)+'\n'); out.flush(); done[key]=rec
 finally: adapter.close()
 records={d:[done[(d,str(r['id']))] for r in rows] for d,rows in selected.items()}; payload={'version':VERSION,'fingerprint':fp,'status':'final','model':'Hulu-Med-14B','target_labels_used':False,'score_interface':'maximum single-token surface energy over case/whitespace forms','inputs':inputs,'records':records}; atomic_json(a.output,payload); print(json.dumps({'fingerprint':fp,'counts':{d:len(v) for d,v in records.items()}},indent=2))
if __name__=='__main__': main()
