#!/usr/bin/env python3
"""One-shot resumable evaluation of a frozen robust margin on RULE MIMIC."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.evaluate_medheval_answers import rule_pope_prediction
from corrected_sgta.evaluate_rule_source_preference_barycenter import _predict
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_preference import canonical_binary_answer,file_sha256,rule_mimic_prompt,stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import canonical_answer
VERSION='rule-source-robust-margin-target-v1'
def exact_p(r,h):
 n=r+h
 return 1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(min(r,h)+1))/(2**n))
def summary(records,base_key):
 base=np.array([r[base_key]==r['ground_truth'] for r in records]); new=np.array([r['calibrated']==r['ground_truth'] for r in records]); rescue=int((~base&new).sum()); harm=int((base&~new).sum())
 return {'n':len(records),'base_accuracy':float(base.mean()),'calibrated_accuracy':float(new.mean()),'delta_pp':float(100*(new.mean()-base.mean())),'rescues':rescue,'harms':harm,'mcnemar_exact_p':exact_p(rescue,harm)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source-manifest',type=Path,required=True); ap.add_argument('--source-gate',type=Path,required=True); ap.add_argument('--calibrator',type=Path,required=True); ap.add_argument('--questions',type=Path,required=True); ap.add_argument('--image-root',type=Path,required=True); ap.add_argument('--greedy-cache',type=Path,required=True); ap.add_argument('--raw',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 m=json.loads(a.source_manifest.read_text()); g=json.loads(a.source_gate.read_text()); c=json.loads(a.calibrator.read_text())
 if g.get('gate',{}).get('target_pilot_allowed') is not True or g.get('gate',{}).get('mcnemar_exact_p',1)>=g.get('gate',{}).get('alpha',0): raise ValueError('source gate did not pass')
 if c.get('target_labels_used') is not False: raise ValueError('calibrator used target labels')
 if Path(m['config']['locked_test']).resolve()!=a.questions.resolve(): raise ValueError('target questions path does not match sealed manifest')
 rows=[json.loads(x) for x in a.questions.read_text().splitlines() if x.strip()]; qids=[str(r['question_id']) for r in rows]
 if len(rows)!=3470 or len(set(qids))!=len(rows): raise ValueError('unexpected target set')
 greedy={}
 for line in a.greedy_cache.read_text().splitlines():
  r=json.loads(line)
  if r.get('status')=='ok': greedy.setdefault(str(r['question_id']),r)
 if set(greedy)!=set(qids): raise ValueError('greedy cache is incomplete or mismatched')
 code_hash=file_sha256(Path(__file__)); fp=stable_json_sha256({'version':VERSION,'source_manifest':file_sha256(a.source_manifest),'source_gate':file_sha256(a.source_gate),'calibrator':file_sha256(a.calibrator),'questions':file_sha256(a.questions),'greedy_cache':file_sha256(a.greedy_cache),'code':code_hash})
 done={}
 if a.raw.exists():
  for line in a.raw.read_text().splitlines():
   r=json.loads(line)
   if r.get('fingerprint')!=fp: raise ValueError('raw fingerprint mismatch')
   done[str(r['question_id'])]=r
 t=float(c['fit']['threshold']); a.raw.parent.mkdir(parents=True,exist_ok=True); adapter=LlavaMedAlignmentAdapter(conv_mode='vicuna_v1')
 for p in adapter.model.parameters(): p.requires_grad_(False)
 adapter.model.eval()
 try:
  with a.raw.open('a') as out:
   for row in tqdm(rows,desc='robust-margin:mimic',initial=len(done),total=len(rows)):
    qid=str(row['question_id'])
    if qid in done: continue
    path=a.image_root/str(row['image'])
    with Image.open(path) as h: image=h.convert('RGB')
    prompt=rule_mimic_prompt(row['question']); gt=canonical_binary_answer(canonical_answer(row['answer'])); _,scores=_predict(adapter,image,prompt,None); margin=scores['Yes.']-scores['No.']; identity='Yes.' if margin>=0 else 'No.'; calibrated='Yes.' if margin>=t else 'No.'
    greedy_pred=canonical_binary_answer(rule_pope_prediction(str(greedy[qid]['base_text'])))
    rec={'version':VERSION,'fingerprint':fp,'question_id':qid,'image':row['image'],'patient_id':str(row['image']).split('/')[1],'ground_truth':gt,'identity_constrained':identity,'identity_greedy_pope':greedy_pred,'calibrated':calibrated,'margin':margin,'threshold':t,'scores':scores}
    out.write(json.dumps(rec)+'\n'); out.flush(); done[qid]=rec
 finally: adapter.close()
 records=[done[q] for q in qids]; constrained=summary(records,'identity_constrained'); greedy_summary=summary(records,'identity_greedy_pope')
 patients=sorted({r['patient_id'] for r in records}); byp={p:[r for r in records if r['patient_id']==p] for p in patients}; rng=np.random.default_rng(20260727); boots=[]
 for _ in range(10000):
  sample=rng.choice(patients,len(patients),replace=True); selected=[r for p in sample for r in byp[p]]; b=np.mean([r['identity_greedy_pope']==r['ground_truth'] for r in selected]); n=np.mean([r['calibrated']==r['ground_truth'] for r in selected]); boots.append(100*(n-b))
 payload={'version':VERSION,'fingerprint':fp,'status':'final','method_frozen_before_target':True,'target_labels_used_for_tuning':False,'threshold':t,'n':len(records),'patients':len(patients),'interfaces':{'complete_sequence_identity':constrained,'rule_greedy_pope_baseline':greedy_summary},'patient_cluster_bootstrap_delta_ci95_pp':[float(x) for x in np.quantile(boots,[.025,.975])],'provenance':{'source_manifest_sha256':file_sha256(a.source_manifest),'source_gate_sha256':file_sha256(a.source_gate),'calibrator_sha256':file_sha256(a.calibrator),'questions_sha256':file_sha256(a.questions),'greedy_cache_sha256':file_sha256(a.greedy_cache),'raw_sha256':file_sha256(a.raw),'code_sha256':code_hash},'records':records}
 atomic_json(a.output,payload); print(json.dumps({k:payload[k] for k in ('fingerprint','n','patients','interfaces','patient_cluster_bootstrap_delta_ci95_pp')},indent=2))
if __name__=='__main__': main()
