#!/usr/bin/env python3
"""Evaluate a frozen source-robust margin threshold on untouched sources."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.evaluate_rule_source_preference_barycenter import _predict,summarize_predictions
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_preference import canonical_binary_answer,file_sha256,rule_mimic_prompt,stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import canonical_answer
VERSION="rule-source-robust-margin-eval-v1"
def exact_p(r,h):
 n=r+h
 return 1.0 if n==0 else min(1.0,2*sum(math.comb(n,k) for k in range(min(r,h)+1))/(2**n))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--calibrator',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 m=json.loads(a.manifest.read_text()); c=json.loads(a.calibrator.read_text())
 if m.get('version') not in {'rule-source-reconfirm-manifest-v1','rule-source-scale-manifest-v1'} or m.get('target_file_opened') is not False: raise ValueError('invalid untouched-source manifest')
 if c.get('version')!='rule-source-robust-margin-v1' or c.get('target_labels_used') is not False: raise ValueError('invalid calibrator')
 t=float(c['fit']['threshold']); domains=tuple(m['domains']); selected={d:json.loads(Path(m['outputs'][d]['json']).read_text()) for d in domains}
 if any(len(rows)!=m['images_per_domain'] for rows in selected.values()): raise ValueError('incomplete split')
 records={d:[] for d in domains}; adapter=LlavaMedAlignmentAdapter(conv_mode='vicuna_v1')
 for p in adapter.model.parameters(): p.requires_grad_(False)
 adapter.model.eval()
 try:
  for d in domains:
   for row in tqdm(selected[d],desc=f'robust-margin:{d}'):
    with Image.open(row['image']) as h: image=h.convert('RGB')
    q=str(row['conversations'][0]['value']).replace('<image>','').strip(); prompt=rule_mimic_prompt(q); gt=canonical_binary_answer(canonical_answer(row['conversations'][1]['value']))
    _,scores=_predict(adapter,image,prompt,None); margin=scores['Yes.']-scores['No.']; base='Yes.' if margin>=0 else 'No.'; calibrated='Yes.' if margin>=t else 'No.'
    records[d].append({'id':row['id'],'image':row['image'],'ground_truth':gt,'margin':margin,'threshold':t,'predictions':{'identity':base,'source_dro':calibrated},'sequence_log_probabilities':scores,'generated_answer_prefix':calibrated})
 finally: adapter.close()
 summary=summarize_predictions(records,['source_dro']); micro=summary['source_dro']['micro']; p=exact_p(int(micro['rescues']),int(micro['harms']))
 deltas=np.array([int(r['predictions']['source_dro']==r['ground_truth'])-int(r['predictions']['identity']==r['ground_truth']) for rows in records.values() for r in rows],dtype=float); rng=np.random.default_rng(20260726); boots=np.array([rng.choice(deltas,len(deltas),replace=True).mean() for _ in range(10000)])*100
 alpha=float(m.get('sequential_alpha',0.05)); passed=micro['delta_pp']>=3 and p<alpha and micro['rescues']>=2*micro['harms']
 gate={'status':'passed' if passed else 'failed','target_pilot_allowed':bool(passed),'alpha':alpha,'checks':{'delta_at_least_3pp':micro['delta_pp']>=3,'mcnemar_p_below_alpha':p<alpha,'rescue_harm_ratio_at_least_2':micro['rescues']>=2*micro['harms']},'mcnemar_exact_p':p}
 payload={'version':VERSION,'fingerprint':stable_json_sha256({'version':VERSION,'manifest':file_sha256(a.manifest),'calibrator':file_sha256(a.calibrator)}),'target_labels_used':False,'threshold':t,'interface':'complete-sequence margin with explicit answer-prefix generation','summary':summary,'paired_bootstrap_ci95_pp':[float(x) for x in np.quantile(boots,[.025,.975])],'gate':gate,'records':records}; atomic_json(a.output,payload); print(json.dumps({k:payload[k] for k in ('fingerprint','threshold','summary','paired_bootstrap_ci95_pp','gate')},indent=2))
if __name__=='__main__': main()
