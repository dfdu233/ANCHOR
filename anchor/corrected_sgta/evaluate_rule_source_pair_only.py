#!/usr/bin/env python3
"""Evaluate the pair-only residual once on the frozen disjoint 85-source-dev gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import numpy as np, torch
from PIL import Image
from tqdm import tqdm
from corrected_sgta.evaluate_rule_source_adapter_nll import atomic_json
from corrected_sgta.evaluate_rule_source_preference_barycenter import _module_from_state,_predict,summarize_predictions
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_pair_only import DEV_EVAL_VERSION,DEV_IMAGES_TOTAL,VERSION as TRAIN_VERSION,dev_gate
from corrected_sgta.rule_source_preference import canonical_binary_answer,file_sha256,rule_mimic_prompt,stable_json_sha256
from corrected_sgta.train_rule_dg_adapter import canonical_answer
from corrected_sgta.train_rule_source_group_adapter import load_rows,normalize_source_rows,parse_named_paths

def args():
 p=argparse.ArgumentParser(); p.add_argument('--source-dev-manifest',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--dev-json',action='append',required=True,metavar='DOMAIN=PATH'); p.add_argument('--dev-image-root',action='append',required=True,metavar='DOMAIN=PATH'); p.add_argument('--output',type=Path,required=True); return p.parse_args()

def validate_dev(manifest_path,checkpoint,dev_jsons):
 m=json.loads(manifest_path.read_text())
 if m.get('version')!='rule-source-manifest-v2': raise ValueError('unsupported source-dev manifest')
 declared=m.get('outputs',{}).get('by_domain',{})
 if set(dev_jsons)!=set(declared): raise ValueError('dev domains mismatch')
 hashes={}
 for d,p in sorted(dev_jsons.items()):
  actual=file_sha256(p)
  if actual!=declared[d]['dev']['json_sha256']: raise ValueError(f'dev hash mismatch: {d}')
  hashes[d]=actual
 dev_rgb={str(r.get('image_sha256','')) for p in dev_jsons.values() for r in load_rows(p)}
 train_rgb=set(checkpoint.get('full_source_train_rgb_sha256',[]))
 overlap=dev_rgb&train_rgb
 if overlap: raise ValueError(f'source train/dev RGB leakage: {len(overlap)}')
 if m.get('locked_test',{}).get('labels_read_for_selection') is not False: raise ValueError('target labels not sealed')
 return {'manifest_fingerprint':m['fingerprint'],'manifest_sha256':file_sha256(manifest_path),'dev_json_sha256':hashes,'expanded_train_dev_rgb_overlap':0,'target_labels_read_for_selection':False}

def main():
 a=args()
 if a.output.exists(): raise FileExistsError(a.output)
 jsons=parse_named_paths(a.dev_json,'--dev-json'); roots=parse_named_paths(a.dev_image_root,'--dev-image-root')
 if set(jsons)!=set(roots) or len(jsons)!=3: raise ValueError('exactly three dev domains required')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if ck.get('version')!=TRAIN_VERSION or ck.get('target_labels_accessed') is not False: raise ValueError('unsupported/unsealed checkpoint')
 contract=validate_dev(a.source_dev_manifest,ck,jsons)
 pair_path=Path(ck['pair_manifest']['path']); pair_sha=file_sha256(pair_path)
 if pair_sha!=ck['pair_manifest']['sha256']: raise ValueError('pair manifest hash mismatch')
 pair_payload=json.loads(pair_path.read_text())
 selected={d:normalize_source_rows(d,jsons[d],roots[d],0,42) for d in sorted(jsons)}
 if sum(map(len,selected.values()))!=DEV_IMAGES_TOTAL: raise ValueError('frozen dev must contain exactly 85 examples')
 fp_payload={'version':DEV_EVAL_VERSION,'checkpoint_sha256':file_sha256(a.checkpoint),'checkpoint_fingerprint':ck['fingerprint'],'dev_contract':contract,'selected':{d:[{'id':r['id'],'image':r['image'],'image_sha256':file_sha256(Path(r['image']))} for r in rs] for d,rs in selected.items()},'prediction_interface':'argmax_complete_yes_no_sequence_log_probability','target_test_labels_accessed':False}
 fp=stable_json_sha256(fp_payload)
 adapter=LlavaMedAlignmentAdapter(conv_mode='vicuna_v1')
 for p in adapter.model.parameters(): p.requires_grad_(False)
 adapter.model.eval(); module=_module_from_state(ck['state_dict'],int(ck['width']),int(ck['rank']),float(ck['max_relative_update']),adapter.model.device)
 records={d:[] for d in selected}; deltas=[]; pair_common=[]; pair_audit=[]
 try:
  with torch.no_grad():
   for d in sorted(selected):
    for r in tqdm(selected[d],desc=f'pair-only-dev:{d}'):
     with Image.open(r['image']) as h: image=h.convert('RGB')
     prompt=rule_mimic_prompt(r['question']); gt=canonical_binary_answer(canonical_answer(r['answer']))
     identity,iscores=_predict(adapter,image,prompt,None); adapted,ascores=_predict(adapter,image,prompt,module)
     delta=(ascores['Yes.']-ascores['No.'])-(iscores['Yes.']-iscores['No.']); deltas.append(delta)
     records[d].append({'id':r['id'],'image':r['image'],'ground_truth':gt,'prompt':prompt,'predictions':{'identity':identity,'source_pair_only':adapted},'sequence_log_probabilities':{'identity':iscores,'source_pair_only':ascores},'yes_minus_no_margin_delta':delta})
   for pair in tqdm(pair_payload['pairs'],desc='pair-common-mode-audit'):
    shifts={}
    for side in ('positive','negative'):
     row=pair[side]
     with Image.open(row['image']) as h: image=h.convert('RGB')
     _,scores=_predict(adapter,image,rule_mimic_prompt(row['question']),module)
     adapted_margin=scores['Yes.']-scores['No.']; reference=ck['reference_margins'][pair['pair_id']][side]; shifts[side]=adapted_margin-reference
    common=0.5*(shifts['positive']+shifts['negative']); pair_common.append(common); pair_audit.append({'pair_id':pair['pair_id'],'positive_margin_delta':shifts['positive'],'negative_margin_delta':shifts['negative'],'common_mode_drift':common,'differential_drift':shifts['positive']-shifts['negative']})
 finally: adapter.close()
 arr=np.asarray(deltas); centered_rms=float(np.sqrt(np.mean((arr-arr.mean())**2))); common=np.asarray(pair_common); diag={'n':len(deltas),'positive_delta_count':int((arr>1e-12).sum()),'negative_delta_count':int((arr< -1e-12).sum()),'zero_delta_count':int((abs(arr)<=1e-12).sum()),'mean_delta':float(arr.mean()),'centered_rms':centered_rms,'absolute_mean_to_centered_rms':float(abs(arr.mean())/centered_rms) if centered_rms>0 else float('inf'),'minimum_delta':float(arr.min()),'maximum_delta':float(arr.max()),'common_mode_pairs':len(pair_common),'common_mode_negative_count':int((common< -1e-12).sum()),'common_mode_negative_rate':float((common< -1e-12).mean()),'common_mode_mean':float(common.mean()),'common_mode_minimum':float(common.min()),'common_mode_maximum':float(common.max())}
 summary=summarize_predictions(records,['source_pair_only']); gate=dev_gate(summary,diag)
 atomic_json(a.output,{'version':DEV_EVAL_VERSION,'fingerprint':fp,'fingerprint_payload':fp_payload,'checkpoint':{'path':str(a.checkpoint),'sha256':file_sha256(a.checkpoint),'version':ck['version'],'fingerprint':ck['fingerprint']},'summary':summary,'margin_diagnostic':diag,'pair_common_mode_audit':pair_audit,'source_dev_gate':gate,'records':records,'target_test_labels_accessed':False})
 print(json.dumps({'output':str(a.output),'summary':summary,'margin_diagnostic':diag,'source_dev_gate':gate},indent=2))
if __name__=='__main__': main()
