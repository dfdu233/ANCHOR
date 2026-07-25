#!/usr/bin/env python3
"""Train the preregistered pair-only shared post-projector residual."""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
from typing import Any
import numpy as np,torch
from PIL import Image
from tqdm import tqdm
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_pair_only import (
 GRADIENT_CLIP,LEARNING_RATE,MAX_RELATIVE_UPDATE,RANK,SEED,STEPS,VERSION,
 experiment_fingerprint,pair_manifest_identity,reference_relative_pair_loss,
)
from corrected_sgta.rule_source_preference import LinearLowRankResidual,file_sha256,rule_mimic_prompt,validate_source_manifest
from corrected_sgta.train_rule_dg_adapter import atomic_torch_save,projector_output_width
from corrected_sgta.train_rule_source_exact_pair import cpu_state,raw_margin,validate_pairs
from corrected_sgta.train_rule_source_group_adapter import load_rows,parse_named_paths

def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--pair-manifest',type=Path,required=True);p.add_argument('--source-json',action='append',required=True,metavar='DOMAIN=PATH');p.add_argument('--locked-test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--save-every',type=int,default=8);p.add_argument('--resume',action='store_true');return p.parse_args()

def hashes():
 names=['rule_source_pair_only.py','rule_source_exact_pair.py','train_rule_source_exact_pair.py','rule_source_preference.py','train_rule_dg_adapter.py','models_alignment.py'];paths=[Path(__file__).resolve()]+[Path(__file__).with_name(n) for n in names];return {str(p):file_sha256(p) for p in paths}

def main():
 a=parse_args();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 if a.save_every<=0:raise ValueError('save-every must be positive')
 jsons=parse_named_paths(a.source_json,'--source-json')
 for p in [a.source_manifest,a.pair_manifest,a.locked_test,*jsons.values()]:
  if not p.is_file():raise FileNotFoundError(p)
 contract=validate_source_manifest(a.source_manifest.resolve(),jsons,a.locked_test.resolve());pairs=validate_pairs(a.pair_manifest,contract)
 pair_contract=pair_manifest_identity(a.pair_manifest,pairs);pair_contract['sha256']=file_sha256(a.pair_manifest)
 config={'objective':'reference_relative_exact_pair_only','rank':RANK,'max_relative_update':MAX_RELATIVE_UPDATE,'steps':STEPS,'learning_rate':LEARNING_RATE,'weight_decay':0.0,'gradient_clip':GRADIENT_CLIP,'seed':SEED,'source_manifest':str(a.source_manifest),'pair_manifest':str(a.pair_manifest),'locked_test':str(a.locked_test),'single_change_from_exact_pair':'remove_absolute_DRO'}
 fp,fp_payload=experiment_fingerprint(manifest_contract=contract,pair_manifest=pair_contract,config=config,code_sha256=hashes())
 if a.output.exists() and not a.resume:raise FileExistsError('output exists; use --resume only for identical run')
 train_rgb=sorted({str(r.get('image_sha256','')) for p in jsons.values() for r in load_rows(p)})
 adapter=LlavaMedAlignmentAdapter(conv_mode='vicuna_v1')
 for p in adapter.model.parameters():p.requires_grad_(False)
 adapter.model.eval();width=projector_output_width(adapter.model);module=LinearLowRankResidual(width,RANK,MAX_RELATIVE_UPDATE).to(adapter.model.device);opt=torch.optim.AdamW(module.parameters(),lr=LEARNING_RATE,weight_decay=0.0)
 reference={}
 with torch.no_grad():
  for pair in tqdm(pairs['pairs'],desc='frozen-pair-reference'):
   shifts={}
   for side in ('positive','negative'):
    row=pair[side]
    with Image.open(row['image']) as h:image=h.convert('RGB')
    shifts[side]=float(raw_margin(adapter,image,rule_mimic_prompt(row['question']),None))
   reference[pair['pair_id']]={'positive':shifts['positive'],'negative':shifts['negative'],'gap':shifts['positive']-shifts['negative']}
 history=[];start=0
 if a.resume and a.output.is_file():
  old=torch.load(a.output,map_location='cpu',weights_only=False)
  if old.get('version')!=VERSION or old.get('fingerprint')!=fp or old.get('reference_margins')!=reference:raise RuntimeError('resume contract mismatch')
  module.load_state_dict(old['state_dict']);opt.load_state_dict(old['optimizer']);history=list(old['history']);start=int(old['next_step'])
 def save(n):
  atomic_torch_save({'version':VERSION,'fingerprint':fp,'fingerprint_payload':fp_payload,'manifest_contract':contract,'pair_manifest':pair_contract,'width':width,'rank':RANK,'max_relative_update':MAX_RELATIVE_UPDATE,'prompt_protocol':'rule_mimic','objective':'reference_relative_exact_pair_only','state_dict':cpu_state(module),'optimizer':opt.state_dict(),'history':history,'next_step':n,'reference_margins':reference,'full_source_train_rgb_sha256':train_rgb,'target_labels_accessed':False},a.output)
 schedule=[pairs['pairs'][i%len(pairs['pairs'])] for i in range(STEPS)];progress=tqdm(range(start,STEPS),desc='source-pair-only')
 try:
  for step in progress:
   pair=schedule[step];adapted={}
   with torch.no_grad():
    for side in ('positive','negative'):
     row=pair[side]
     with Image.open(row['image']) as h:image=h.convert('RGB')
     adapted[side]=float(raw_margin(adapter,image,rule_mimic_prompt(row['question']),module))
   ref=reference[pair['pair_id']];loss,improvement=reference_relative_pair_loss(torch.tensor(adapted['positive'],device=adapter.model.device),torch.tensor(adapted['negative'],device=adapter.model.device),ref['gap']);derivative=-torch.sigmoid(-improvement).detach();opt.zero_grad(set_to_none=True)
   for side,sign in (('positive',1.0),('negative',-1.0)):
    row=pair[side]
    with Image.open(row['image']) as h:image=h.convert('RGB')
    raw_margin(adapter,image,rule_mimic_prompt(row['question']),module).backward(gradient=derivative*sign)
   grad=torch.nn.utils.clip_grad_norm_(module.parameters(),GRADIENT_CLIP)
   if not math.isfinite(float(grad)):raise FloatingPointError('non-finite gradient')
   opt.step();dp=adapted['positive']-ref['positive'];dn=adapted['negative']-ref['negative']
   item={'step':step,'pair_id':pair['pair_id'],'loss':float(loss),'pair_improvement':float(improvement),'positive_margin_delta':dp,'negative_margin_delta':dn,'pair_common_mode_drift':0.5*(dp+dn),'pair_differential_drift':dp-dn,'gradient_norm':float(grad),'mean_relative_update':module.last_mean_relative_norm,'maximum_relative_update':module.last_max_relative_norm}
   history.append(item);progress.set_postfix(loss=f'{item["loss"]:.4f}',diff=f'{item["pair_differential_drift"]:.3f}')
   if (step+1)%a.save_every==0 or step+1==STEPS:save(step+1)
   torch.cuda.empty_cache()
 finally:adapter.close()
 print(json.dumps({'output':str(a.output),'version':VERSION,'fingerprint':fp,'pairs':len(pairs['pairs']),'steps_complete':len(history),'target_labels_accessed':False},indent=2))
if __name__=='__main__':main()
