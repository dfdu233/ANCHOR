#!/usr/bin/env python3
"""Train one shared DG residual from source risk and exact visual pairs."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from corrected_sgta.models_alignment import LlavaMedAlignmentAdapter
from corrected_sgta.rule_source_absolute_margin import absolute_margin_loss, binary_sign, select_worst_domain
from corrected_sgta.rule_source_exact_pair import (
    GRADIENT_CLIP, LEARNING_RATE, MAX_RELATIVE_UPDATE, PAIR_MANIFEST_VERSION,
    PAIR_WEIGHT, RANK, SEED, SOURCE_DOMAINS, STEPS, VERSION, canonical_label,
    canonical_question, exact_pair_experiment_fingerprint, pair_manifest_identity,
    reference_relative_pair_loss,
)
from corrected_sgta.rule_source_preference import (
    LinearLowRankResidual, file_sha256, rule_mimic_prompt,
    sequence_log_probability, stable_json_sha256, target_ids_from_labels,
    validate_source_manifest,
)
from corrected_sgta.train_rule_dg_adapter import (
    atomic_torch_save, build_teacher_forcing, canonical_answer,
    projector_output_width, sequence_forward,
)
from corrected_sgta.train_rule_source_group_adapter import (
    balanced_schedule, extract_question_only, load_rows, parse_named_paths,
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--pair-manifest',type=Path,required=True)
    p.add_argument('--source-json',action='append',required=True,metavar='DOMAIN=PATH')
    p.add_argument('--source-image-root',action='append',required=True,metavar='DOMAIN=PATH')
    p.add_argument('--locked-test',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-every',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    return p.parse_args()

def code_hashes():
    names=['rule_source_exact_pair.py','rule_source_absolute_margin.py','rule_source_preference.py','train_rule_dg_adapter.py','train_rule_source_group_adapter.py','models_alignment.py']
    paths=[Path(__file__).resolve()]+[Path(__file__).with_name(x) for x in names]
    return {str(p):file_sha256(p) for p in paths}

def candidate_logp(adapter,image,prompt,answer,module):
    ids,labels=build_teacher_forcing(adapter,prompt,answer)
    _,logits=sequence_forward(adapter,image,ids,labels,module,adapter_location='post')
    return sequence_log_probability(logits,target_ids_from_labels(labels))

def raw_margin(adapter,image,prompt,module):
    return candidate_logp(adapter,image,prompt,'Yes.',module)-candidate_logp(adapter,image,prompt,'No.',module)

def example_loss(adapter,image,prompt,answer,module):
    yes=candidate_logp(adapter,image,prompt,'Yes.',module)
    no=candidate_logp(adapter,image,prompt,'No.',module)
    loss,margin=absolute_margin_loss(yes,no,binary_sign(canonical_answer(answer)))
    return loss,margin,yes,no

def normalize_all(name,path,root):
    out=[]
    for raw in load_rows(path):
        conv=raw.get('conversations')
        if not isinstance(conv,list) or len(conv)!=2: raise ValueError(f'invalid conversations in {path}')
        image=(root/str(raw.get('image',''))).resolve()
        if not image.is_file(): raise FileNotFoundError(image)
        out.append({'domain':name,'id':str(raw.get('id','')),'image':str(image),
          'question':extract_question_only(conv[0].get('value','')),
          'answer':canonical_label(conv[1].get('value','')),
          'rgb_sha256':str(raw.get('image_sha256',''))})
    if not out: raise ValueError(f'empty source {name}')
    return sorted(out,key=lambda r:stable_json_sha256({'seed':SEED,**r}))

def validate_pairs(path,contract):
    data=json.loads(path.read_text()); fp=data.pop('fingerprint',None)
    if data.get('version')!=PAIR_MANIFEST_VERSION: raise ValueError('unsupported pair manifest')
    if fp!=stable_json_sha256(data): raise ValueError('pair manifest fingerprint mismatch')
    if data['source_manifest']['fingerprint']!=contract['manifest_fingerprint']: raise ValueError('pair/source manifest mismatch')
    if not data.get('pairs'): raise ValueError('empty pair manifest')
    for pair in data['pairs']:
        if canonical_question(pair['question'])!=pair['canonical_question']: raise ValueError('canonical question mismatch')
        if canonical_label(pair['positive']['label'])!='yes' or canonical_label(pair['negative']['label'])!='no': raise ValueError('non-opposite pair labels')
        for side in ('positive','negative'):
            row=pair[side]
            if file_sha256(Path(row['image']))!=row['file_sha256']: raise ValueError('pair image hash mismatch')
    return {'fingerprint':fp,**data}

def cpu_state(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def main():
    a=parse_args()
    if a.save_every<=0: raise ValueError('save-every must be positive')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    jsons=parse_named_paths(a.source_json,'--source-json'); roots=parse_named_paths(a.source_image_root,'--source-image-root')
    if set(jsons)!=set(roots) or len(jsons)!=SOURCE_DOMAINS: raise ValueError('exactly three matching source domains required')
    for p in [a.source_manifest,a.pair_manifest,a.locked_test,*jsons.values()]:
        if not p.is_file(): raise FileNotFoundError(p)
    contract=validate_source_manifest(a.source_manifest.resolve(),jsons,a.locked_test.resolve())
    pair_manifest=validate_pairs(a.pair_manifest,contract)
    groups={n:normalize_all(n,jsons[n],roots[n]) for n in sorted(jsons)}
    schedule=balanced_schedule(groups,STEPS); names=sorted(groups)
    selected={n:[{'id':r['id'],'image':r['image'],'question':r['question'],'answer':r['answer'],'rgb_sha256':r['rgb_sha256'],'file_sha256':file_sha256(Path(r['image']))} for r in groups[n]] for n in names}
    config={'source_manifest':str(a.source_manifest),'pair_manifest':str(a.pair_manifest),'source_json':{n:str(p) for n,p in sorted(jsons.items())},'source_image_root':{n:str(p) for n,p in sorted(roots.items())},'locked_test':str(a.locked_test),'objective':'max_domain_absolute_margin_plus_reference_relative_exact_pair','rank':RANK,'max_relative_update':MAX_RELATIVE_UPDATE,'steps':STEPS,'pair_weight':PAIR_WEIGHT,'learning_rate':LEARNING_RATE,'weight_decay':0.0,'gradient_clip':GRADIENT_CLIP,'seed':SEED}
    pair_contract=pair_manifest_identity(a.pair_manifest,pair_manifest); pair_contract['sha256']=file_sha256(a.pair_manifest)
    fp,fp_payload=exact_pair_experiment_fingerprint(manifest_contract=contract,pair_manifest=pair_contract,config=config,selected=selected,code_sha256=code_hashes())
    if a.output.exists() and not a.resume: raise FileExistsError('output exists; use --resume only for identical run')
    adapter=LlavaMedAlignmentAdapter(conv_mode='vicuna_v1')
    for p in adapter.model.parameters(): p.requires_grad_(False)
    adapter.model.eval(); width=projector_output_width(adapter.model)
    module=LinearLowRankResidual(width,RANK,MAX_RELATIVE_UPDATE).to(adapter.model.device)
    optimizer=torch.optim.AdamW(module.parameters(),lr=LEARNING_RATE,weight_decay=0.0)
    pair_schedule=[pair_manifest['pairs'][i%len(pair_manifest['pairs'])] for i in range(STEPS)]
    reference={}
    with torch.no_grad():
        for pair in tqdm(pair_manifest['pairs'],desc='frozen-pair-reference'):
            m={}
            for side in ('positive','negative'):
                row=pair[side]
                with Image.open(row['image']) as h: image=h.convert('RGB')
                m[side]=float(raw_margin(adapter,image,rule_mimic_prompt(row['question']),None))
            reference[pair['pair_id']]=m['positive']-m['negative']
    history=[]; start=0
    if a.resume and a.output.is_file():
        old=torch.load(a.output,map_location='cpu',weights_only=False)
        if old.get('version')!=VERSION or old.get('fingerprint')!=fp: raise RuntimeError('resume fingerprint mismatch')
        if old.get('reference_gaps')!=reference: raise RuntimeError('resume reference mismatch')
        module.load_state_dict(old['state_dict']); optimizer.load_state_dict(old['optimizer']); history=list(old['history']); start=int(old['next_step'])
    def save(next_step):
        atomic_torch_save({'version':VERSION,'fingerprint':fp,'fingerprint_payload':fp_payload,'manifest_contract':contract,'pair_manifest':pair_contract,'width':width,'rank':RANK,'max_relative_update':MAX_RELATIVE_UPDATE,'prompt_protocol':'rule_mimic','objective':'worst_absolute_margin_plus_reference_relative_exact_pair','pair_weight':PAIR_WEIGHT,'state_dict':cpu_state(module),'optimizer':optimizer.state_dict(),'history':history,'next_step':next_step,'reference_gaps':reference,'full_source_train_rgb_sha256':sorted({r['rgb_sha256'] for rs in groups.values() for r in rs}),'target_labels_accessed':False},a.output)
    progress=tqdm(range(start,STEPS),desc='source-exact-pair')
    try:
        for step in progress:
            batch={}; detached={}
            with torch.no_grad():
                for name in names:
                    row=schedule[step][name]
                    with Image.open(row['image']) as h: image=h.convert('RGB')
                    prompt=rule_mimic_prompt(row['question']); answer=row['answer']
                    detached[name]=example_loss(adapter,image,prompt,answer,module)[0].detach()
                    batch[name]=(image,prompt,answer)
            worst=select_worst_domain(detached); image,prompt,answer=batch[worst]
            optimizer.zero_grad(set_to_none=True)
            abs_loss,margin,yes,no=example_loss(adapter,image,prompt,answer,module); abs_loss.backward()
            pair=pair_schedule[step]; adapted={}
            with torch.no_grad():
                for side in ('positive','negative'):
                    row=pair[side]
                    with Image.open(row['image']) as h: pimage=h.convert('RGB')
                    adapted[side]=float(raw_margin(adapter,pimage,rule_mimic_prompt(row['question']),module))
            pair_loss,improvement=reference_relative_pair_loss(torch.tensor(adapted['positive'],device=adapter.model.device),torch.tensor(adapted['negative'],device=adapter.model.device),reference[pair['pair_id']])
            derivative=-torch.sigmoid(-improvement).detach()*PAIR_WEIGHT
            for side,sign in (('positive',1.0),('negative',-1.0)):
                row=pair[side]
                with Image.open(row['image']) as h: pimage=h.convert('RGB')
                raw_margin(adapter,pimage,rule_mimic_prompt(row['question']),module).backward(gradient=derivative*sign)
            grad=torch.nn.utils.clip_grad_norm_(module.parameters(),GRADIENT_CLIP)
            if not math.isfinite(float(grad)): raise FloatingPointError('non-finite gradient')
            optimizer.step(); total=float(abs_loss.detach()+PAIR_WEIGHT*pair_loss.detach())
            item={'step':step,'worst_domain':worst,'loss':total,'absolute_loss':float(abs_loss.detach()),'pair_loss':float(pair_loss.detach()),'pair_improvement':float(improvement.detach()),'pair_id':pair['pair_id'],'margin':float(margin.detach()),'yes_log_probability':float(yes.detach()),'no_log_probability':float(no.detach()),'gradient_norm':float(grad),'mean_relative_update':module.last_mean_relative_norm,'maximum_relative_update':module.last_max_relative_norm,'detached_domain_losses':{n:float(v) for n,v in detached.items()}}
            history.append(item); progress.set_postfix(loss=f'{total:.4f}',worst=worst)
            if (step+1)%a.save_every==0 or step+1==STEPS: save(step+1)
            del batch,detached; torch.cuda.empty_cache()
    finally: adapter.close()
    print(json.dumps({'output':str(a.output),'version':VERSION,'fingerprint':fp,'pairs':len(pair_manifest['pairs']),'source_sizes':{n:len(groups[n]) for n in names},'steps_complete':len(history),'target_labels_accessed':False},indent=2))
if __name__=='__main__': main()
