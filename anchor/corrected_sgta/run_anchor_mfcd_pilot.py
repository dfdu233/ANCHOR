#!/usr/bin/env python3
"""ANCHOR-MFCD pilot: source-frequency orbit contrastive decoding.

Training-free full-sentence decoding.  It does not use canonical yes/no logits as
predictions.  SGTA/FedDG views define a style-induced distribution to subtract
from the original distribution during autoregressive generation.
"""
from __future__ import annotations

import argparse, json, hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.evaluate_medheval_answers import evaluate_rows, rule_pope_prediction
from corrected_sgta.infer_ce import resize_image
from corrected_sgta.models_oe import LlavaMedOEAdapter, Generation
from corrected_sgta.run_anchor_flow_sgta_gate import DEFAULT_CENTER, make_views, normalize_target_record

ImageFile.LOAD_TRUNCATED_IMAGES = True
VERSION='anchor-mfcd-pilot-v1'

def load_json_or_jsonl(path: Path):
    text=path.read_text()
    if path.suffix=='.jsonl': return [json.loads(l) for l in text.splitlines() if l.strip()]
    v=json.loads(text)
    if isinstance(v,list): return v
    for k in ('records','data','questions','samples'):
        if isinstance(v.get(k),list): return v[k]
    raise ValueError(path)

def normalize_rows(path: Path, max_samples:int):
    rows=[]
    for row in load_json_or_jsonl(path):
        if 'prompt' in row and 'image' in row and ('answer' in row or 'reference' in row):
            rows.append({'id':str(row.get('id')), 'image':str(row.get('image')), 'prompt':str(row.get('prompt')).replace('<image>','').strip(), 'answer':str(row.get('answer',row.get('reference'))), 'patient_id':str(row.get('patient_id',row.get('id'))), 'raw':row})
        else:
            rows.append(normalize_target_record(row, task='ce', default_domain='mimic', default_prompt='', require_answer=True))
        if max_samples and len(rows)>=max_samples: break
    return rows

def parse_ce(text, reference, prompt):
    detail=evaluate_rows([{'qid':'sample','question':prompt,'ground_truth':reference,'text':text,'question_type':'binary'}])['details'][0]
    pred=detail.get('prediction'); gt=detail.get('ground_truth')
    if pred is None:
        pred=rule_pope_prediction(text); gt=rule_pope_prediction(reference)
    return {'parsed_answer':pred,'ground_truth':gt,'correct':bool(pred is not None and gt is not None and pred==gt),'parseable':pred is not None}

def geometric_mean_logp(logps: torch.Tensor) -> torch.Tensor:
    # [weak_views, vocab] -> normalized geometric mean log-prob
    out=logps.mean(dim=0)
    return out - torch.logsumexp(out, dim=-1)

@torch.inference_mode()
def generate_mfcd(adapter: LlavaMedOEAdapter, images: list[Image.Image], prompt: str, alpha: float, max_new_tokens:int) -> Generation:
    count=len(images)
    input_ids=adapter._prompt_ids(prompt).repeat(count,1).to(adapter.model.device)
    image_tensor=adapter._process_images(images)
    if isinstance(image_tensor,list): image_tensor=[x.to(adapter.model.device,dtype=adapter.model.dtype) for x in image_tensor]
    else: image_tensor=image_tensor.to(adapter.model.device,dtype=adapter.model.dtype)
    _,pos,mask,_,embeds,_=adapter.model.prepare_inputs_labels_for_multimodal(input_ids,None,None,None,None,image_tensor,image_sizes=[im.size for im in images])
    out=adapter.model.model(input_ids=None,attention_mask=mask,position_ids=pos,inputs_embeds=embeds,use_cache=True,return_dict=True)
    past=out.past_key_values; hidden=out.last_hidden_state[:,-1]
    weight=adapter.model.get_output_embeddings().weight
    eos=adapter.tokenizer.eos_token_id
    generated=[]; nll=[]; js=[]
    for step in range(max_new_tokens):
        logits=hidden.to(weight.dtype) @ weight.T
        logp=torch.log_softmax(logits.float(),dim=-1)
        orig=logp[0]
        weak=geometric_mean_logp(logp[1:]) if count>1 else orig
        score=orig + alpha*(orig-weak)
        score=score - torch.logsumexp(score,dim=-1)
        token=int(score.argmax().item())
        nll.append(float(-score[token].item()))
        probs=logp.exp(); mean=probs.mean(0).clamp_min(1e-12).log()
        js.append(float((probs*(logp-mean.unsqueeze(0))).sum(-1).mean().clamp_min(0).item()))
        if eos is not None and token==eos: break
        generated.append(token)
        if step+1==max_new_tokens: break
        token_ids=torch.full((count,1),token,dtype=torch.long,device=adapter.model.device)
        past_len=int(past.get_seq_length() if hasattr(past,'get_seq_length') else past[0][0].shape[-2])
        next_mask=torch.ones((count,past_len+1),dtype=torch.long,device=adapter.model.device)
        next_pos=torch.full((count,1),past_len,dtype=torch.long,device=adapter.model.device)
        out=adapter.model.model(input_ids=token_ids,attention_mask=next_mask,position_ids=next_pos,past_key_values=past,use_cache=True,return_dict=True)
        past=out.past_key_values; hidden=out.last_hidden_state[:,-1]
    text=adapter.tokenizer.decode(generated,skip_special_tokens=True).strip()
    return Generation(text=text, uncertainty=float(np.mean(nll)) if nll else float('inf'), token_count=len(generated))

def summarize(records, alphas):
    base=[r['methods']['alpha_0.0']['eval']['correct'] for r in records]
    out={'version':VERSION,'n':len(records),'methods':{}}
    for a in alphas:
        k=f'alpha_{a}'
        vals=[r['methods'][k]['eval']['correct'] for r in records]
        parse=[r['methods'][k]['eval']['parseable'] for r in records]
        out['methods'][k]={'accuracy':float(np.mean(vals)) if vals else 0.0,'parse_rate':float(np.mean(parse)) if parse else 0.0,'delta_vs_greedy':float(np.mean(vals)-np.mean(base)) if vals else 0.0,'rescue':int(sum((not b) and v for b,v in zip(base,vals))),'harm':int(sum(b and (not v) for b,v in zip(base,vals))),'mean_words':float(np.mean([len(r['methods'][k]['text'].split()) for r in records])) if records else 0.0}
    out['best_alpha_by_accuracy']=max(out['methods'], key=lambda k:(out['methods'][k]['accuracy'],-out['methods'][k]['harm'])) if records else None
    out['continue_gate']=bool(any(m['delta_vs_greedy']>=0.03 and m['rescue']>m['harm'] for k,m in out['methods'].items() if k!='alpha_0.0'))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=Path('corrected_runs/final_anchor_riemann_gate_v1/manifests/mimic_ce.json'))
    ap.add_argument('--image-root',type=Path,default=Path('/root/autodl-tmp/MedHEval/images'))
    ap.add_argument('--output-dir',type=Path,default=Path('corrected_runs/final_anchor_mfcd_pilot_v1'))
    ap.add_argument('--max-samples',type=int,default=16)
    ap.add_argument('--max-new-tokens',type=int,default=48)
    ap.add_argument('--max-image-side',type=int,default=384)
    ap.add_argument('--alphas',type=float,nargs='+',default=[0.0,0.25,0.5,1.0])
    ap.add_argument('--spectrum-alpha',type=float,default=0.05)
    ap.add_argument('--low-frequency-ratio',type=float,default=0.03)
    ap.add_argument('--gamma',type=float,default=0.8)
    ap.add_argument('--center',type=Path,default=DEFAULT_CENTER)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw_path=args.output_dir/'ce_raw.jsonl'; summary_path=args.output_dir/'ce_summary.json'
    rows=normalize_rows(args.manifest,args.max_samples)
    center=np.load(args.center)
    fp=stable_json_sha256({'version':VERSION,'manifest':str(args.manifest),'image_root':str(args.image_root),'max_samples':args.max_samples,'alphas':args.alphas,'views':[args.spectrum_alpha,args.low_frequency_ratio,args.gamma],'no_yes_no_logits_as_results':True})
    adapter=LlavaMedOEAdapter(conv_mode='mistral_instruct')
    records=[]
    with raw_path.open('w') as f:
        for row in tqdm(rows,desc='ANCHOR-MFCD CE'):
            image_path=resolve_image_path(row['image'],args.image_root)
            with Image.open(image_path) as src: image=resize_image(src.convert('RGB'),args.max_image_side)
            views=make_views(image, center=center, max_side=args.max_image_side, spectrum_alpha=args.spectrum_alpha, low_frequency_ratio=args.low_frequency_ratio, gamma=args.gamma)
            images=[v['image'] for v in views]
            methods={}
            for a in args.alphas:
                gen=generate_mfcd(adapter,images,row['prompt'],float(a),args.max_new_tokens)
                methods[f'alpha_{a}']={'text':gen.text,'token_count':gen.token_count,'uncertainty':gen.uncertainty,'eval':parse_ce(gen.text,row['answer'],row['prompt'])}
            rec={'version':VERSION,'fingerprint':fp,'id':row['id'],'patient_id':row.get('patient_id'),'image':row['image'],'prompt':row['prompt'],'reference':row['answer'],'methods':methods,'target_labels_used_for_generation':False,'uses_yes_no_logits_for_prediction':False,'views':[v['name'] for v in views]}
            records.append(rec); f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush()
    summary=summarize(records,args.alphas); summary['fingerprint']=fp; summary['raw']=str(raw_path)
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
