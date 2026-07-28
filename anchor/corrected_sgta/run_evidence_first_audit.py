#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np
from tqdm import tqdm
from corrected_sgta.anchor_transport import resolve_image_path, stable_json_sha256
from corrected_sgta.evaluate_medheval_answers import evaluate_rows, rule_pope_prediction
from corrected_sgta.models_oe import LlavaMedOEAdapter

def load_rows(path: Path, max_samples: int):
    payload=json.loads(path.read_text())
    rows=payload if isinstance(payload,list) else payload.get('records',[])
    return rows[:max_samples]

def parse_ce(text: str, reference: str, question: str) -> dict[str, Any]:
    detail=evaluate_rows([{'qid':'sample','question':question,'ground_truth':reference,'text':text,'question_type':'binary'}])['details'][0]
    pred=detail.get('prediction') or rule_pope_prediction(text)
    gt=detail.get('ground_truth') or rule_pope_prediction(reference)
    return {'prediction':pred,'ground_truth':gt,'correct':bool(pred is not None and gt is not None and pred==gt),'parseable':pred is not None}

def atomic_json(path: Path, payload: dict[str,Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--image-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('corrected_runs/final_evidence_first_audit_v1'))
    ap.add_argument('--max-samples',type=int,default=32)
    ap.add_argument('--max-evidence-tokens',type=int,default=64)
    ap.add_argument('--max-answer-tokens',type=int,default=32)
    ap.add_argument('--seed',type=int,default=20260728)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw=args.output_dir/'raw.jsonl'; summary=args.output_dir/'summary.json'
    rows=load_rows(args.manifest,args.max_samples)
    fp=stable_json_sha256({'version':'evidence-first-audit-v1','manifest':str(args.manifest),'max_samples':args.max_samples,'prompt_family':'visual_evidence_then_answer','uses_yes_no_logits':False})
    adapter=LlavaMedOEAdapter(conv_mode='mistral_instruct')
    records=[]
    with raw.open('w') as h:
        for i,row in enumerate(tqdm(rows,desc='evidence-first audit')):
            question=str(row.get('question',row.get('prompt',''))).replace('<image>','').strip()
            answer=str(row.get('answer',row.get('reference','')))
            image_path=resolve_image_path(str(row['image']),args.image_root)
            with Image.open(image_path) as src:
                image=src.convert('RGB')
            direct=adapter._generate_once(image,question,1,False,0.0,1.0,args.max_answer_tokens,args.seed+i)[0]
            evidence_prompt=(
                'Inspect the chest X-ray carefully. List only visual findings relevant to this question.\n'
                f'Question: {question}\n'
                'Relevant visual evidence:'
            )
            evidence=adapter._generate_once(image,evidence_prompt,1,False,0.0,1.0,args.max_evidence_tokens,args.seed+i)[0]
            answer_prompt=(
                f'Question: {question}\n'
                f'Visual evidence from the image: {evidence.text}\n'
                'Based only on the visual evidence, answer the question in one complete sentence.'
            )
            chained=adapter._generate_once(image,answer_prompt,1,False,0.0,1.0,args.max_answer_tokens,args.seed+i)[0]
            rec={'version':'evidence-first-audit-v1','fingerprint':fp,'id':row.get('id'), 'patient_id':row.get('patient_id'), 'image':row['image'], 'question':question, 'reference':answer,
                 'direct_text':direct.text,'evidence_text':evidence.text,'chained_text':chained.text,
                 'direct_eval':parse_ce(direct.text,answer,question),'chained_eval':parse_ce(chained.text,answer,question),
                 'target_labels_used_for_generation':False,'uses_yes_no_logits_for_prediction':False}
            rec['outcome']='rescue' if rec['chained_eval']['correct'] and not rec['direct_eval']['correct'] else 'harm' if rec['direct_eval']['correct'] and not rec['chained_eval']['correct'] else 'unchanged'
            records.append(rec); h.write(json.dumps(rec,ensure_ascii=False)+'\n'); h.flush()
    base=[r['direct_eval']['correct'] for r in records]; vals=[r['chained_eval']['correct'] for r in records]
    out={'version':'evidence-first-audit-v1','fingerprint':fp,'n':len(records),'direct_accuracy':float(np.mean(base)) if records else 0.0,'chained_accuracy':float(np.mean(vals)) if records else 0.0,'delta':float(np.mean(vals)-np.mean(base)) if records else 0.0,'rescue':sum((not b) and v for b,v in zip(base,vals)),'harm':sum(b and (not v) for b,v in zip(base,vals)),'raw':str(raw),'continue_gate':bool(records and np.mean(vals)-np.mean(base)>=0.05 and sum((not b) and v for b,v in zip(base,vals))>sum(b and (not v) for b,v in zip(base,vals)))}
    atomic_json(summary,out); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
