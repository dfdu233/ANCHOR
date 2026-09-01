#!/usr/bin/env python3
"""Add retrieved different-patient reports without changing answer contracts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .hashing import sha256_file

def read(path):
    text=path.read_text(); return json.loads(text) if path.suffix=='.json' else [json.loads(x) for x in text.splitlines() if x.strip()]

def retrieval_provenance(retrieval: Path) -> dict:
    """Describe the signal used to retrieve context without guessing from prompts."""
    candidates = (retrieval.parent / 'retrieval_manifest.json', retrieval.parent / 'manifest.json')
    for path in candidates:
        if not path.exists():
            continue
        metadata = read(path)
        query_signal = metadata.get('query_signal')
        if query_signal is None:
            query_signal = metadata.get('query_schema')
        if query_signal is not None:
            return {
                'query_field': str(query_signal).replace('_', ' '),
                'retrieval_manifest': str(path.resolve()),
                'retrieval_manifest_sha256': sha256_file(path),
            }
    return {'query_field': 'question only'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--queries',type=Path,required=True); p.add_argument('--retrieval',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    queries=read(a.queries); retrieved=read(a.retrieval); by={str(x['sample_id']):x for x in retrieved}; rows=[]; controls=[]
    for i,row in enumerate(queries):
        qid=str(row.get('qid',row.get('question_id',row.get('id',i)))); hit=by.get(qid)
        if hit is None: raise ValueError(f'missing retrieval for {qid}')
        context='\n'.join(f"[{d['rank']}] {d['report']}" for d in hit['documents'])
        question=("Use the medical image as primary evidence. The following reports are from different patients and may be irrelevant; do not copy unsupported findings.\n"
                  f"Retrieved reports:\n{context}\nQuestion:\n{row['question']}")
        out=dict(row); out['question']=question; out['source_question']=row['question']; out['context_condition']='retrieved_top3'; out['retrieved_doc_ids']=[d['doc_id'] for d in hit['documents']]; rows.append(out)
        control=dict(row); control['question']=("Use the medical image as primary evidence. The following reports are from different patients and may be irrelevant; do not copy unsupported findings.\n"
            f"Retrieved reports:\n[none]\nQuestion:\n{row['question']}"); control['source_question']=row['question']; control['context_condition']='none'; control['retrieved_doc_ids']=[]; controls.append(control)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    control_output=a.output.parent/'no_context.json'; control_output.write_text(json.dumps(controls,ensure_ascii=False,indent=2)+'\n')
    manifest={'protocol':'shared-medical-rag-prompts-v1','queries':str(a.queries.resolve()),'queries_sha256':sha256_file(a.queries),'retrieval':str(a.retrieval.resolve()),'retrieval_sha256':sha256_file(a.retrieval),'output':str(a.output.resolve()),'output_sha256':sha256_file(a.output),'no_context_output':str(control_output.resolve()),'no_context_sha256':sha256_file(control_output),'rows':len(rows),'matched_prompt_except_context':True,'reference_used_in_prompt':False,'top_k':3}
    manifest.update(retrieval_provenance(a.retrieval))
    (a.output.parent/'prompt_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
