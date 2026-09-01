#!/usr/bin/env python3
"""Combine already decontaminated corpora with a hash-closed manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .hashing import sha256_file,sha256_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',type=Path,nargs='+',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    rows=[]; seen=set()
    for source in a.corpus:
        for line in source.read_text().splitlines():
            if not line.strip(): continue
            row=json.loads(line); key=str(row['doc_id'])
            if key in seen: raise ValueError(f'duplicate doc_id: {key}')
            seen.add(key); rows.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/'corpus.jsonl'
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'protocol':'combined-decontaminated-rag-corpus-v1','sources':[{'path':str(x.resolve()),'sha256':sha256_file(x)} for x in a.corpus],'rows':len(rows),'ordered_rows_sha256':sha256_json(rows),'corpus':str(out.resolve()),'corpus_sha256':sha256_file(out)}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
