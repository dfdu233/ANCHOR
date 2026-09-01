#!/usr/bin/env python3
"""Convert a frozen unified manifest to the official LLaVA JSONL schema."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .hashing import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=32);a=p.parse_args()
 rows=json.loads(a.manifest.read_text())[:a.limit]; out=[]
 for i,row in enumerate(rows):
  out.append({'question_id':str(row.get('qid',row.get('id',i))),'image':str(row['img_name']),'text':str(row['question'])})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x)+'\n' for x in out))
 audit={'protocol':'official-llava-t2-manifest-v1','source':str(a.manifest.resolve()),'source_sha256':sha256_file(a.manifest),'output':str(a.output.resolve()),'output_sha256':sha256_file(a.output),'rows':len(out),'prompt_transform':'none'}
 a.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
