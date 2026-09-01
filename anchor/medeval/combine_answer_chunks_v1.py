#!/usr/bin/env python3
"""Combine answer chunks only when they exactly match a frozen manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .hashing import sha256_file

def rid(row,i):return str(row.get('question_id',row.get('qid',row.get('id',i))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--answers',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads(a.manifest.read_text()); rows=[]
 for path in a.answers:rows.extend(json.loads(x) for x in path.read_text().splitlines() if x.strip())
 if [rid(x,i) for i,x in enumerate(rows)] != [rid(x,i) for i,x in enumerate(m)]:raise ValueError('combined answers are not an exact manifest sequence')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 audit={'protocol':'combine-answer-chunks-v1','manifest_sha256':sha256_file(a.manifest),'inputs':[{'path':str(x.resolve()),'sha256':sha256_file(x)} for x in a.answers],'output_sha256':sha256_file(a.output),'rows':len(rows)}
 a.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
